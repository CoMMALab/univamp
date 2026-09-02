"""VAMP-RRTC trajectory seeder for cuRobo.

cuRobo's MotionPlanner seeds TrajOpt by querying its own graph planner
(`_get_graph_seed_trajectories`) for a collision-free C-space path, then linearly
interpolating it to the trajopt horizon. VAMP-RRTC produces the same kind of path far
faster. `VampSeeder` is a drop-in replacement; `attach_vamp_seeder` monkeypatches a live
MotionPlanner additively (original method preserved for fallback).

The seed trajectories are written into a *managed* CUDA buffer (cudaMallocManaged) so the
CPU producer (VAMP) and GPU consumer (cuRobo TrajOpt) share them without an explicit copy.
cuRobo's own allocator + CUDA graphs are left untouched (surgical managed memory).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import torch
import vamp

from univamp import bridge
from univamp.managed_allocator import managed_empty
from univamp.seed_expand import expand_seeds_uma

_HANDOFF_MODES = ("discrete", "managed_copy", "uma_kernel")

# Shared worker pool for parallel multi-path seeding. VAMP's single-planner bindings release the
# GIL (nb::gil_scoped_release via the PLANNER/MFG macros), so independent rrtc() calls on Python
# threads run truly in parallel across CPU cores. One pool avoids per-call thread-spawn overhead.
_SEED_POOL = ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4)))


def _interp_to_horizon(path: np.ndarray, horizon: int) -> np.ndarray:
    """Resample a (M, dof) waypoint path to (horizon, dof) by arc length (linear)."""
    if path.shape[0] == 1:
        return np.repeat(path, horizon, axis=0)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1] if cum[-1] > 0 else 1.0
    q = np.linspace(0.0, total, horizon)
    out = np.empty((horizon, path.shape[1]), dtype=np.float32)
    for j in range(path.shape[1]):
        out[:, j] = np.interp(q, cum, path[:, j])
    return out


class VampSeeder:
    """Generate cuRobo TrajOpt seeds with VAMP-RRTC for a fixed scene."""

    def __init__(self, problem: dict, robot: str = "panda", planner: str = "rrtc",
                 handoff: str = "uma_kernel", dof: Optional[int] = None,
                 use_managed: Optional[bool] = None, multi_seed: bool = False,
                 skip_stride: int = 500, **plan_kwargs):
        if use_managed is not None:   # back-compat with pre-Phase-5 callers
            handoff = "managed_copy" if use_managed else "discrete"
        if handoff not in _HANDOFF_MODES:
            raise ValueError(f"handoff must be one of {_HANDOFF_MODES}, got {handoff!r}")
        self.robot = robot
        self.handoff = handoff
        # multi_seed: fill ALL of cuRobo's TrajOpt seed slots with *diverse* VAMP-RRTC paths
        # (vs the default 1 VAMP path + linear-filled remaining slots). Diversity comes from
        # advancing the Halton sampler by ``skip_stride`` per path, giving independent low-
        # discrepancy sample streams -> different homotopy classes for TrajOpt to optimize.
        self.multi_seed = multi_seed
        self.skip_stride = skip_stride
        self.vmod, self.pfunc, self.psettings, _ = \
            vamp.configure_robot_and_planner_with_kwargs(robot, planner, **plan_kwargs)
        # DOF from the VAMP robot module (no panda-7 assumption); allow override for new robots.
        self.dof = int(dof) if dof is not None else int(self.vmod.dimension())
        # Build VAMP env from the same robometrics problem cuRobo solves (drop the mount).
        start = list(problem["start"])[:self.dof]
        self.env, self.dropped = bridge.robometrics_to_vamp_env(
            problem, filter_start=start, robot=robot)
        self.last_plan_ms = 0.0            # measured wall time (real parallel for multi_seed)
        self.last_plan_ms_serial = 0.0     # sum of per-path times (sequential-equivalent cost)
        self.last_plan_ms_parallel = 0.0   # == last_plan_ms (kept for back-compat)
        self.last_n_success = 0
        self.last_handoff_ms = 0.0

    def seed_trajectories(self, current_state, seed_config: torch.Tensor,
                          horizon: int) -> Optional[torch.Tensor]:
        """Match cuRobo's _get_graph_seed_trajectories contract.

        Args:
            current_state: JointState start (position (1, dof)).
            seed_config: goal configs (1, num_seeds, dof).
            horizon: trajopt action_horizon.
        Returns:
            (1, n_success, horizon, dof) managed CUDA tensor, or None if VAMP solved none.
        """
        dof = seed_config.shape[-1]
        goals = seed_config.view(-1, dof).detach().cpu().numpy().astype(np.float32)
        start = current_state.position.view(-1).detach().cpu().numpy().astype(np.float32)[:dof]
        start_l = start.tolist()

        # cuRobo's IK often returns the same solution across all trajopt seeds; plan each
        # distinct goal once (rounded) and reuse, so the seed cost isn't paid num_seeds times.
        uniq: List[np.ndarray] = []
        for g in goals:
            if not any(np.allclose(g, u, atol=1e-4) for u in uniq):
                uniq.append(g)

        # Plan list: single-seed = one path per unique goal; multi-seed = one diverse path per
        # TrajOpt slot (num_seeds = len(goals)), cycling goals and advancing the Halton stream.
        n_slots = goals.shape[0]
        if self.multi_seed:
            plan = [(uniq[i % len(uniq)], i * self.skip_stride) for i in range(n_slots)]
        else:
            plan = [(g, 0) for g in uniq]

        def _plan_one(job):
            """Run one RRTC (own Halton stream); returns (path|None, wall_ms). GIL is released
            inside pfunc, so calls dispatched on threads run concurrently across cores."""
            g, skip = job
            sampler = self.vmod.halton()
            sampler.reset()
            if skip:
                sampler.skip(skip)
            t0 = time.perf_counter()
            res = self.pfunc(start_l, [g.tolist()], self.env, self.psettings, sampler)
            ms = (time.perf_counter() - t0) * 1e3
            if not res.solved:
                return None, ms
            path = res.path
            arr = (np.asarray(path.numpy(), dtype=np.float32)
                   if hasattr(path, "numpy")
                   else np.asarray([list(c) for c in path], dtype=np.float32))
            return np.ascontiguousarray(arr[:, :dof], dtype=np.float32), ms

        t_wall = time.perf_counter()
        if self.multi_seed and len(plan) > 1:
            results = list(_SEED_POOL.map(_plan_one, plan))   # parallel across CPU cores
        else:
            results = [_plan_one(job) for job in plan]
        wall_ms = (time.perf_counter() - t_wall) * 1e3

        raw = [p for p, _ in results if p is not None]
        per_ms = [ms for _, ms in results]
        # Measured wall time already reflects real parallelism (threads + GIL-released RRTC).
        self.last_plan_ms = wall_ms
        # Sum of per-path times = the sequential cost this parallelism avoided (for reporting).
        self.last_plan_ms_serial = sum(per_ms)
        self.last_plan_ms_parallel = wall_ms
        self.last_n_success = len(raw)

        if not raw:
            return None
        return self._handoff(raw, horizon, dof, seed_config.device)

    def _handoff(self, raw: List[np.ndarray], horizon: int, dof: int, device):
        """Move the raw VAMP paths to a ``(1, n, H, dof)`` device seed tensor. The three modes
        isolate where the CPU->GPU work happens, so the UMA benefit is attributable:

        * ``discrete``     -- CPU arc-length interp -> pinned host buffer -> explicit H2D copy
                              of the *expanded* seed (emulates a discrete-GPU handoff).
        * ``managed_copy`` -- CPU interp -> torch ``copy_`` into a managed buffer (prior UniVAMP).
        * ``uma_kernel``   -- raw waypoints written into managed memory -> fused GPU kernel
                              expands them zero-copy into a device tensor (no expanded-seed copy,
                              interp runs on the GPU).
        """
        h0 = time.perf_counter()
        if self.handoff == "uma_kernel":
            out = expand_seeds_uma(raw, horizon, dof, device=device)
            torch.cuda.synchronize()
            self.last_handoff_ms = (time.perf_counter() - h0) * 1e3
            return out

        # CPU interp shared by discrete / managed_copy
        stacked = np.stack([_interp_to_horizon(p, horizon) for p in raw], axis=0)[None]
        if self.handoff == "managed_copy":
            buf = managed_empty(*stacked.shape, dtype=torch.float32, device=device)
            buf.copy_(torch.from_numpy(stacked))
            torch.cuda.synchronize()
            self.last_handoff_ms = (time.perf_counter() - h0) * 1e3
            return buf
        # discrete: pinned host staging + explicit async H2D copy
        host = torch.from_numpy(stacked).pin_memory()
        dev = torch.empty(stacked.shape, dtype=torch.float32, device=device)
        dev.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        self.last_handoff_ms = (time.perf_counter() - h0) * 1e3
        return dev


def attach_vamp_seeder(planner, seeder: VampSeeder):
    """Additively override a MotionPlanner's graph-seeding with VAMP-RRTC.

    Keeps the original method as ``_orig_get_graph_seed_trajectories`` for fallback.
    """
    if not hasattr(planner, "_orig_get_graph_seed_trajectories"):
        planner._orig_get_graph_seed_trajectories = planner._get_graph_seed_trajectories

    def _vamp_seed(current_state, seed_config):
        horizon = planner.trajopt_solver.action_horizon
        out = seeder.seed_trajectories(current_state, seed_config, horizon)
        if out is None:  # fall back to cuRobo's graph planner
            return planner._orig_get_graph_seed_trajectories(current_state, seed_config)
        return out

    planner._get_graph_seed_trajectories = _vamp_seed
    planner._vamp_seeder = seeder
    return planner
