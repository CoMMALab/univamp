"""Full-MBM seeding experiment: LERP vs cuRobo-PRM vs VAMP-RRTC+UMA.

One pass over the *entire* curated MBM dataset -- every scene, every valid problem, for
ur5 (6 DoF), panda (7), fetch (8) and baxter (14; note baxter has its own 3-scene
both-arms problem set, not the 7-scene set the other three share). Each problem is planned
under three seeding arms, all else identical:

  * ``lerp``      -- cuRobo stock, first attempt: straight-line C-space seed, generated on device.
  * ``prm``       -- cuRobo stock graph fallback (its PRM seeder), also on device. This is the
                     path stock cuRobo takes when the LERP seed fails.
  * ``vamp_uma``  -- proposed: VAMP-RRTC plans the seed on the CPU, the raw waypoints land in a
                     managed (UMA) buffer and a fused kernel expands them zero-copy on the GPU.

Planning is **fully joint-space**: cuRobo's ``plan_cspace`` takes the MBM goal *configuration*
directly, exactly as MBM defines the problem and exactly as VAMP is scored on it. There is no
Cartesian pose goal and no IK stage anywhere in the loop -- an earlier revision of this
experiment went through ``plan_pose`` on the FK pose of the goal config, which inserted an IK
solve that a joint-space planner is never scored against and which dominated failures on the
non-redundant UR5 (47.5% of its problems died there). That data is kept as
``exp_mbm_posegoal.jsonl`` for reference.

Metrics recorded per problem: success, total plan time (ms), joint-space path length (rad), and
the phase decomposition (seed-gen / handoff / TrajOpt / other) that feeds the ablation figure.
Aggregates are means over all trials of all problems.

TrajOpt runs in early-exit-on-convergence mode (budget 200) so that seed quality can express
itself as time rather than being masked by a fixed iteration count.

Run (writes JSONL incrementally, safe to resume):
    python results/exp_mbm_full.py --json results/exp_mbm_full.jsonl
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

_REAL_ARGV = sys.argv[1:]
sys.argv = ["b"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BD = os.path.join(ROOT, "curobo", "benchmark")  # cuRobo's benchmark cwd: its configs are relative
sys.path.insert(0, BD)
sys.path.insert(0, ROOT)
os.chdir(BD)

import torch  # noqa: E402
from copy import deepcopy  # noqa: E402
from curobo._src.geom.types import SceneCfg  # noqa: E402
from curobo._src.state.state_joint import JointState  # noqa: E402
from curobo._src.util.logging import setup_curobo_logger  # noqa: E402
from univamp.curobo_loader import build_motion_planner  # noqa: E402
from univamp.seeder import VampSeeder, attach_vamp_seeder  # noqa: E402
from univamp import bridge  # noqa: E402
import vamp  # noqa: E402
from results.mbm_problems import load_mbm  # noqa: E402

bridge.CYLINDERS_AS_BOXES = True  # identical box geometry for VAMP + cuRobo-OBB

ROBOTS = ["ur5", "panda", "fetch", "baxter"]
ARMS = ["lerp", "prm", "vamp_uma"]
LBL = {"lerp": "LERP (stock)", "prm": "PRM (stock fallback)", "vamp_uma": "VAMP-RRTC + UMA"}

# Per-arm planning-attempt budget. PRM gets cuRobo's stock default (plan_cspace's
# max_attempts=5); LERP and VAMP-RRTC get a single attempt.
#
# Why asymmetric: PRM's characteristic failure is returning *no seed at all* (the graph planner
# finds no roadmap path, so plan_cspace `continue`s and TrajOpt never runs), and the roadmap draw
# is stochastic -- back-to-back calls on the same problem disagree ~8% of the time. Retries
# therefore repair PRM's dominant failure mode, while LERP can never suffer it (a straight line
# always exists) and VAMP-RRTC almost never does. Holding every arm at one attempt is internally
# consistent but penalises the one baseline whose failure is stochastic, so we give PRM the retry
# budget cuRobo ships with and let it compete at full strength. Measured on 120 UR5 problems,
# this lifts PRM from 5.0% to 19.2%. Note the budget also inflates PRM's mean plan time, since a
# failed problem now pays up to five graph searches.
ARM_ATTEMPTS = {"lerp": 1, "prm": 5, "vamp_uma": 1}


def sync():
    torch.cuda.synchronize()


def path_length(res, dof):
    """Joint-space L2 path length (rad) of the best successful optimized solution."""
    try:
        if res is None or res.js_solution is None:
            return None
        pos = res.js_solution.position.detach()
        pos = pos.reshape(-1, pos.shape[-2], pos.shape[-1])[..., :dof]  # (S, H, dof)
        succ = res.success.reshape(-1).bool()
        if succ.shape[0] == pos.shape[0]:
            pos = pos[succ]
        if pos.shape[0] == 0:
            return None
        lens = torch.linalg.norm(pos[:, 1:] - pos[:, :-1], dim=-1).sum(-1)  # (S,)
        return float(lens.min().item())
    except Exception:
        return None


def configure_trajopt(mg, early_exit, budget, converged_ratio):
    """Toggle cuRobo's LBFGS TrajOpt between fixed-iteration and early-exit-on-convergence and
    set the iteration budget. The baked-in default (fixed_iters=True) always burns the full
    budget, which hides the benefit of a strong seed."""
    opt = mg.trajopt_solver.optimizer.optimizers[-1]
    if not hasattr(opt.config, "num_iters"):
        return
    opt.config.fixed_iters = not early_exit
    if early_exit:
        opt.config.converged_ratio = converged_ratio
    opt.update_niters(budget)
    opt._og_num_iters = budget  # so reinitialize(reset_num_iters=True) keeps the new budget


def restore_stock(mg):
    if hasattr(mg, "_orig_get_graph_seed_trajectories"):
        mg._get_graph_seed_trajectories = mg._orig_get_graph_seed_trajectories


class PhaseTimer:
    """Accumulate CUDA-synced wall time per phase, so seed generation can be isolated as
    total - IK - TrajOpt for whichever seeder is active."""

    def __init__(self):
        self.acc = {"trajopt": 0.0}

    def wrap(self, obj, attr, name):
        orig = getattr(obj, attr)

        def timed(*a, **k):
            sync(); t = time.perf_counter()
            r = orig(*a, **k)
            sync(); self.acc[name] += (time.perf_counter() - t) * 1e3
            return r
        setattr(obj, attr, timed)

    def reset(self):
        self.acc = {"trajopt": 0.0}


def plan_once(mg, gs, ss, arm, pt, attempts=1):
    """One joint-space plan (start config -> goal config) under the given seeding arm.

    ``plan_cspace`` never touches IK: with ``enable_graph_attempt=10`` and ``max_attempts=1`` the
    seed stays None and TrajOpt runs on its own straight-line initialization (the LERP arm); with
    ``enable_graph_attempt=0`` the seed comes from ``_get_graph_seed_trajectories`` -- cuRobo's
    PRM, or the VAMP seeder when one is attached over that hook.

    Returns (ok, total_ms, seed_gen_ms, result), seed_gen = total - trajopt."""
    mg.reset_seed()
    pt.reset()
    sync()
    t = time.perf_counter()
    if arm == "lerp":
        # enable_graph_attempt > max_attempts -> seed stays None on every attempt
        res = mg.plan_cspace(gs, ss, max_attempts=attempts, enable_graph_attempt=attempts + 1)
    else:
        res = mg.plan_cspace(gs, ss, max_attempts=attempts, enable_graph_attempt=0)
    sync()
    ms = (time.perf_counter() - t) * 1e3
    seed_gen = max(0.0, ms - pt.acc["trajopt"])
    ok = res is not None and torch.count_nonzero(res.success) > 0
    return ok, ms, seed_gen, res


def load_done(path):
    """Read an existing JSONL so an interrupted run can resume where it stopped."""
    done = set()
    if not path or not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated final line from a hard kill
            done.add((r["robot"], r["scene"], r["problem"], r["strategy"]))
    return done


def run(robots, limit, use_cuda_graph, early_exit, budget, converged_ratio, arms, jsonl):
    setup_curobo_logger("error")
    done = load_done(jsonl)
    if done:
        print(f"### resuming: {len(done)} records already present in {jsonl} ###", flush=True)
    jf = open(jsonl, "a") if jsonl else None
    mode = (f"EARLY-EXIT (budget={budget}, converged_ratio={converged_ratio})"
            if early_exit else f"FIXED-ITERS (budget={budget})")
    print(f"### TrajOpt mode: {mode} | arms: {' '.join(arms)} ###", flush=True)
    print("### attempts per arm: "
          + ", ".join(f"{a}={ARM_ATTEMPTS[a]}" for a in arms) + " ###", flush=True)

    agg = defaultdict(lambda: {"n": 0, "succ": 0, "ms": 0.0, "path": []})

    for robot in robots:
        problems = load_mbm(robot, per_scene=limit or 0)
        dof = int(getattr(vamp, robot).dimension())
        print(f"\n##### {robot} ({dof} DoF): {len(problems)} problems #####", flush=True)
        mg = build_motion_planner(robot, use_cuda_graph=use_cuda_graph)
        mg.warmup(enable_graph=True)
        configure_trajopt(mg, early_exit, budget, converged_ratio)
        # warmup() only exercises plan_pose; run a few cspace solves so the CUDA graphs for the
        # solve_cspace path are captured before the first timed problem.
        _ss = JointState.from_position(mg.device_cfg.to_device([problems[0]["start"]]))
        _gs = JointState.from_position(mg.device_cfg.to_device([problems[0]["goal"]]))
        for _ in range(3):
            mg.plan_cspace(_gs, _ss, max_attempts=1, enable_graph_attempt=2)
        sync()
        pt = PhaseTimer()
        pt.wrap(mg.trajopt_solver, "solve_cspace", "trajopt")
        t_robot = time.perf_counter()

        for pi, problem in enumerate(problems):
            scene = problem["scene"]
            todo = [a for a in arms if (robot, scene, pi, a) not in done]
            if not todo:
                continue
            world = SceneCfg.create(deepcopy(problem["obstacles"])).get_obb_world()
            mg.scene_collision_checker.clear_cache()
            mg.update_world(world)
            ss = JointState.from_position(mg.device_cfg.to_device([problem["start"]]))
            gs = JointState.from_position(mg.device_cfg.to_device([problem["goal"]]))

            for arm in todo:
                restore_stock(mg)
                seeder = None
                if arm == "vamp_uma":
                    seeder = VampSeeder(problem, robot=robot, planner="rrtc",
                                        handoff="uma_kernel")
                    attach_vamp_seeder(mg, seeder)
                ok, ms, seed, res = plan_once(mg, gs, ss, arm, pt, ARM_ATTEMPTS[arm])
                restore_stock(mg)

                handoff_ms = seeder.last_handoff_ms if seeder else 0.0
                plen = path_length(res, dof)
                a = agg[(robot, arm)]
                a["n"] += 1; a["succ"] += ok; a["ms"] += ms
                if ok and plen is not None:
                    a["path"].append(plen)
                if jf is not None:
                    jf.write(json.dumps({
                        "robot": robot, "dof": dof, "scene": scene, "problem": pi,
                        "strategy": arm, "attempts": ARM_ATTEMPTS[arm],
                        "success": bool(ok), "total_ms": round(ms, 3),
                        "trajopt_ms": round(pt.acc["trajopt"], 3),
                        "seed_ms": round(seed, 3), "handoff_ms": round(handoff_ms, 4),
                        "vamp_seed_ms": (None if seeder is None
                                         else round(seeder.last_plan_ms, 3)),
                        "path_len": plen,
                    }) + "\n")
                    jf.flush()

            if (pi + 1) % 50 == 0:
                el = time.perf_counter() - t_robot
                print(f"  [{robot}] {pi+1}/{len(problems)} problems  "
                      f"({el:.0f}s elapsed, {el/(pi+1):.2f}s/problem)", flush=True)

        del mg
        torch.cuda.empty_cache()

    if jf is not None:
        jf.close()
    report(agg, robots, arms)


def report(agg, robots, arms):
    print("\n\n============ FULL-MBM SEEDING EXPERIMENT (this run) ============")
    print("Averages over all trials of all problems, per robot.\n")
    hdr = (f"{'robot':8s} {'dof':>4s} {'arm':22s} {'N':>6s} {'success':>10s} "
           f"{'plan ms':>10s} {'path rad':>10s}")
    print(hdr); print("-" * len(hdr))
    for robot in robots:
        dof = int(getattr(vamp, robot).dimension())
        for arm in arms:
            a = agg.get((robot, arm))
            if not a or a["n"] == 0:
                continue
            print(f"{robot:8s} {dof:4d} {LBL[arm]:22s} {a['n']:6d} "
                  f"{100.0*a['succ']/a['n']:9.1f}% {a['ms']/a['n']:10.1f} "
                  f"{(np.mean(a['path']) if a['path'] else float('nan')):10.2f}")
        print()
    print("(Definitive aggregates come from the JSONL via paper/make_paper_figs.py, which also"
          " covers records carried over from an earlier resumed run.)")


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--robots", nargs="+", default=ROBOTS)
    P.add_argument("--limit", type=int, default=0,
                   help="max problems per SCENE (0 = all; the paper uses 0)")
    P.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    P.add_argument("--no-cuda-graph", action="store_true")
    P.add_argument("--fixed-iters", action="store_true",
                   help="disable TrajOpt early-exit (paper uses early-exit)")
    P.add_argument("--budget", type=int, default=200,
                   help="LBFGS num_iters budget (multiple of inner_iters=25)")
    P.add_argument("--converged-ratio", type=float, default=0.1)
    P.add_argument("--json", default=os.path.join(ROOT, "results", "exp_mbm_full.jsonl"),
                   help="per-problem records (appended; existing records are skipped)")
    A = P.parse_args(_REAL_ARGV)
    run(A.robots, A.limit, not A.no_cuda_graph, not A.fixed_iters, A.budget,
        A.converged_ratio, A.arms, A.json)
    sys.stdout.flush()
    os._exit(0)  # cuRobo cuda_core + VAMP crash at interpreter teardown
