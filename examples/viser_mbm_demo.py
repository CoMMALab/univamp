"""Interactive viser demo: VAMP-RRTC seeded cuRobo TrajOpt on one MBM problem.

Plans a single problem from VAMP's curated MBM set with the UniVAMP pipeline --
VAMP-RRTC generates the TrajOpt seed on the CPU, the raw waypoints land in a managed
(unified-memory) buffer, and a fused CUDA kernel expands them zero-copy into cuRobo's
TrajOpt seed tensor -- then serves the scene and the optimized trajectory in a viser
browser view with a play button and the measured solve time.

    python examples/viser_mbm_demo.py --robot panda --problem cage_panda --index 0

``--problem`` is the MBM scene name (``--list`` prints the scenes available for a robot)
and ``--index`` selects among that scene's problems.
"""
import argparse
import os
import sys
import time

import numpy as np

# cuRobo resolves its optimizer configs relative to the benchmark dir, and argparse must not
# see our flags while cuRobo's benchmark modules import. Same dance as results/exp_mbm_full.py.
_REAL_ARGV = sys.argv[1:]
sys.argv = ["viser_mbm_demo"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "curobo", "benchmark")
sys.path.insert(0, BENCH)
sys.path.insert(0, ROOT)
os.chdir(BENCH)

import torch  # noqa: E402
import viser  # noqa: E402
import yourdfpy  # noqa: E402
from copy import deepcopy  # noqa: E402
from viser.extras import ViserUrdf  # noqa: E402

from curobo._src.geom.types import SceneCfg  # noqa: E402
from curobo._src.state.state_joint import JointState  # noqa: E402
from curobo._src.util.logging import setup_curobo_logger  # noqa: E402

import vamp  # noqa: E402
from univamp import bridge  # noqa: E402
from univamp.curobo_loader import build_motion_planner  # noqa: E402
from univamp.seeder import VampSeeder, attach_vamp_seeder  # noqa: E402
from results.mbm_problems import load_mbm, scene_list  # noqa: E402

# cuRobo's OBB collision world inflates cylinders to their bounding boxes; give VAMP the same
# geometry so a VAMP seed is valid in exactly the world cuRobo optimizes in.
bridge.CYLINDERS_AS_BOXES = True

OBSTACLE_COLOR = (150, 160, 175)
TRACE_COLOR = (255, 140, 0)


# --------------------------------------------------------------------------- planning


def build_planner(robot: str, budget: int, use_cuda_graph: bool):
    """cuRobo MotionPlanner for ``robot``, warmed up and set to early-exit TrajOpt."""
    setup_curobo_logger("error")
    mg = build_motion_planner(robot, use_cuda_graph=use_cuda_graph)
    mg.warmup(enable_graph=True)

    # TrajOpt early-exit on convergence: a good seed then shows up as less time rather than
    # being masked by a fixed iteration count.
    opt = mg.trajopt_solver.optimizer.optimizers[-1]
    if hasattr(opt.config, "num_iters"):
        opt.config.fixed_iters = False
        opt.config.converged_ratio = 0.1
        opt.update_niters(budget)
        opt._og_num_iters = budget
    return mg


def solve(mg, robot: str, problem: dict, multi_seed: bool):
    """Plan the problem with a VAMP-seeded cuRobo TrajOpt.

    Returns ``(traj, dt, timings)`` where ``traj`` is ``(N, dof)`` in cuRobo's cspace joint
    order, or ``(None, None, timings)`` if the solve failed.
    """
    world = SceneCfg.create(deepcopy(problem["obstacles"])).get_obb_world()
    mg.scene_collision_checker.clear_cache()
    mg.update_world(world)

    start = JointState.from_position(mg.device_cfg.to_device([problem["start"]]))
    goal = JointState.from_position(mg.device_cfg.to_device([problem["goal"]]))

    # Capture the CUDA graphs for the cspace path before timing (warmup() only exercises
    # plan_pose), so graph capture is not charged to the reported solve time.
    for _ in range(3):
        mg.plan_cspace(goal, start, max_attempts=1, enable_graph_attempt=2)
    torch.cuda.synchronize()

    seeder = VampSeeder(problem, robot=robot, planner="rrtc", handoff="uma_kernel",
                        multi_seed=multi_seed)
    attach_vamp_seeder(mg, seeder)
    try:
        mg.reset_seed()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # enable_graph_attempt=0 routes seeding through _get_graph_seed_trajectories on attempt
        # 0, which attach_vamp_seeder has replaced with VAMP-RRTC.
        res = mg.plan_cspace(goal, start, max_attempts=1, enable_graph_attempt=0)
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1e3
    finally:
        if hasattr(mg, "_orig_get_graph_seed_trajectories"):
            mg._get_graph_seed_trajectories = mg._orig_get_graph_seed_trajectories

    timings = {
        "total_ms": total_ms,
        "seed_ms": seeder.last_plan_ms,
        "handoff_ms": seeder.last_handoff_ms,
        "n_seeds": seeder.last_n_success,
        "status": None if res is None else getattr(res, "status", None),
    }
    if res is None or torch.count_nonzero(res.success) == 0:
        return None, None, timings

    traj, dt = extract_trajectory(res, len(problem["start"]))
    timings["motion_s"] = dt * (len(traj) - 1)
    return traj, dt, timings


def _scalar_dt(state, default: float = 0.02) -> float:
    """cuRobo reports dt as a float or as a per-seed tensor; reduce it to one number."""
    dt = getattr(state, "dt", None)
    if dt is None:
        return default
    try:
        return float(torch.as_tensor(dt).reshape(-1)[0].item())
    except (TypeError, RuntimeError, IndexError):
        return default


def extract_trajectory(res, dof: int):
    """Best successful trajectory as ``(N, dof)`` numpy + the timestep between samples.

    Prefers cuRobo's densely interpolated trajectory (trimmed at its last valid tstep) so
    playback runs at the planner's own time parameterization.
    """
    succ = res.success.reshape(-1).bool()
    best = int(torch.nonzero(succ)[0].item())

    interp = getattr(res, "interpolated_trajectory", None)
    if interp is not None and interp.position is not None:
        pos = interp.position.reshape(-1, interp.position.shape[-2], interp.position.shape[-1])
        traj = pos[best]
        last = getattr(res, "interpolated_last_tstep", None)
        if last is not None:
            traj = traj[: int(last.reshape(-1)[best].item()) + 1]
        dt = _scalar_dt(interp)
    else:
        pos = res.js_solution.position
        pos = pos.reshape(-1, pos.shape[-2], pos.shape[-1])
        traj = pos[best]
        dt = _scalar_dt(res.js_solution)
    return traj[:, :dof].detach().cpu().numpy().astype(np.float64), dt


def ee_trace(mg, traj: np.ndarray):
    """End-effector positions along the trajectory, or None if FK isn't available."""
    try:
        js = JointState.from_position(mg.device_cfg.to_device(traj.tolist()))
        state = mg.kinematics.compute_kinematics(js)
        pos = state.tool_poses.position.reshape(-1, 3)
        return pos.detach().cpu().numpy()
    except Exception:
        return None


# --------------------------------------------------------------------------- visualization


def load_urdf(robot: str) -> yourdfpy.URDF:
    """VAMP's full URDF (with visual meshes); its root is the world frame MBM is defined in."""
    d = os.path.join(ROOT, "vamp", "resources", robot)

    def handler(fname: str) -> str:
        return os.path.join(d, fname.replace("package://", ""))

    return yourdfpy.URDF.load(os.path.join(d, f"{robot}.urdf"),
                              filename_handler=handler, load_collision_meshes=False)


def add_obstacles(server: viser.ViserServer, obstacles: dict) -> None:
    """Draw the scene as the *planner* sees it: cuRobo's OBB world, so cylinders are drawn as
    the bounding boxes both cuRobo and (with CYLINDERS_AS_BOXES) VAMP collision-check against."""
    for name, c in obstacles.get("cuboid", {}).items():
        server.scene.add_box(f"/world/box_{name}", dimensions=tuple(c["dims"]),
                             position=tuple(c["pose"][:3]), wxyz=tuple(c["pose"][3:7]),
                             color=OBSTACLE_COLOR, opacity=0.6)
    for name, c in obstacles.get("cylinder", {}).items():
        dims = (2 * c["radius"], 2 * c["radius"], c["height"])
        server.scene.add_box(f"/world/cyl_{name}", dimensions=dims,
                             position=tuple(c["pose"][:3]), wxyz=tuple(c["pose"][3:7]),
                             color=OBSTACLE_COLOR, opacity=0.6)
    for name, c in obstacles.get("sphere", {}).items():
        server.scene.add_icosphere(f"/world/sph_{name}", radius=c["radius"],
                                   position=tuple(c["pose"][:3]),
                                   color=OBSTACLE_COLOR, opacity=0.6)


def joint_reorder(urdf: yourdfpy.URDF, curobo_joint_names) -> np.ndarray:
    """Index array mapping a cuRobo cspace config to the URDF's actuated-joint order.

    These orders differ for some robots (baxter: cuRobo lists left arm first, the URDF right),
    so the mapping is by joint *name*, never by position.
    """
    lookup = {n: i for i, n in enumerate(curobo_joint_names)}
    missing = [n for n in urdf.actuated_joint_names if n not in lookup]
    if missing:
        raise RuntimeError(f"URDF joints absent from the cuRobo cspace: {missing}")
    return np.array([lookup[n] for n in urdf.actuated_joint_names])


def serve(robot, problem, traj, dt, timings, mg, port, trace):
    server = viser.ViserServer(port=port)
    server.scene.add_grid("/grid", width=4.0, height=4.0, position=(0.0, 0.0, 0.0))
    add_obstacles(server, problem["obstacles"])

    urdf = load_urdf(robot)
    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot")
    order = joint_reorder(urdf, mg.kinematics.joint_names)

    if trace is not None and len(trace) > 1:
        server.scene.add_spline_catmull_rom("/world/ee_trace", points=trace,
                                            line_width=3.0, color=TRACE_COLOR)

    solved = traj is not None
    n = len(traj) if solved else 1

    lines = [f"### {robot} / {problem['scene']}",
             f"**Solve time: {timings['total_ms']:.1f} ms**",
             ""]
    if timings["n_seeds"]:
        lines.append(f"- VAMP seed: {timings['seed_ms']:.1f} ms "
                     f"({timings['n_seeds']} path(s))")
    else:
        # attach_vamp_seeder falls back to cuRobo's own graph seeder when RRTC finds nothing,
        # so say so rather than showing a 0 ms handoff with no explanation.
        lines.append(f"- VAMP seed: {timings['seed_ms']:.1f} ms "
                     f"(no path -- fell back to cuRobo's graph seeder)")
    lines.append(f"- UMA handoff: {timings['handoff_ms']:.2f} ms")
    lines.append("- TrajOpt + other: "
                 f"{timings['total_ms'] - timings['seed_ms'] - timings['handoff_ms']:.1f} ms")
    if solved:
        lines.append(f"- Motion duration: {timings['motion_s']:.2f} s ({n} waypoints)")
    else:
        # cuRobo leaves status unset on the cspace path, so only show it when there is one.
        why = f" ({timings['status']})" if timings["status"] else ""
        lines.append(f"- **no solution found**{why} -- try another --index")
    server.gui.add_markdown("\n".join(lines))

    play = server.gui.add_button("▶ Play", disabled=not solved)
    slider = server.gui.add_slider("Waypoint", min=0, max=max(n - 1, 1), step=1,
                                   initial_value=0, disabled=not solved)
    speed = server.gui.add_slider("Speed", min=0.1, max=4.0, step=0.1, initial_value=1.0)
    loop = server.gui.add_checkbox("Loop", False)

    playing = {"on": False}

    def show(i: int) -> None:
        cfg = traj[i] if solved else np.asarray(problem["start"], dtype=np.float64)
        viser_urdf.update_cfg(cfg[order])

    @play.on_click
    def _(_evt) -> None:
        if not solved:
            return
        if slider.value >= n - 1:
            slider.value = 0
        playing["on"] = not playing["on"]
        play.label = "⏸ Pause" if playing["on"] else "▶ Play"

    @slider.on_update
    def _(_evt) -> None:
        show(int(slider.value))

    show(0)
    # viser silently falls back to a free port if the requested one is taken; report the real one.
    print(f"\nviser: http://localhost:{server.get_port()}  (ctrl-C to stop)", flush=True)

    # Playback advances in real time at the planner's dt, scaled by the speed slider.
    while True:
        step = dt / max(speed.value, 1e-3) if solved else 0.05
        time.sleep(min(step, 0.1))
        if not playing["on"]:
            continue
        if slider.value >= n - 1:
            if loop.value:
                slider.value = 0
            else:
                playing["on"] = False
                play.label = "▶ Play"
            continue
        slider.value = int(slider.value) + 1


# --------------------------------------------------------------------------- entrypoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="panda", choices=["ur5", "panda", "fetch", "baxter"])
    p.add_argument("--problem", default=None, help="MBM scene name (default: first scene)")
    p.add_argument("--index", type=int, default=0, help="problem index within the scene")
    p.add_argument("--list", action="store_true", help="list scenes for --robot and exit")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--budget", type=int, default=200, help="LBFGS TrajOpt iteration budget")
    p.add_argument("--multi-seed", action="store_true",
                   help="fill every TrajOpt seed slot with a diverse VAMP-RRTC path")
    p.add_argument("--no-cuda-graph", action="store_true")
    a = p.parse_args(_REAL_ARGV)

    if a.list:
        print(f"{a.robot} scenes: {' '.join(scene_list(a.robot))}")
        return

    scene = a.problem or scene_list(a.robot)[0]
    problems = load_mbm(a.robot, scenes=[scene])
    if not problems:
        raise SystemExit(f"no problems for {a.robot}/{scene}; "
                         f"scenes: {' '.join(scene_list(a.robot))}")
    if not 0 <= a.index < len(problems):
        raise SystemExit(f"--index {a.index} out of range: {scene} has {len(problems)} problems")
    problem = problems[a.index]
    print(f"{a.robot} | scene {scene} | problem {a.index}/{len(problems) - 1} "
          f"| {int(getattr(vamp, a.robot).dimension())} DoF", flush=True)

    mg = build_planner(a.robot, a.budget, not a.no_cuda_graph)
    traj, dt, timings = solve(mg, a.robot, problem, a.multi_seed)
    outcome = "solved" if traj is not None else f"FAILED {timings['status'] or ''}".strip()
    print(f"solve: {timings['total_ms']:.1f} ms "
          f"(VAMP seed {timings['seed_ms']:.1f} ms, handoff {timings['handoff_ms']:.2f} ms) "
          f"-> {outcome}", flush=True)

    trace = ee_trace(mg, traj) if traj is not None else None
    serve(a.robot, problem, traj, dt, timings, mg, a.port, trace)


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:                      # argparse / our own usage errors
        print(e, file=sys.stderr)
        code = 1 if e.code else 0
    except BaseException:                        # incl. KeyboardInterrupt from the serve loop
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # cuRobo cuda_core + VAMP crash at interpreter teardown
