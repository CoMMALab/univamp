"""Bridge between the robometrics MBM problem format (used by cuRobo) and VAMP.

Single source of truth = robometrics `motion_benchmaker_raw()`. We convert each problem's
obstacles into a VAMP Environment and expose start + joint-space goals (goal_ik) so VAMP-RRTC
can plan in the same scene cuRobo solves (cuRobo uses the Cartesian goal_pose).
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np
import vamp


def _quat_wxyz_to_euler_xyz(q) -> List[float]:
    from scipy.spatial.transform import Rotation as R
    qw, qx, qy, qz = q
    return R.from_quat([qx, qy, qz, qw]).as_euler("xyz").tolist()


# cuRobo's collision scene only supports cuboids (OBB) and meshes, and its TrajOpt is tuned for
# cuboids. To give VAMP and cuRobo an *identical* obstacle model (so a VAMP seed is valid in the
# same world cuRobo optimizes in), set this True to add cylinders to VAMP as their bounding
# cuboids too, matching cuRobo's ``get_obb_world()``. Left False preserves the original capsule
# behaviour used by the earlier panda phases.
CYLINDERS_AS_BOXES = False


def _obstacle_adders(problem: Dict[str, Any]):
    """Yield (name, add_fn) where add_fn(env) adds one obstacle to a VAMP Environment."""
    obstacles = problem["obstacles"]

    for name, c in obstacles.get("cuboid", {}).items():
        euler = _quat_wxyz_to_euler_xyz(c["pose"][3:7])
        half = [d / 2.0 for d in c["dims"]]
        pos = list(c["pose"][:3])
        def add(env, pos=pos, euler=euler, half=half, name=name):
            o = vamp.Cuboid(pos, euler, half); o.name = name; env.add_cuboid(o)
        yield name, add

    for name, c in obstacles.get("cylinder", {}).items():
        euler = _quat_wxyz_to_euler_xyz(c["pose"][3:7])
        pos = list(c["pose"][:3]); r = c["radius"]; h = c["height"]
        if CYLINDERS_AS_BOXES:
            half = [r, r, h / 2.0]
            def add(env, pos=pos, euler=euler, half=half, name=name):
                o = vamp.Cuboid(pos, euler, half); o.name = name; env.add_cuboid(o)
        else:
            def add(env, pos=pos, euler=euler, r=r, h=h, name=name):
                o = vamp.Cylinder(pos, euler, r, h); o.name = name; env.add_capsule(o)
        yield name, add

    for name, c in obstacles.get("sphere", {}).items():
        pos = list(c["pose"][:3]); r = c["radius"]
        def add(env, pos=pos, r=r, name=name):
            o = vamp.Sphere(pos, r); o.name = name; env.add_sphere(o)
        yield name, add


def robometrics_to_vamp_env(problem: Dict[str, Any],
                            filter_start: List[float] = None,
                            robot: str = "panda") -> Tuple[vamp.Environment, List[str]]:
    """Build a VAMP Environment from a robometrics MBM problem.

    If ``filter_start`` (a known collision-free config, e.g. the problem start) is given,
    any single obstacle that collides with it is dropped: MBM scenes include the robot's
    mounting platform (e.g. ``cube_robot_stand``) as an obstacle, which cuRobo treats as an
    allowed collision. Returns (env, dropped_names).
    """
    vmod = getattr(vamp, robot)
    env = vamp.Environment()
    dropped: List[str] = []
    for name, add in _obstacle_adders(problem):
        if filter_start is not None:
            probe = vamp.Environment(); add(probe)
            if not vmod.validate(filter_start, probe):
                dropped.append(name)
                continue
        add(env)
    return env, dropped


def robometrics_to_pointcloud(problem: Dict[str, Any], samples_per_object: int = 2000,
                              skip_names=()):
    """Sample a surface point cloud from a robometrics problem's obstacles (simulates the
    perception stack emitting points). ``skip_names`` (e.g. the robot mount) are excluded.
    Returns an (N,3) float32 array."""
    from vamp import pointcloud as vpc
    skip = set(skip_names)
    pcs = []
    obstacles = problem["obstacles"]
    for name, c in obstacles.get("cuboid", {}).items():
        if name in skip:
            continue
        q = c["pose"][3:7]  # wxyz
        box = {"position": list(c["pose"][:3]),
               "orientation_quat_xyzw": [q[1], q[2], q[3], q[0]],
               "half_extents": [d / 2.0 for d in c["dims"]]}
        pcs.append(vpc.box_to_pc(box, samples_per_object))
    for name, c in obstacles.get("cylinder", {}).items():
        if name in skip:
            continue
        q = c["pose"][3:7]
        cyl = {"position": list(c["pose"][:3]),
               "orientation_quat_xyzw": [q[1], q[2], q[3], q[0]],
               "radius": c["radius"], "length": c["height"]}
        pcs.append(vpc.cylinder_to_pc(cyl, samples_per_object))
    return np.vstack(pcs).astype(np.float32) if pcs else np.zeros((0, 3), np.float32)


def robometrics_to_capt_env(problem: Dict[str, Any], robot: str = "panda",
                            samples_per_object: int = 2000, filter_radius: float = 0.02,
                            filter_cull: bool = True, filter_start: List[float] = None):
    """Build a VAMP CAPT Environment from a perception-style point cloud (the CPU/VAMP env
    representation, separate from cuRobo's NVBlox). If ``filter_start`` is given, obstacles
    colliding with it (the robot mount) are excluded from the cloud. Returns
    (env, n_points_raw, n_points_filtered, filter_ms, build_ms)."""
    from vamp import pointcloud as vpc
    skip = ()
    if filter_start is not None:
        _, skip = robometrics_to_vamp_env(problem, filter_start=filter_start, robot=robot)
    raw = robometrics_to_pointcloud(problem, samples_per_object, skip_names=skip)
    r_min, r_max = getattr(vamp, robot).min_max_radii()

    origin = vpc.ROBOT_FIRST_JOINT_LOCATIONS.get(robot, [0.0, 0.0, 0.0])
    cull_r = vpc.ROBOT_MAX_RADII.get(robot, 1.4)
    lo = (np.asarray(origin) - cull_r).tolist()
    hi = (np.asarray(origin) + cull_r).tolist()
    filtered, filter_time = vpc.filter_pointcloud(
        raw.tolist(), filter_radius, cull_r, origin, lo, hi, filter_cull)

    env = vamp.Environment()
    build_time = env.add_pointcloud(filtered, r_min, r_max, vpc.POINT_RADIUS)
    return env, len(raw), len(filtered), filter_time / 1e6, build_time / 1e6


def vamp_start_goals(problem: Dict[str, Any]) -> Tuple[List[float], List[List[float]]]:
    """Return (start_config, [goal_configs]) in panda's 7 arm DOF."""
    start = list(problem["start"])[:7]
    goals = [list(g)[:7] for g in problem["goal_ik"]]
    return start, goals


def plan_rrtc(problem: Dict[str, Any], robot: str = "panda", planner: str = "rrtc",
              **kwargs):
    """Run VAMP-RRTC on a robometrics problem. Returns (result, path_np, plan_ms, env)."""
    import time
    vmod, pfunc, psettings, _ssettings = vamp.configure_robot_and_planner_with_kwargs(
        robot, planner, **kwargs
    )
    sampler = vmod.halton()
    start, goals = vamp_start_goals(problem)
    env, _dropped = robometrics_to_vamp_env(problem, filter_start=start, robot=robot)

    t = time.perf_counter()
    result = pfunc(start, goals, env, psettings, sampler)
    plan_ms = (time.perf_counter() - t) * 1e3

    path_np = None
    if result.solved:
        path = result.path
        if hasattr(path, "numpy"):
            path_np = np.asarray(path.numpy(), dtype=np.float32)
        else:
            path_np = np.asarray([list(c) for c in path], dtype=np.float32)
    return result, path_np, plan_ms, env
