"""Robot-agnostic cuRobo MotionPlanner factory for the seed-strategy benchmark.

Mirrors ``benchmark/motion_plan_benchmark.load_curobo`` but loads any of our generated robot
configs (``univamp/robot_assets/<robot>/<robot>.yml``) instead of hard-coding franka. Must be
imported with the curobo ``benchmark`` dir on ``sys.path`` (for the optimizer config names),
which the benchmark scripts already arrange.
"""
from __future__ import annotations

import copy
import os
from typing import Optional

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo._src.geom.types import SceneCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util_file import (get_scene_configs_path, join_path, load_yaml)

_HERE = os.path.dirname(os.path.abspath(__file__))


def robot_yaml(robot: str) -> str:
    return os.path.join(_HERE, "robot_assets", robot, f"{robot}.yml")


def build_motion_planner(
    robot: str,
    n_cubes: int = 40,
    num_ik_seeds: int = 32,
    num_trajopt_seeds: int = 4,
    collision_buffer: float = 0.0,
    use_cuda_graph: bool = True,
) -> MotionPlanner:
    robot_cfg = load_yaml(robot_yaml(robot))
    if "robot_cfg" in robot_cfg:
        robot_cfg = robot_cfg["robot_cfg"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] = collision_buffer
    robot_cfg["load_dynamics"] = False

    scene_cfg = SceneCfg.create(
        load_yaml(join_path(get_scene_configs_path(), "collision_table.yml"))
    ).get_obb_world()

    robot_cfg_instance = RobotCfg.create(copy.deepcopy(robot_cfg), device_cfg=DeviceCfg())

    cfg = MotionPlannerCfg.create(
        robot=robot_cfg_instance,
        scene_model=scene_cfg,
        ik_optimizer_configs=["ik/particle_ik.yml", "ik/lbfgs_ik.yml"],
        ik_transition_model="ik/transition_ik.yml",
        metrics_rollout="metrics_base.yml",
        trajopt_optimizer_configs=["trajopt/particle_trajopt.yml",
                                   "trajopt/lbfgs_bspline_trajopt.yml"],
        trajopt_transition_model="trajopt/transition_bspline_trajopt.yml",
        use_cuda_graph=use_cuda_graph,
        num_ik_seeds=num_ik_seeds,
        num_trajopt_seeds=num_trajopt_seeds,
        # Mesh cache too: VAMP models cylinders (the "cans") as capsules, and cuRobo's OBB world
        # inflates cylinders into bounding boxes -> false collisions. get_mesh_world() keeps
        # cylinders accurate, matching VAMP's collision model for a fair comparison.
        collision_cache={"obb": n_cubes, "mesh": n_cubes},
        store_debug=False,
        optimizer_collision_activation_distance=0.0025,
    )
    return MotionPlanner(cfg)
