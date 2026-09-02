"""Load VAMP's curated MBM benchmark problems (resources/<robot>/problems.pkl) and convert them
to the robometrics-style dict the seed-strategy benchmark uses.

These are the tried-and-tested problem sets from the VAMP / pRRTC benchmarks (VAMP-RRTC solves
them ~100%), defined for all four robots in VAMP's world frame -- which is exactly the frame our
cuRobo configs use (base_link = the URDF root / offset_link), so obstacles feed both planners
unchanged. Each converted problem exposes:

    robot, scene, start, goal (= goals[0]), goals (full goal set), obstacles (robometrics dict)

``goals`` is a goal *set*; VAMP plans to the set, while cuRobo targets the Cartesian pose of a
representative goal (goals[0]) and its seeder plans VAMP-RRTC toward cuRobo's IK solutions.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKL = os.path.join(_ROOT, "vamp", "resources", "{robot}", "problems.pkl")


def _quat_xyzw_to_wxyz(q):
    x, y, z, w = q
    return [w, x, y, z]


def _obstacles_to_robometrics(data: Dict) -> Dict:
    """Convert VAMP pkl obstacle lists (box/cylinder/sphere) to robometrics obstacle dict."""
    obs = {"cuboid": {}, "cylinder": {}, "sphere": {}}
    for b in data.get("box", []):
        obs["cuboid"][b["name"]] = {
            "pose": list(b["position"]) + _quat_xyzw_to_wxyz(b["orientation_quat_xyzw"]),
            "dims": [2 * h for h in b["half_extents"]],
        }
    for c in data.get("cylinder", []):
        obs["cylinder"][c["name"]] = {
            "pose": list(c["position"]) + _quat_xyzw_to_wxyz(c["orientation_quat_xyzw"]),
            "radius": c["radius"], "height": c["length"],
        }
    for s in data.get("sphere", []):
        obs["sphere"][s["name"]] = {
            "pose": list(s["position"]) + [1.0, 0.0, 0.0, 0.0],
            "radius": s["radius"],
        }
    return {k: v for k, v in obs.items() if v}


def load_mbm(robot: str, per_scene: int = 0, scenes: List[str] = None) -> List[Dict]:
    """Return converted problems for ``robot``. ``per_scene`` caps problems per scene (0 = all)."""
    with open(_PKL.format(robot=robot), "rb") as f:
        raw = pickle.load(f)
    out: List[Dict] = []
    for scene, pset in raw["problems"].items():
        if scenes and scene not in scenes:
            continue
        kept = 0
        for data in pset:
            if not data.get("valid", False):
                continue
            goals = [list(g) for g in data["goals"]]
            out.append({
                "robot": robot, "scene": scene,
                "start": list(data["start"]),
                "goal": goals[0],
                "goals": goals,
                "obstacles": _obstacles_to_robometrics(data),
            })
            kept += 1
            if per_scene and kept >= per_scene:
                break
    return out


def scene_list(robot: str) -> List[str]:
    with open(_PKL.format(robot=robot), "rb") as f:
        return list(pickle.load(f)["problems"].keys())
