"""Generate cuRobo robot configs for the VAMP robots (ur5, fetch, baxter, panda).

cuRobo only ships a franka (panda) config, but the seed-strategy benchmark needs to run
linear / PRM / VAMP-RRTC seeding across several robots. VAMP ships *spherized* URDFs for all
four robots: a clean kinematic tree whose collision geometry is a set of per-link spheres
(``<collision><geometry><sphere>`` + ``<origin>``). We reuse exactly that geometry so cuRobo
and VAMP plan against an *identical* robot collision model and the comparison is fair.

For each robot we emit, under ``univamp/robot_assets/<robot>/``:
  * ``<robot>_curobo.urdf`` -- the spherized URDF with ``<visual>``/``<collision>`` stripped
    (cuRobo only needs the kinematic tree; collision comes from the YAML spheres below, and the
    package:// mesh refs don't resolve here).
  * ``<robot>.yml``         -- a cuRobo ``robot_cfg`` with collision_spheres parsed from the URDF,
    cspace joint order/limits taken from the VAMP module (so seed configs line up 1:1), and
    adjacency-based self_collision_ignore.

The shared world frame is the URDF root link (e.g. ur5's ``offset_link``), which is also the
frame VAMP's ``fk`` uses -- so obstacle coordinates are identical between the two planners.
"""
from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np
import vamp
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_ROOT = os.path.join(_HERE, "robot_assets")

# cuRobo tool frame per robot (the arm tip VAMP's end_effector() reports).
TOOL_FRAME = {
    "ur5": "ee_link",
    "fetch": "gripper_link",
    "baxter": "right_gripper",
    "panda": "panda_grasptarget",
}
VAMP_SPHERIZED = {
    r: os.path.join(_HERE, "..", "vamp", "resources", r, f"{r}_spherized.urdf")
    for r in TOOL_FRAME
}


def _float_list(s: str) -> List[float]:
    return [float(x) for x in s.replace(",", " ").split()]


def _parse_urdf(path: str) -> ET.ElementTree:
    return ET.parse(path)


def _root_link(root: ET.Element) -> str:
    links = {l.get("name") for l in root.findall("link")}
    children = {j.find("child").get("link") for j in root.findall("joint")}
    (r,) = links - children
    return r


def _parse_spheres(root: ET.Element) -> Dict[str, List[dict]]:
    """link name -> [{center:[x,y,z], radius:r}, ...] from <collision><sphere>."""
    out: Dict[str, List[dict]] = {}
    for link in root.findall("link"):
        spheres = []
        for col in link.findall("collision"):
            sph = col.find("geometry/sphere")
            if sph is None:
                continue
            origin = col.find("origin")
            xyz = _float_list(origin.get("xyz")) if origin is not None and origin.get("xyz") \
                else [0.0, 0.0, 0.0]
            spheres.append({"center": [round(v, 6) for v in xyz],
                            "radius": round(float(sph.get("radius")), 6)})
        if spheres:
            out[link.get("name")] = spheres
    return out


def _adjacency_ignore(root: ET.Element, sphere_links: List[str]) -> Dict[str, List[str]]:
    """Self-collision ignore: for each link, ignore its parent/child/sibling collision links
    (1-hop in the joint tree) so adjacent links never spuriously collide in trajopt."""
    parent: Dict[str, str] = {}
    kids: Dict[str, List[str]] = {}
    for j in root.findall("joint"):
        p = j.find("parent").get("link")
        c = j.find("child").get("link")
        parent[c] = p
        kids.setdefault(p, []).append(c)
    sset = set(sphere_links)
    ignore: Dict[str, List[str]] = {}
    for link in sphere_links:
        nbrs = set()
        # walk up to the nearest sphere-bearing ancestor and down to sphere-bearing descendants
        p = parent.get(link)
        while p is not None and p not in sset:
            p = parent.get(p)
        if p is not None:
            nbrs.add(p)
        stack = list(kids.get(link, []))
        seen = set()
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            if c in sset:
                nbrs.add(c)
            else:
                stack.extend(kids.get(c, []))
        nbrs.discard(link)
        if nbrs:
            ignore[link] = sorted(nbrs)
    return ignore


def _strip_geometry(tree: ET.ElementTree, active: set) -> ET.ElementTree:
    """Return a copy of the URDF with <visual>/<collision> removed (kinematics only) and the
    *active* joints' velocity limits raised to realistic values. Some shipped URDFs cap joint
    velocity very conservatively (ur5: 0.5 rad/s; fetch torso: 0.1 m/s); cuRobo's TrajOpt must
    time-parameterize within these limits over a fixed horizon, so such caps make otherwise
    feasible motions unsolvable. We floor revolute vel at 2.0 rad/s and prismatic at 0.2 m/s."""
    t = copy.deepcopy(tree)
    for link in t.getroot().findall("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
    for j in t.getroot().findall("joint"):
        if j.get("name") not in active:
            continue
        lim = j.find("limit")
        if lim is None:
            continue
        floor = 0.2 if j.get("type") == "prismatic" else 2.0
        try:
            v = float(lim.get("velocity", "0"))
        except (TypeError, ValueError):
            v = 0.0
        if v < floor:
            lim.set("velocity", str(floor))
    return t


def generate(robot: str) -> Tuple[str, str]:
    """Write the cleaned URDF + cuRobo YAML for ``robot``. Returns (urdf_path, yaml_path)."""
    tree = _parse_urdf(VAMP_SPHERIZED[robot])
    root = tree.getroot()
    vmod = getattr(vamp, robot)

    base_link = _root_link(root)
    spheres = _parse_spheres(root)
    sphere_links = list(spheres.keys())
    ignore = _adjacency_ignore(root, sphere_links)

    joint_names = list(vmod.joint_names())
    lower = [float(x) for x in vmod.lower_bounds()]
    upper = [float(x) for x in vmod.upper_bounds()]
    retract = [round(0.5 * (lo + hi), 4) for lo, hi in zip(lower, upper)]

    out_dir = os.path.join(ASSET_ROOT, robot)
    os.makedirs(out_dir, exist_ok=True)
    urdf_out = os.path.join(out_dir, f"{robot}_curobo.urdf")
    _strip_geometry(tree, set(joint_names)).write(urdf_out)

    robot_cfg = {
        "robot_cfg": {
            "kinematics": {
                "format_version": 2.0,
                "urdf_path": urdf_out,
                "asset_root_path": "",
                "base_link": base_link,
                "tool_frames": [TOOL_FRAME[robot]],
                "collision_link_names": sphere_links,
                "collision_spheres": spheres,
                "collision_sphere_buffer": 0.0,
                "self_collision_ignore": ignore,
                "self_collision_buffer": {l: 0.0 for l in sphere_links},
                "mesh_link_names": [],
                "use_global_cumul": True,
                "cspace": {
                    "joint_names": joint_names,
                    "null_space_weight": [1.0] * len(joint_names),
                    "cspace_distance_weight": [1.0] * len(joint_names),
                    "max_acceleration": 15.0,
                    "max_jerk": 500.0,
                    "default_joint_position": retract,
                },
            },
        }
    }
    yaml_out = os.path.join(out_dir, f"{robot}.yml")
    with open(yaml_out, "w") as f:
        yaml.safe_dump(robot_cfg, f, sort_keys=False, default_flow_style=None)
    return urdf_out, yaml_out


if __name__ == "__main__":
    import sys
    robots = sys.argv[1:] or ["ur5", "fetch", "baxter", "panda"]
    for r in robots:
        u, y = generate(r)
        nsph = sum(len(v) for v in _parse_spheres(_parse_urdf(VAMP_SPHERIZED[r]).getroot()).values())
        print(f"{r:8s} dof={getattr(vamp, r).dimension():2d} spheres={nsph:3d}  -> {y}")
