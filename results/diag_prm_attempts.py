"""Is PRM's no-seed failure a property of cuRobo's graph planner, or an artifact of
max_attempts=1?

For a sample of UR5 problems we ask the graph planner for a seed under three settings:
  (a) one attempt    -- what the headline experiment uses
  (b) many attempts  -- does retrying find a roadmap path it missed?
and separately check whether two back-to-back calls on an identical problem even differ
(the shipped config pins sampler_seed: 0, so retries may be deterministic).
"""
import os, sys, time
sys.argv = ["b"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BD = os.path.join(ROOT, "curobo", "benchmark")
sys.path.insert(0, BD); sys.path.insert(0, ROOT); os.chdir(BD)

import torch
from copy import deepcopy
from curobo._src.geom.types import SceneCfg
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import setup_curobo_logger
from univamp.curobo_loader import build_motion_planner
from univamp import bridge
bridge.CYLINDERS_AS_BOXES = True
from results.mbm_problems import load_mbm

N = int(os.environ.get("N", "120"))
setup_curobo_logger("error")
probs = load_mbm("ur5")[:N]
mg = build_motion_planner("ur5", use_cuda_graph=True)
mg.warmup(enable_graph=True)

def seed_exists(mg, ss, gs):
    """Ask the graph planner directly for a seed; True if it returned one."""
    ns = mg.trajopt_solver.config.num_seeds
    gc = gs.position.view(1, 1, -1).repeat(1, ns, 1)
    return mg._get_graph_seed_trajectories(ss, gc) is not None

det_same = det_tot = 0
for label, attempts in (("1 attempt", 1), ("8 attempts", 8)):
    ok = noseed = 0
    for p in probs:
        world = SceneCfg.create(deepcopy(p["obstacles"])).get_obb_world()
        mg.scene_collision_checker.clear_cache(); mg.update_world(world)
        ss = JointState.from_position(mg.device_cfg.to_device([p["start"]]))
        gs = JointState.from_position(mg.device_cfg.to_device([p["goal"]]))
        mg.reset_seed()
        r = mg.plan_cspace(gs, ss, max_attempts=attempts, enable_graph_attempt=0)
        got = r is not None and torch.count_nonzero(r.success) > 0
        ok += got
        # did the graph ever yield a seed for this problem?
        mg.reset_seed()
        a = seed_exists(mg, ss, gs)
        if not a:
            noseed += 1
        if attempts == 1:  # determinism probe: same problem, immediate repeat
            b = seed_exists(mg, ss, gs)
            det_tot += 1; det_same += (a == b)
    print(f"{label:12s}: success {ok}/{len(probs)} ({100*ok/len(probs):.1f}%), "
          f"graph produced NO seed on {noseed}/{len(probs)} ({100*noseed/len(probs):.1f}%)",
          flush=True)

print(f"\ndeterminism: back-to-back graph calls agreed on {det_same}/{det_tot} problems "
      f"({100*det_same/det_tot:.0f}%)")
sys.stdout.flush()
os._exit(0)
