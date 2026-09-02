# UniVAMP

VAMP (CPU sampling-based planner) seeding cuRobo (GPU trajectory optimization) on the NVIDIA
Jetson Thor's unified memory. VAMP-RRTC replaces cuRobo's PRM graph planner as the TrajOpt
*seeder*: the CPU path lands in a `cudaMallocManaged` buffer and a fused kernel expands it
zero-copy into cuRobo's device seed tensor, with cuRobo's own allocator and CUDA graphs left
untouched.

Benchmark results and the paper assets live in [results/README.md](results/README.md).

## Install

Jetson Thor (sm_110, CUDA 13, JetPack R38/7). The env **must be Python 3.12** — Thor CUDA torch
wheels exist only for cp312.

```bash
git clone <this repo> univamp && cd univamp
conda create -n univamp python=3.12 && conda activate univamp

# torch for Thor. Do NOT add --extra-index-url pypi.org, or pip pulls the CPU-only wheel.
pip install torch==2.10.0 --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130

# runtime libs the wheel does not pull, plus cuda.core for cuRobo's NVRTC backend
pip install nvpl nvidia-cudss-cu13 nvidia-nccl-cu13 nvidia-cusparselt-cu13 \
            cuda-core cuda-bindings cuda-pathfinder
pip install robometrics pin            # MBM dataset + pinocchio
pip install viser yourdfpy             # only needed for the interactive demo

# both planners editable, so C++/CUDA edits rebuild into the env
pip install -e vamp   --no-build-isolation
pip install -e curobo --no-build-isolation --no-deps

bash univamp/build.sh    # libunivamp_managed.so + libunivamp_seedexpand.so (hardcodes sm_110)
```

`LD_LIBRARY_PATH` must carry NVPL, the pip `nvidia-*` libs and
`/usr/local/cuda-13.0/targets/sbsa-linux/lib`, or `import torch` fails on
`libnvpl_lapack*`/`libcudss`. Put that in `$CONDA_PREFIX/etc/conda/activate.d/`.

Verify:

```bash
python -c "import torch, vamp, curobo; print(torch.cuda.is_available(), torch.version.cuda)"
```

## Interactive demo

[`examples/viser_mbm_demo.py`](examples/viser_mbm_demo.py) solves one MBM problem with the full
VAMP-seeded pipeline and serves the scene, the robot and the optimized trajectory in the browser
with a play button and the measured solve time.

```bash
python examples/viser_mbm_demo.py --robot panda --problem cage --index 1
```

Then open the printed `http://localhost:<port>`.

| Flag | Meaning |
|---|---|
| `--robot` | `ur5` (6 DoF), `panda` (7), `fetch` (8), `baxter` (14) |
| `--problem` | MBM scene name; `--list` prints the scenes for a robot |
| `--index` | which problem within that scene (default 0) |
| `--multi-seed` | fill every TrajOpt seed slot with a diverse VAMP-RRTC path |
| `--port` | viser port (default 8080) |
| `--budget` | LBFGS TrajOpt iteration budget (default 200) |

The GUI panel reports total solve time in ms broken into VAMP seed generation, the UMA handoff,
and TrajOpt; **Play** animates the trajectory at the planner's own time parameterization, with
speed and loop controls and a waypoint slider for scrubbing. The orange spline is the
end-effector path. Obstacles are drawn as the planner sees them (cuRobo's OBB world, so
cylinders appear as their bounding boxes).

Not every MBM problem is solvable in one attempt — a failed solve still serves the scene at the
start configuration and reports the failure status, so try another `--index`.

## Layout

| Path | Role |
|---|---|
| `univamp/seeder.py` | VAMP-RRTC seeder + `attach_vamp_seeder` (patches cuRobo's graph-seed hook) |
| `univamp/bridge.py` | robometrics problem → VAMP environment |
| `univamp/curobo_loader.py` | robot-agnostic cuRobo `MotionPlanner` factory |
| `univamp/csrc/` | managed allocator + zero-copy seed-expansion CUDA kernels |
| `univamp/robot_assets/` | generated cuRobo configs from VAMP's spherized URDFs |
| `examples/` | the interactive viser demo |
| `results/` | the full-MBM benchmark and its write-up |
| `vamp/`, `curobo/` | the two upstream repos (editable installs) |
