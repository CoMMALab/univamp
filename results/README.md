# UniVAMP results — full-MBM seeding experiment

Prototype integrating **VAMP** (CPU sampling-based planner) with **cuRobo** (GPU trajectory
optimization) on the **NVIDIA Jetson Thor** UMA. VAMP-RRTC replaces cuRobo's PRM graph planner as
the TrajOpt *seeder*: the CPU path lands in a `cudaMallocManaged` buffer and a fused kernel
expands it zero-copy into cuRobo's device seed tensor.

Everything in `../paper/` is generated from a single data file — `exp_mbm_full.jsonl` — by
`../paper/make_paper_figs.py`. Nothing is transcribed by hand.

## Headline (full MBM: UR5 689 + Panda 699 + Baxter 1470 = 2858 problems per arm)

| Robot | DoF | Success: LERP / PRM / **Ours** | Plan ms: LERP / PRM / **Ours** |
|---|---|---|---|
| UR5 | 6 | 27.0 / 26.3 / **38.2** | 145 / 470 / **148** |
| Panda | 7 | 83.3 / 78.1 / **98.9** | 376 / 943 / **372** |
| Baxter | 14 | 78.6 / 78.7 / **79.3** | 321 / 539 / **346** |
| *All* | | 67.3 / 65.9 / **74.1** | 292 / 621 / **305** |

Seeding phase alone (pooled): PRM 266.4 ms → **VAMP-RRTC 14.7 ms**, with the CPU→GPU transfer at
**0.41 ms** and CUDA graphs retained. Success rates sit below the ~100% each planner reports
alone because success here demands a dynamically feasible, time-parameterized trajectory, not a
valid geometric path.

Fetch (8 DoF) records are collected by the script but excluded from the paper: both stock arms
score ~2% on it, a harness defect in the prismatic-torso TrajOpt rather than a hard benchmark.

## Files

| File | Role |
|---|---|
| `exp_mbm_full.py` | the experiment — one pass over all of MBM × three seeding arms |
| `exp_mbm_full.jsonl` | its per-problem records; **the sole input to the paper assets** |
| `mbm_problems.py` | shared loader: VAMP's `resources/<robot>/problems.pkl` → robometrics dicts |
| `diag_prm_attempts.py` | diagnostic behind PRM's asymmetric retry budget (see below) |
| `exp_mbm_prm1attempt.jsonl` | superseded: same experiment with PRM held to one attempt |
| `exp_mbm_posegoal.jsonl` | superseded: earlier `plan_pose` revision, which inserted an IK stage |

PRM gets cuRobo's stock `max_attempts=5` while LERP and VAMP-RRTC get one attempt: PRM's dominant
failure is returning *no seed at all* and its roadmap draw is stochastic, so retries repair a
failure mode the other two arms cannot suffer. `diag_prm_attempts.py` measures that (back-to-back
graph calls disagree ~8% of the time; retries lift PRM from 5.0% to 19.2% on 120 UR5 problems).

## Reproducing

### 1. Environment (Jetson Thor, sm_110, CUDA 13, JetPack R38/7)

The `univamp` conda env **must be Python 3.12** — Thor CUDA torch wheels exist only for cp312.

```bash
conda create -n univamp python=3.12 && conda activate univamp

# torch for Thor. Do NOT add --extra-index-url pypi.org, or pip pulls the CPU-only wheel.
pip install torch==2.10.0 --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130

# runtime libs the wheel does not pull, plus cuda.core for cuRobo's NVRTC backend
pip install nvpl nvidia-cudss-cu13 nvidia-nccl-cu13 nvidia-cusparselt-cu13 \
            cuda-core cuda-bindings cuda-pathfinder
pip install robometrics pin            # MBM dataset + pinocchio

# both planners editable, so C++/CUDA edits rebuild into the env
pip install -e vamp   --no-build-isolation
pip install -e curobo --no-build-isolation --no-deps
```

`LD_LIBRARY_PATH` must carry NVPL, the pip `nvidia-*` libs and
`/usr/local/cuda-13.0/targets/sbsa-linux/lib`, or `import torch` fails on
`libnvpl_lapack*`/`libcudss`. Put that in `$CONDA_PREFIX/etc/conda/activate.d/`.

Verify before going further:

```bash
python -c "import torch, vamp, curobo; print(torch.cuda.is_available(), torch.version.cuda)"
```

### 2. Build the CUDA shared libraries

```bash
bash univamp/build.sh    # libunivamp_managed.so (managed allocator) + libunivamp_seedexpand.so
```

`build.sh` hardcodes `sm_110`; change `-gencode` for another target.

### 3. Run the experiment

```bash
python results/exp_mbm_full.py                    # all robots, all arms, ~all of MBM
python results/exp_mbm_full.py --robots panda --limit 5   # quick smoke test
```

Records append to `results/exp_mbm_full.jsonl` incrementally and already-present
`(robot, scene, problem, arm)` records are skipped, so an interrupted run resumes by re-running
the same command. **To collect from scratch, move the existing JSONL aside first** — otherwise
the run finds every record present and does nothing. Paths are derived from the script location,
so a clone anywhere works. Full dataset is a multi-hour run on Thor.

### 4. Regenerate the paper assets

```bash
python paper/make_paper_figs.py
```

Writes `exp1_headline.{pdf,png}`, `exp1_table.tex`, `fig_phase_breakdown.{pdf,png}` and
`exp2_phase_table.tex` into `paper/`, and prints the pooled aggregates. `paper/exp1_text.tex`
is prose and is maintained by hand — update its numbers if the data changes.
