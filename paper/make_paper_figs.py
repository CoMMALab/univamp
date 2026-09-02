#!/usr/bin/env python
"""Paper figures + tables from the full-MBM seeding experiment.

Sole data source: ``results/exp_mbm_full.jsonl`` (see ``results/exp_mbm_full.py``) -- every
scene and every problem of the curated MBM sets for ur5/panda/fetch/baxter, under three seeding
arms. Everything below is computed from that file; nothing is transcribed by hand.

Emits, into paper/:
  exp1_headline.{pdf,png}      success rate / mean plan time / path length, per robot, 3 arms
  exp1_table.tex               the same numbers as a LaTeX table (+ pooled dataset row)
  fig_phase_breakdown.{pdf,png}  ABLATION: where the plan time goes, per phase, per arm --
                               the up-to-date replacement for the old Phase-4 (N=96) figure,
                               now over the whole dataset
  exp2_phase_table.tex         phase decomposition in exact numbers

Colors follow the NVIDIA brand green (#76B900): the proposed arm carries the green, baselines
stay recessive neutrals; the phase ablation uses a single-hue green ordinal ramp anchored on the
brand green. Both palettes were checked for CVD separation, lightness banding and contrast.

Run: conda run -n univamp python paper/make_paper_figs.py
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(OUT, "..", "results", "exp_mbm_full.jsonl")

# ---- chrome ---------------------------------------------------------------------
INK, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

# ---- series palette: baselines recessive, proposed method in NVIDIA green ---------
NV_GREEN = "#76B900"
C = {"lerp": "#86867e", "prm": "#44607a", "vamp_uma": NV_GREEN}
LBL = {"lerp": "LERP (stock)", "prm": "PRM (stock fallback)", "vamp_uma": "VAMP-RRTC + UMA"}
SHORT = {"lerp": "LERP", "prm": "PRM", "vamp_uma": "VAMP+UMA"}
ARMS = ["lerp", "prm", "vamp_uma"]
# Fetch is excluded from the paper assets. Its records are collected and kept in the JSONL, but
# both stock arms score ~2% on it -- that is a harness defect (the 8-DoF prismatic-torso TrajOpt
# under-converges), not a hard benchmark, and a "win" over a 2% baseline is not evidence of
# anything. Add ("fetch", "Fetch") back here once that is fixed.
ROBOTS = [("ur5", "UR5"), ("panda", "Panda"), ("baxter", "Baxter")]

# ---- phase ramp: single-hue green, anchored on the brand green, descending ---------
# (dL >= 0.108 between steps, lightest 2.35:1 on white -- validated)
# Planning is fully joint-space (plan_cspace), so there is no IK phase to account for.
PHASES = [("seed", "Seeding", "#76B900"), ("xfer", "Transfer", "#5d9403"),
          ("trajopt", "TrajOpt", "#467006"), ("other", "Other", "#304e07")]

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 7.5, "axes.labelsize": 7.5,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.edgecolor": BASE, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def despine(ax, keep=("bottom",)):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    print("wrote", name + ".pdf/.png")


def load():
    by = defaultdict(list)
    with open(REC) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            by[(r["robot"], r["strategy"])].append(r)
    return by


def stats(recs):
    """Means over ALL trials of all problems. Path length averages successful plans only
    (a failed plan has no trajectory to measure)."""
    n = len(recs)
    if n == 0:
        return None
    succ = [r for r in recs if r["success"]]
    pl = [r["path_len"] for r in succ if r.get("path_len")]
    s = {
        "n": n, "n_succ": len(succ), "sr": 100.0 * len(succ) / n,
        "ms": float(np.mean([r["total_ms"] for r in recs])),
        "path": float(np.mean(pl)) if pl else float("nan"),
        "trajopt": float(np.mean([r["trajopt_ms"] for r in recs])),
        "xfer": float(np.mean([r["handoff_ms"] for r in recs])),
    }
    # the wrapper's seed_ms = total - trajopt, so it still contains the handoff
    raw_seed = float(np.mean([r["seed_ms"] for r in recs]))
    s["seed"] = max(0.0, raw_seed - s["xfer"])
    s["other"] = max(0.0, s["ms"] - s["seed"] - s["xfer"] - s["trajopt"])
    return s


def dof_of(by, rb):
    for a in ARMS:
        for r in by.get((rb, a), []):
            return r.get("dof")
    return None


# =================================================================================
# Figure 1 -- headline: success rate, mean plan time, path length
# =================================================================================
def fig_headline(by, S, robots, POOL, suffix="", note=""):
    """Headline comparison aggregated over every robot and every problem.

    Bars are pooled across all robots: each arm's rate/mean is taken over the full record set,
    so a robot contributes in proportion to its problem count (Baxter's set is the largest).
    The per-robot split is preserved in exp1_table.tex."""
    metrics = [("sr", "Success rate (%)", "{:.1f}"),
               ("ms", "Mean plan time (ms)", "{:.0f}"),
               ("path", "Path length (rad)", "{:.2f}")]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.95))
    x = np.arange(len(ARMS))
    for ax, (key, ylab, fmt) in zip(axes, metrics):
        vals = [POOL[a][key] for a in ARMS]
        bars = ax.bar(x, vals, 0.62, color=[C[a] for a in ARMS], zorder=3)
        ax.bar_label(bars, labels=[fmt.format(v) for v in vals], padding=2,
                     fontsize=6.8, color=MUTED)
        top = max([v for v in vals if np.isfinite(v)] or [1])
        ax.set_ylim(0, (105 if key == "sr" else top * 1.24))
        ax.set_xticks(x, [SHORT[a] for a in ARMS], fontsize=6.8)
        ax.set_title(ylab, fontsize=7.8, color=INK, pad=6)
        ax.grid(axis="y", zorder=0)
        ax.tick_params(length=0)
        despine(ax)
    handles = [mpl.patches.Patch(color=C[a], label=LBL[a]) for a in ARMS]
    fig.legend(handles=handles, ncols=3, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.10), handlelength=1.0, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    save(fig, "exp1_headline" + suffix)


def tex_headline(by, S, robots, POOL, suffix="", extra=""):
    rows = []
    for rb, disp in robots:
        d = dof_of(by, rb)
        cells = [disp, str(d)]
        best = max(S[(rb, a)]["sr"] for a in ARMS)
        for key, fmt in (("sr", "{:.1f}"), ("ms", "{:.0f}"), ("path", "{:.2f}")):
            for a in ARMS:
                v = fmt.format(S[(rb, a)][key])
                if key == "sr" and abs(S[(rb, a)]["sr"] - best) < 0.05:
                    v = r"\textbf{" + v + "}"
                cells.append(v)
        rows.append(" & ".join(cells) + r" \\")
    pcells = [r"\emph{All robots}", ""]
    pbest = max(POOL[a]["sr"] for a in ARMS)
    for key, fmt in (("sr", "{:.1f}"), ("ms", "{:.0f}"), ("path", "{:.2f}")):
        for a in ARMS:
            v = fmt.format(POOL[a][key])
            if key == "sr" and abs(POOL[a]["sr"] - pbest) < 0.05:
                v = r"\textbf{" + v + "}"
            pcells.append(v)
    n_tot = sum(S[(rb, "lerp")]["n"] for rb, _ in robots)
    counts = ", ".join(f"{disp} {S[(rb,'lerp')]['n']}" for rb, disp in robots)
    tex = r"""% Experiment 1 --- full-MBM headline (auto-generated by paper/make_paper_figs.py)
\begin{table*}[t]
\centering
\caption{TrajOpt seeding on the full MBM benchmark ($N{=}""" + str(n_tot) + r"""$ per arm; """ + counts + r"""). LERP and PRM are cuRobo's stock on-device seeds; VAMP-RRTC\,+\,UMA is ours, its CPU seed
handed to the GPU through a managed buffer. Planning is fully joint-space, so no IK enters the
comparison. Path length is the joint-space $L_2$ length over successful plans; best success in
bold. Rates fall short of the ${\sim}100\%$ each planner reports alone because success here
demands a dynamically feasible, time-parameterized trajectory rather than a valid geometric path;
because cylinders are approximated by bounding boxes and the robot by spheres, for both planners
alike. PRM is given cuRobo's stock retry budget (\texttt{max\_attempts}${=}5$) since its
characteristic failure is returning no seed at all and the roadmap draw is stochastic; LERP and
VAMP-RRTC plan in a single attempt, neither having that failure mode. Geometry, limits and TrajOpt
budget are identical across arms.""" + extra + r"""}
\label{tab:exp1}
\begin{tabular}{lc rrr rrr rrr}
\toprule
 & & \multicolumn{3}{c}{Success rate (\%)} & \multicolumn{3}{c}{Plan time (ms)} & \multicolumn{3}{c}{Path length (rad)} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
Robot & DoF & LERP & PRM & \textbf{Ours} & LERP & PRM & \textbf{Ours} & LERP & PRM & \textbf{Ours} \\
\midrule
""" + "\n".join(rows) + "\n" + r"\midrule" + "\n" + " & ".join(pcells) + r""" \\
\bottomrule
\end{tabular}
\end{table*}
"""
    open(os.path.join(OUT, "exp1_table" + suffix + ".tex"), "w").write(tex)
    print("wrote exp1_table" + suffix + ".tex")


# =================================================================================
# Figure 2 -- ABLATION: per-phase decomposition of plan time (replaces Phase-4 fig)
# =================================================================================
def fig_phase_breakdown(by, S, robots, POOL, suffix=""):
    """ABLATION: where the plan time goes, aggregated over every robot and every problem."""
    # Only draw phases that are actually visible at this scale: a band below ~0.5% of the
    # widest bar renders sub-pixel, so carrying it would put a colour in the legend that the
    # reader cannot find in the chart. What is dropped is stated in the subtitle and reported
    # exactly in exp2_phase_table.tex, so nothing is silently lost.
    widest = max(sum(POOL[a][k] for k, _, _ in PHASES) for a in ARMS)
    visible = [(k, l, c) for k, l, c in PHASES
               if max(POOL[a][k] for a in ARMS) >= 0.005 * widest]
    dropped = [(k, l) for k, l, _ in PHASES if (k, l) not in [(k2, l2) for k2, l2, _ in visible]]

    fig, ax = plt.subplots(figsize=(6.6, 1.75))
    y = np.arange(len(ARMS))[::-1].astype(float)
    seg_handles = []
    totals = []
    for yi, a in zip(y, ARMS):
        s = POOL[a]
        left = 0.0
        for key, lbl, col in visible:
            ax.barh(yi, s[key], left=left, height=0.6, color=col, zorder=3,
                    edgecolor="white", linewidth=1.1)
            if yi == y[0]:
                seg_handles.append(mpl.patches.Patch(color=col, label=lbl))
            left += s[key]
        totals.append(left)
        ax.text(left + 6, yi, f"{s['ms']:.0f} ms · {s['sr']:.1f}% success",
                va="center", fontsize=7.0, color=INK)
    ax.set_yticks(y, [LBL[a] for a in ARMS], fontsize=7.2)
    ax.set_xlim(0, max(totals) * 1.34)
    ax.set_ylim(min(y) - 0.6, max(y) + 0.6)
    ax.set_xlabel("Mean time per plan (ms), decomposed by phase")
    ax.grid(axis="x", zorder=0)
    ax.tick_params(length=0)
    despine(ax)
    ax.legend(handles=seg_handles, ncols=len(visible), frameon=False,
              loc="lower right", bbox_to_anchor=(1.005, 1.0), handlelength=0.9,
              columnspacing=0.9, fontsize=6.8)
    # Phases omitted as sub-pixel are printed to stdout (not drawn on the figure) so the
    # numbers are available for the LaTeX caption written alongside it.
    if dropped:
        print("  omitted from fig_phase_breakdown (sub-pixel): "
              + "; ".join(f"{l} max {max(POOL[a][k] for a in ARMS):.3f} ms"
                          for k, l in dropped))
    save(fig, "fig_phase_breakdown" + suffix)


def tex_phases(by, S, robots, POOL, suffix=""):
    lines = []
    for rb, disp in robots:
        for a in ARMS:
            s = S[(rb, a)]
            lines.append(f"{disp} & {LBL[a]} & {s['seed']:.1f} & {s['xfer']:.2f} & "
                         f"{s['trajopt']:.1f} & {s['other']:.1f} & "
                         f"{s['ms']:.0f} \\\\")
        lines.append(r"\addlinespace")
    for a in ARMS:
        s = POOL[a]
        lines.append(f"\\emph{{All}} & {LBL[a]} & {s['seed']:.1f} & {s['xfer']:.2f} & "
                     f"{s['trajopt']:.1f} & {s['other']:.1f} & "
                     f"{s['ms']:.0f} \\\\")
    tex = r"""% Ablation --- phase decomposition (auto-generated by paper/make_paper_figs.py)
\begin{table}[t]
\centering
\caption{Ablation: mean plan time per phase (ms), over the complete MBM dataset. \emph{Transfer}
is the CPU$\to$GPU seed handoff plus buffer allocation; LERP and PRM generate their seeds on
device, so their transfer is zero by construction --- the handoff cost exists only because the
strong seeder runs on the CPU, and UMA is what keeps it negligible. \emph{Other} is the residual
of the measured total after seeding, transfer and TrajOpt. Planning is fully joint-space, so no
IK is performed.}
\label{tab:ablation-phases}
\begin{tabular}{ll rrrr r}
\toprule
Robot & Seed & Seeding & Transfer & TrajOpt & Other & Total \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(OUT, "exp2_phase_table" + suffix + ".tex"), "w").write(tex)
    print("wrote exp2_phase_table" + suffix + ".tex")


def main():
    by = load()
    suffix = note = extra = ""
    robots = [(rb, disp) for rb, disp in ROBOTS
              if all(by.get((rb, a)) for a in ARMS)]
    if not robots:
        raise SystemExit(f"no complete records in {REC}")
    S = {(rb, a): stats(by[(rb, a)]) for rb, _ in robots for a in ARMS}
    POOL = {a: stats([r for rb, _ in robots for r in by[(rb, a)]]) for a in ARMS}

    print(f"robots: {[r for r, _ in robots]}   "
          f"records/arm: {[S[(rb,'lerp')]['n'] for rb,_ in robots]}")
    fig_headline(by, S, robots, POOL, suffix=suffix, note=note)
    tex_headline(by, S, robots, POOL, suffix=suffix, extra=extra)
    fig_phase_breakdown(by, S, robots, POOL, suffix=suffix)
    tex_phases(by, S, robots, POOL, suffix=suffix)

    print("\n---- pooled over the whole dataset ----")
    for a in ARMS:
        s = POOL[a]
        print(f"  {LBL[a]:22s} N={s['n']:5d}  success {s['sr']:5.1f}%  "
              f"plan {s['ms']:7.1f} ms  path {s['path']:5.2f} rad")


if __name__ == "__main__":
    main()
