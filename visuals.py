"""Extended figures (saved to outputs/extended/). All data shown is SYNTHETIC."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

import main_timetable_ga as core
from main_timetable_ga import N_DAYS, N_PERIODS, N_SLOTS, SYN

COL = {"GA": "#1f77b4", "GA (DSATUR-seeded)": "#17becf", "GA + local search": "#2ca02c", "Simulated annealing": "#ff7f0e",
       "Hill climbing": "#9467bd", "Random search": "#7f7f7f"}


def _save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=220)
    plt.close(fig)


def _box(ax, data, labels, horizontal=False, colors=None):
    bp = ax.boxplot(data, vert=not horizontal, patch_artist=True, widths=0.6)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor((colors[i] if colors else "#9ecae1"))
        b.set_alpha(0.75)
    for m in bp["medians"]:
        m.set_color("black")
    (ax.set_yticklabels if horizontal else ax.set_xticklabels)(labels)


def algorithm_boxplot(res, greedy_fit, out_dir):
    names = list(res)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    _box(ax, [res[n] for n in names], names, colors=[COL.get(n, "#9ecae1") for n in names])
    ax.axhline(greedy_fit, color="crimson", ls="--", lw=1.4,
               label=f"Greedy DSATUR (deterministic): {greedy_fit:.0f}")
    ax.set_yscale("log")
    ax.set_ylabel("Final penalty (log scale, lower is better)")
    ax.set_title(f"Algorithm comparison, 10 seeds, ~9000 evaluations each ({SYN})", fontsize=10)
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=15)
    _save(fig, out_dir, "fig5_algorithm_comparison.png")


def algorithm_convergence(grid, traces, greedy_fit, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for n, t in traces.items():
        med = np.median(t, axis=0)
        ax.plot(grid, med, lw=2, color=COL.get(n), label=n)
        ax.fill_between(grid, np.percentile(t, 25, axis=0), np.percentile(t, 75, axis=0),
                        color=COL.get(n), alpha=0.18)
    ax.axhline(greedy_fit, color="crimson", ls="--", lw=1.2, label="Greedy DSATUR")
    ax.set_yscale("log")
    ax.set_xlabel("Fitness evaluations")
    ax.set_ylabel("Best-so-far penalty (log scale)")
    ax.set_title(f"Convergence per evaluation: median and interquartile band, 10 seeds ({SYN})",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig6_algorithm_convergence.png")


def ablation_plot(res, out_dir):
    names = sorted(res, key=lambda n: np.median(res[n]), reverse=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    _box(ax, [res[n] for n in names], names, horizontal=True)
    ax.set_xscale("log")
    ax.set_xlabel("Final penalty (log scale, lower is better)")
    ax.set_title(f"GA ablation: one change at a time, 10 seeds\n({SYN})", fontsize=10)
    ax.grid(alpha=0.3, axis="x", which="both")
    _save(fig, out_dir, "fig7_ablation.png")


def breakdown_plot(brk, out_dir):
    g = np.arange(len(brk))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    terms = [("Student clashes x1000", brk[:, 0] * core.W_CLASH, "crimson"),
             ("Room failures x1000", brk[:, 1] * core.W_ROOM, "black"),
             ("Consecutive x10", brk[:, 2] * core.W_CONSEC, "#ff7f0e"),
             ("Load x2", brk[:, 3] * core.W_LOAD, "#1f77b4")]
    for lab, y, c in terms:
        ax.plot(g, y, lw=2, color=c, label=lab)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Weighted penalty term of best-so-far (symlog)")
    ax.set_title(f"Which constraint is solved when? GA seed 7 ({SYN})", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    _save(fig, out_dir, "fig8_penalty_breakdown.png")


def diversity_plot(div, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for n, d in div.items():
        ax.plot(np.arange(len(d)), d, lw=2, label=n)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population diversity (mean pairwise Hamming distance)")
    ax.set_title(f"Population diversity over generations, seed 7 ({SYN})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig9_diversity.png")


def conflict_heatmap(data, chrom, out_dir):
    order = np.argsort(chrom, kind="stable")
    M = data["conflict"].copy()
    np.fill_diagonal(M, 0)
    M = M[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(M, cmap="Blues")
    slots = chrom[order]
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and slots[j + 1] == slots[i]:
            j += 1
        if j > i:
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), j - i + 1, j - i + 1, fill=False,
                                   ec="crimson", lw=2))
        i = j + 1
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    labs = [f"{core.exam_name(e)} (slot {int(chrom[e]) + 1})" for e in order]
    ax.set_xticklabels(labs, rotation=90, fontsize=7)
    ax.set_yticklabels(labs, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Students taking both exams")
    ax.set_title("Exam conflict matrix ordered by slot; red boxes = exams sharing a slot\n"
                 f"(every cell inside a red box is 0, so no clash) ({SYN})", fontsize=10)
    _save(fig, out_dir, "fig10_conflict_heatmap.png")


def student_load(data, chroms, titles, out_dir):
    cmap = ListedColormap(["#f0f0f0", "#9ecae1", "#fd8d3c", "#d7301f"])
    fig, axes = plt.subplots(1, len(chroms), figsize=(5 * len(chroms), 6), sharey=True)
    for ax, c, t in zip(np.atleast_1d(axes), chroms, titles):
        counts = data["enroll"] @ np.eye(N_DAYS)[c // N_PERIODS]
        counts = counts[np.lexsort((counts.sum(axis=1), counts.max(axis=1)))]
        im = ax.imshow(np.minimum(counts, 3), cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto",
                       interpolation="nearest")
        ax.set_xticks(range(N_DAYS))
        ax.set_xticklabels([f"Day {d + 1}" for d in range(N_DAYS)])
        ax.set_title(f"{t}\nstudent-days with 2+ exams: {int((counts >= 2).sum())}, "
                     f"with 3+: {int((counts >= 3).sum())}", fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("Students (sorted by load)")
    cb = fig.colorbar(im, ax=axes, ticks=[0, 1, 2, 3], fraction=0.03)
    cb.ax.set_yticklabels(["0", "1", "2", "3+"])
    cb.set_label("Exams on that day")
    fig.suptitle(f"Per-student daily exam load ({SYN})", fontsize=11)
    fig.savefig(os.path.join(out_dir, "fig11_student_load.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def room_utilisation(data, chrom, out_dir):
    total = float(sum(core.ROOM_CAPS))
    used = [data["sizes"][chrom == s].sum() / total * 100 for s in range(N_SLOTS)]
    n_ex = [int((chrom == s).sum()) for s in range(N_SLOTS)]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.bar(range(1, N_SLOTS + 1), used, color="#6baed6")
    for b, n in zip(bars, n_ex):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{n}", ha="center", fontsize=8)
    ax.set_xticks(range(1, N_SLOTS + 1))
    ax.set_xticklabels([f"D{(s // N_PERIODS) + 1}P{(s % N_PERIODS) + 1}" for s in range(N_SLOTS)],
                       rotation=60, fontsize=8)
    ax.set_ylabel(f"Seats occupied (% of {int(total)} total)")
    ax.set_title(f"Seat usage per slot (number above bar = exams in slot) ({SYN})", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, out_dir, "fig12_room_utilisation.png")


def sweeps_plot(mut, pop, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, sw, xl in [(axes[0], mut, "Per-gene mutation probability"),
                       (axes[1], pop, "Population size (150 generations)")]:
        xs = list(sw)
        m = np.array([np.mean(sw[x]) for x in xs])
        s = np.array([np.std(sw[x]) for x in xs])
        ax.errorbar(xs, m, yerr=s, marker="o", capsize=4, lw=2)
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("Final penalty, mean +/- std (log)")
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"Parameter sensitivity, 10 seeds per point ({SYN})", fontsize=11)
    _save(fig, out_dir, "fig13_param_sensitivity.png")
