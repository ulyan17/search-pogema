"""Analysis of the full grid: leaderboard, scaling, density, robustness, communication trade-off, win map.

    python scripts/analyze_full.py results/full_grid.csv --tag full
Outputs results/analysis_<tag>.md and figures/<tag>_*.png. All numbers are computed from the CSV.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from make_plots import COLOR, GRID, INK, MARK, MUTED, load, ordered  # noqa: E402
from pogema_search.metrics import bootstrap_ci, robustness_ci  # noqa: E402

NO_COMM = ["swarm", "ippo", "frontier", "levy", "random_walk"]
EXTRA = {"ippo global": "#8c1d1d", "frontier_coord r_c=2": "#3d4f5c", "frontier_coord r_c=5": "#3d4f5c",
         "frontier_coord global": "#3d4f5c"}
MARK_EXTRA = {"ippo global": "h", "frontier_coord r_c=2": "<", "frontier_coord r_c=5": ">",
              "frontier_coord global": "p"}


def col(m):
    return COLOR.get(m, EXTRA.get(m, MUTED))


def style(ax, grid_axis="both"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def ci_str(x, d=3):
    m, lo, hi = bootstrap_ci(x)
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def leaderboard(df, md):
    rows = []
    for m in ordered(df.method.unique()):
        g = df[df.method == m]
        rows.append(f"| {m} | {len(g)} | {ci_str(g.success.values, 3)} | {ci_str(g.t_last_norm.values)} | "
                    f"{ci_str((g.t_first / g.max_steps).values)} | {ci_str(g.coverage_auc.values)} | "
                    f"{ci_str(g.revisit_ratio.values)} | {g.messages.mean():.0f} |")
    md.append("## Leaderboard (pooled over all configurations)\n\nMean [95% bootstrap CI] over episodes; every "
              "method is evaluated on the same seeds (maps, starts, targets, failures).\n")
    md.append("| method | episodes | success | t_last / T | t_first / T | coverage AUC | revisit | messages |")
    md.append("|---|---|---|---|---|---|---|---|")
    md += rows
    md.append("")


def line_by(df, x, out, title, metric="t_last_norm", ylabel="time to last detection / max_steps",
            facet="size", where=None, methods=None):
    d = df if where is None else df.query(where)
    fv = sorted(d[facet].unique())
    methods = [m for m in (methods or ordered(d.method.unique(), figure=True)) if m in set(d.method)]
    fig, axes = plt.subplots(1, len(fv), figsize=(3.6 * len(fv) + 1.8, 3.4), sharey=True, squeeze=False)
    for j, f in enumerate(fv):
        ax = axes[0, j]
        g = d[d[facet] == f]
        xs = sorted(g[x].unique())
        for m in methods:
            gm = g[g.method == m]
            if gm.empty:
                continue
            st = [bootstrap_ci(gm[gm[x] == v][metric].values) for v in xs]
            mu = [s[0] for s in st]
            ax.fill_between(range(len(xs)), [s[1] for s in st], [s[2] for s in st], color=col(m), alpha=0.12, lw=0)
            ax.plot(range(len(xs)), mu, color=col(m), lw=2, marker=MARK.get(m, MARK_EXTRA.get(m, "o")), markersize=5,
                    markeredgecolor="white", label=m)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(v) for v in xs], fontsize=8)
        ax.set_xlabel(x.replace("_", " "), fontsize=8.5, color=MUTED)
        ax.set_title(f"{facet} = {f}", fontsize=9, color=INK)
        style(ax)
        if j == 0:
            ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)
    axes[0, -1].legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def robustness(df, md):
    md.append("## Robustness to agent loss\n\nratio = metric(p) / metric(p = 0) on the same seeds, pooled over "
              "configurations, independent-bootstrap CI. success: higher is better; t_last: lower is better.\n")
    md.append("| method | success p=0 | success p=0.25 | success p=0.5 | t_last ratio p=0.25 | t_last ratio p=0.5 |")
    md.append("|---|---|---|---|---|---|")
    for m in ordered(df.method.unique()):
        g = df[df.method == m]
        s = [g[g.fail_frac == p].success.mean() for p in (0, 0.25, 0.5)]
        r = [robustness_ci(g[g.fail_frac == p].t_last_norm.values, g[g.fail_frac == 0].t_last_norm.values)
             for p in (0.25, 0.5)]
        md.append(f"| {m} | {s[0]:.3f} | {s[1]:.3f} | {s[2]:.3f} | "
                  + " | ".join(f"{x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]" for x in r) + " |")
    md.append("")


def comm_tradeoff(df, out, md):
    d = df[df.fail_frac == 0]
    methods = [m for m in ordered(d.method.unique()) if m != "levy" and m != "random_walk"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    md.append("## Communication trade-off (p = 0, pooled)\n")
    md.append("| method | messages / episode | cells transmitted / episode | t_last / T |")
    md.append("|---|---|---|---|")
    for m in methods:
        g = d[d.method == m]
        msg = g.messages.mean()
        t = bootstrap_ci(g.t_last_norm.values)
        xpos = {"swarm": 0.7, "ippo": 1.4, "frontier": 1.0}.get(m, max(msg, 1)) if msg == 0 else msg
        ax.errorbar(xpos, t[0], yerr=[[t[0] - t[1]], [t[2] - t[0]]], fmt=MARK.get(m, MARK_EXTRA.get(m, "h")), color=col(m),
                    markersize=8, markeredgecolor="white", capsize=0, lw=2)
        left = m == "swarm" or m.startswith("frontier_comm")
        ax.annotate(m, (xpos, t[0]), xytext=(-8, 3) if left else (8, -3), textcoords="offset points",
                    fontsize=7.5, color=INK, ha="right" if left else "left", va="center")
        md.append(f"| {m} | {msg:.0f} | {g.cells_tx.mean():.0f} | {t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}] |")
    md.append("")
    ax.set_xscale("log")
    ax.set_xlabel("messages per episode (log; methods without communication drawn near 1)", fontsize=8.5,
                  color=MUTED)
    ax.set_ylabel("time to last detection / max_steps", fontsize=8.5, color=MUTED)
    ax.set_title("Search time vs communication volume (no failures)", fontsize=9.5, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def paired_diff_ci(g, a, b, n_boot=2000, seed=0):
    """Mean and 95% CI of t_last_norm(a) - t_last_norm(b) over paired episodes (same seed and config)."""
    keys = ["density", "num_targets", "seed"]
    ja = g[g.method == a].set_index(keys).t_last_norm
    jb = g[g.method == b].set_index(keys).t_last_norm
    d = (ja - jb).dropna().values
    rng = np.random.default_rng(seed)
    bm = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return d.mean(), *np.quantile(bm, [0.025, 0.975])


def win_map(df, out, md, methods=("swarm", "ippo", "frontier")):
    """Fastest method without communication per (size, agents). A tie unless the paired CI excludes 0."""
    d = df[df.method.isin(methods) & (df.fail_frac == 0)]
    sizes, ns = sorted(d["size"].unique()), sorted(d.num_agents.unique())
    best = np.empty((len(sizes), len(ns)), dtype=object)
    val = np.zeros((len(sizes), len(ns)))
    md.append("## Best communication-free method per (map size, agents), p = 0\n")
    md.append("Cell: method with the lowest mean t_last/T; the winner is declared only if the paired 95% CI of "
              "its difference to the runner-up excludes 0 (else 'tie'). Also: paired mean difference "
              "swarm - ippo [95% CI] (negative = swarm faster).\n")
    md.append("| size \\ agents | " + " | ".join(map(str, ns)) + " |")
    md.append("|---|" + "---|" * len(ns))
    for i, s in enumerate(sizes):
        cells = []
        for j, n in enumerate(ns):
            g = d[(d["size"] == s) & (d.num_agents == n)]
            means = g.groupby("method").t_last_norm.mean().sort_values()
            b, second = means.index[0], means.index[1]
            dm, lo, hi = paired_diff_ci(g, b, second)
            sig = hi < 0
            best[i, j] = b if sig else f"tie:{b}~{second}"
            val[i, j] = means[b]
            sd = paired_diff_ci(g, "swarm", "ippo")
            cells.append(f"{b if sig else 'tie ' + b + '~' + second} ({means[b]:.2f}); "
                         f"swarm-ippo {sd[0]:+.3f} [{sd[1]:+.3f}, {sd[2]:+.3f}]")
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    md.append("")
    fig, ax = plt.subplots(figsize=(1.25 * len(ns) + 1.5, 0.9 * len(sizes) + 1.0))
    for i in range(len(sizes)):
        for j in range(len(ns)):
            b = best[i, j]
            tie = b.startswith("tie:")
            label = b[4:].replace("~", " ≈ ") if tie else b
            ax.add_patch(plt.Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, color="#b9b9b4" if tie else col(b),
                                       alpha=0.85, lw=0))
            ax.text(j, i, f"{label}\n{val[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color=INK if tie else "white")
    ax.set_xlim(-0.5, len(ns) - 0.5), ax.set_ylim(len(sizes) - 0.5, -0.5)
    ax.set_xticks(range(len(ns))), ax.set_xticklabels(ns, fontsize=8)
    ax.set_yticks(range(len(sizes))), ax.set_yticklabels([f"{s}x{s}" for s in sizes], fontsize=8)
    ax.set_xlabel("agents", fontsize=8.5, color=MUTED)
    ax.set_title("Fastest method without communication (mean t_last / T; grey = no significant winner)",
                 fontsize=9.5, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--tag", default="full")
    p.add_argument("--where", default=None, help="pandas query to filter rows, e.g. \"map_type == 'maze'\"")
    a = p.parse_args()
    df = load(a.csv)
    if a.where:
        df = df.query(a.where)
    seeds = df.groupby("method").seed.nunique().min()
    md = [f"# Full-grid analysis ({a.csv})", "", f"{len(df)} episodes, {df.method.nunique()} methods, "
          f"seeds per cell: {seeds}. t_* censored at max_steps (T = 8*size).", ""]
    leaderboard(df, md)
    robustness(df, md)
    comm_tradeoff(df, f"figures/{a.tag}_comm_tradeoff.png", md)
    win_map(df, f"figures/{a.tag}_winmap.png", md)
    nc = ["swarm", "ippo", "frontier", "frontier_comm r_c=5", "frontier_coord r_c=5", "levy", "random_walk"]
    line_by(df, "num_agents", f"figures/{a.tag}_scaling_agents.png",
            "Scaling with the number of agents (p = 0, all densities and target counts)", where="fail_frac == 0",
            methods=nc)
    if df.density.nunique() > 1:
        line_by(df, "density", f"figures/{a.tag}_density.png", "Effect of obstacle density (p = 0)",
                where="fail_frac == 0", methods=nc)
    line_by(df, "fail_frac", f"figures/{a.tag}_failures.png", "Effect of agent loss", metric="success",
            ylabel="success rate", methods=nc)
    open(f"results/analysis_{a.tag}.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md[:30]))


if __name__ == "__main__":
    main()
