"""Summaries (markdown tables with bootstrap 95% CI) and figures from an experiment CSV.

    python scripts/make_plots.py results/stage2_small.csv --tag stage2
"""
import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pogema_search.metrics import bootstrap_ci, robustness_ci  # noqa: E402

# fixed method -> colour (validated categorical order, light surface); never re-assigned by rank
METHOD_ORDER = ["swarm", "frontier", "frontier_comm r_c=2", "frontier_comm r_c=5", "frontier_comm global",
                "levy", "random_walk", "ippo"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*"]
COLOR = dict(zip(METHOD_ORDER, PALETTE))
MARK = dict(zip(METHOD_ORDER, MARKERS))
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e3"


def method_name(row):
    rc = float(row["comm_radius"])
    if row["label"] == "frontier_comm" or (row["label"] not in ("frontier",) and rc > 0):
        return f"{row['label']} global" if math.isinf(rc) else f"{row['label']} r_c={rc:g}"
    return row["label"]


def load(path):
    df = pd.concat([pd.read_csv(p) for p in path.split(",")], ignore_index=True)
    df["comm_radius"] = df["comm_radius"].replace({"inf": np.inf}).astype(float)
    df["method"] = df.apply(method_name, axis=1)
    df["t_last_norm"] = df["t_last"] / df["max_steps"]
    return df


def ordered(methods, figure=False):
    known = [m for m in METHOD_ORDER if m in set(methods)]
    return known if figure else known + sorted(set(methods) - set(METHOD_ORDER))


def fmt(m, lo, hi, d=2):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def summary_table(df, metrics, by, out_md, title):
    lines = [f"## {title}", "", "Mean [95% bootstrap CI] over evaluation seeds. "
             "t_* are censored at max_steps; t_last_norm = t_last / max_steps.", ""]
    for keys, g in df.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        lines.append("**" + ", ".join(f"{k}={v}" for k, v in zip(by, keys)) + f"** (n={g.groupby('method').size().iloc[0]} per method)")
        lines.append("")
        lines.append("| method | " + " | ".join(metrics) + " |")
        lines.append("|---|" + "---|" * len(metrics))
        for m in ordered(g["method"].unique()):
            gm = g[g["method"] == m]
            cells = []
            for k in metrics:
                d = 0 if k in ("t_first", "t_last", "messages") else 2
                cells.append(fmt(*bootstrap_ci(gm[k].values), d=d))
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    with open(out_md, "a") as f:
        f.write("\n".join(lines) + "\n")


def robustness_table(df, by, out_md, metrics=("success", "t_last_norm", "coverage_auc")):
    lines = ["## Robustness: metric(fail_frac=p) / metric(fail_frac=0)", "",
             "Same seeds (same maps/targets) with and without failures; CI by independent bootstrap. "
             "For success and coverage >1 is better; for t_last_norm <1 is better.", ""]
    for p in sorted(df["fail_frac"].unique()):
        if p == 0:
            continue
        lines += [f"**p = {p}**", "", "| " + " | ".join(by) + " | method | " + " | ".join(metrics) + " |",
                  "|" + "---|" * (len(by) + 1 + len(metrics))]
        for keys, g in df.groupby(by):
            keys = keys if isinstance(keys, tuple) else (keys,)
            for m in ordered(g["method"].unique()):
                a = g[(g["method"] == m) & (g["fail_frac"] == p)]
                b = g[(g["method"] == m) & (g["fail_frac"] == 0)]
                cells = [fmt(*robustness_ci(a[k].values, b[k].values)) for k in metrics]
                lines.append("| " + " | ".join(map(str, keys)) + f" | {m} | " + " | ".join(cells) + " |")
        lines.append("")
    with open(out_md, "a") as f:
        f.write("\n".join(lines) + "\n")


def style(ax):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def dot_ci_figure(df, metric, xlabel, out, rows="target_dist", cols=("size", "num_agents"), fail=0.0):
    d = df[df["fail_frac"] == fail]
    rv = sorted(d[rows].unique())
    cv = sorted({tuple(x) for x in d[list(cols)].values})
    methods = ordered(d["method"].unique(), figure=True)
    fig, axes = plt.subplots(len(rv), len(cv), figsize=(2.7 * len(cv) + 1.6, 0.36 * len(methods) * len(rv) + 1.2),
                             sharex=True, squeeze=False)
    for i, r in enumerate(rv):
        for j, c in enumerate(cv):
            ax = axes[i, j]
            g = d[(d[rows] == r) & (d[cols[0]] == c[0]) & (d[cols[1]] == c[1])]
            for y, m in enumerate(methods):
                mean, lo, hi = bootstrap_ci(g[g["method"] == m][metric].values)
                ax.plot([lo, hi], [y, y], color=COLOR.get(m, MUTED), lw=2, solid_capstyle="round")
                ax.scatter([mean], [y], s=42, marker=MARK.get(m, "o"), color=COLOR.get(m, MUTED),
                           edgecolors="white", linewidths=1.2, zorder=3)
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels(methods if j == 0 else [""] * len(methods), fontsize=8, color=INK)
            ax.set_ylim(len(methods) - 0.5, -0.5)
            style(ax)
            if i == 0:
                ax.set_title(f"{c[0]}x{c[0]} map, {c[1]} agents", fontsize=9, color=INK)
            if j == len(cv) - 1:
                ax.text(1.02, 0.5, f"targets: {r}", transform=ax.transAxes, rotation=270, va="center",
                        fontsize=9, color=MUTED)
            if i == len(rv) - 1:
                ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    fig.suptitle(f"{xlabel} (mean, 95% bootstrap CI){'' if fail == 0 else f', {int(fail*100)}% agents fail'}",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def coverage_figure(df, out, where):
    d = df.copy()
    for k, v in where.items():
        d = d[d[k] == v]
    methods = ordered(d["method"].unique(), figure=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in methods:
        curves = np.array([json.loads(s) for s in d[d["method"] == m]["coverage_curve"]])
        x = np.linspace(0, 1, curves.shape[1])
        boots = np.array([curves[np.random.default_rng(b).integers(0, len(curves), len(curves))].mean(0)
                          for b in range(500)])
        lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
        mean = curves.mean(0)
        ax.fill_between(x, lo, hi, color=COLOR.get(m, MUTED), alpha=0.12, lw=0)
        ax.plot(x, mean, color=COLOR.get(m, MUTED), lw=2, marker=MARK.get(m, "o"), markersize=4, markevery=4,
                label=m)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1.02)
    ax.set_xlabel("time / max_steps", fontsize=8.5, color=MUTED)
    ax.set_ylabel("searched fraction of reachable area", fontsize=8.5, color=MUTED)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ttl = ", ".join(f"{k}={v}" for k, v in where.items())
    ax.set_title(f"Searched area over time: {ttl}", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="one CSV or several comma-separated")
    p.add_argument("--tag", default="results")
    a = p.parse_args()
    df = load(a.csv)
    os.makedirs("figures", exist_ok=True)
    md = f"results/summary_{a.tag}.md"
    open(md, "w").write(f"# Summary: {a.csv}\n\nEpisodes: {len(df)}. Per-policy wall time per step "
                        f"(ms, mean over episodes):\n\n" +
                        (df.assign(ms=1000 * df.wall_s / df.steps.clip(lower=1)).groupby("method")["ms"].mean()
                         .reindex(ordered(df.method.unique())).round(2).to_frame().to_markdown()) + "\n\n")
    by = ["size", "num_agents", "target_dist", "fail_frac"]
    summary_table(df, ["success", "t_first", "t_last", "coverage_auc", "revisit_ratio", "messages"], by, md,
                  "Main metrics")
    robustness_table(df, ["size", "num_agents", "target_dist"], md)
    stop = df["stop_when_all_found"].astype(bool)
    if stop.all():  # search runs: detection-time and success figures
        dot_ci_figure(df, "t_last_norm", "time to last detection / max_steps", f"figures/{a.tag}_t_last.png")
        dot_ci_figure(df, "t_last_norm", "time to last detection / max_steps",
                      f"figures/{a.tag}_t_last_fail.png", fail=0.5)
        dot_ci_figure(df, "success", "success rate", f"figures/{a.tag}_success_fail.png", fail=0.5)
    elif (~stop).all():  # full-horizon runs: coverage curves are meaningful
        for (s, n, p), _ in df.groupby(["size", "num_agents", "fail_frac"]):
            coverage_figure(df, f"figures/{a.tag}_coverage_s{s}_n{n}_p{p}.png",
                            {"size": s, "num_agents": n, "fail_frac": p})
    print("wrote", md, "and figures/", a.tag + "_*.png")


if __name__ == "__main__":
    main()
