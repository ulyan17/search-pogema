"""Learning curve of IPPO: training-episode success (rolling) and checkpoint evaluation on tuning seeds.

    python scripts/plot_learning_curve.py --run runs/ippo_v1 --check results/ippo_check.csv
"""
import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, RED, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#e34948", "#222222", "#6b6b6b", "#e6e6e3"


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/ippo_v1")
    p.add_argument("--check", default="results/ippo_check.csv")
    p.add_argument("--out", default="figures/stage3_learning_curve.png")
    a = p.parse_args()
    lc = pd.read_csv(os.path.join(a.run, "learning_curve.csv")).drop_duplicates("it", keep="last")
    lc = lc.sort_values("it")
    w = 20
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    x = lc["samples"] / 1e6
    a1.plot(x, lc["success"].rolling(w, min_periods=1).mean(), color=RED, lw=2)
    ph = lc[lc["phase"].diff() > 0]
    for _, r in ph.iterrows():
        a1.axvline(r["samples"] / 1e6, color=MUTED, lw=1, ls="--")
        a1.text(r["samples"] / 1e6, 0.05, "  16x16 -> 32x32", fontsize=7.5, color=MUTED)
    a1.set_ylim(0, 1.02)
    a1.set_xlabel("agent-steps (millions)", fontsize=8.5, color=MUTED)
    a1.set_title(f"IPPO: success on training episodes (rolling mean of {w} iterations)", fontsize=9,
                 color=INK, loc="left")
    style(a1)

    df = pd.read_csv(a.check)
    df["tl"] = df["t_last"] / df["max_steps"]
    pts = []
    for lab, g in df.groupby("label"):
        m = re.match(r"ippo_it(\d+)_sample", lab)
        if m:
            it = int(m.group(1))
            s = lc.loc[lc["it"] <= it, "samples"].max() / 1e6
            vals = g["tl"].values
            boots = [vals[np.random.default_rng(b).integers(0, len(vals), len(vals))].mean() for b in range(1000)]
            pts.append((s, vals.mean(), *np.quantile(boots, [0.025, 0.975])))
    pts.sort()
    pts = np.array(pts)
    a2.plot(pts[:, 0], pts[:, 1], color=RED, lw=2, marker="o", markersize=7, markeredgecolor="white")
    a2.fill_between(pts[:, 0], pts[:, 2], pts[:, 3], color=RED, alpha=0.12, lw=0)
    for lab, col in (("swarm", BLUE), ("frontier", ORANGE)):
        v = df[df["label"] == lab]["tl"].mean()
        a2.axhline(v, color=col, lw=1.5, ls="--")
        a2.text(pts[-1, 0], v, f" {lab} (no comm)", color=INK, fontsize=7.5, va="bottom", ha="right")
    a2.set_ylim(0, None)
    a2.set_xlabel("agent-steps (millions)", fontsize=8.5, color=MUTED)
    a2.set_ylabel("time to last detection / max_steps", fontsize=8.5, color=MUTED)
    a2.set_title("Checkpoint evaluation on tuning seeds (lower is better)", fontsize=9, color=INK, loc="left")
    style(a2)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print("saved", a.out)


if __name__ == "__main__":
    main()
