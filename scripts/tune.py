"""Hyper-parameter selection for the swarm rule and the Lévy exponent on TUNING seeds (10000+),
disjoint from evaluation seeds (0..). Objective: mean t_last / max_steps (censored), lower is better,
averaged over a mixed set of task settings. Output: results/tuning.csv and results/tuning_best.yaml.

    python scripts/tune.py [--budget-min 9]
"""
import argparse
import itertools
import os
import sys

import pandas as pd
import yaml

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pogema_search.experiment import run_spec  # noqa: E402

OUT = "results/tuning.csv"


def spec():
    methods = []
    for wp, wr, wv, sb in itertools.product([0.05, 0.1, 0.25], [0.3, 0.6, 1.2], [0.1, 0.3], [2.0, 4.0, 8.0]):
        kw = dict(w_pull=wp, w_rep=wr, w_visit=wv, stuck_boost=sb)
        methods.append({"label": "swarm", "policy": "swarm", "kwargs": kw})
    for mu in [1.5, 2.0, 2.5, 3.0]:
        methods.append({"label": "levy", "policy": "levy", "kwargs": {"mu": mu}})
    return {
        "name": "tuning",
        "seeds": {"start": 10000, "count": 4},
        "base": {"size": 32, "num_targets": 3},
        "grid": {"density": [0.1, 0.3], "num_agents": [4, 16], "target_dist": ["uniform", "prior"]},
        "methods": methods,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget-min", type=float, default=None)
    a = p.parse_args()
    pending = run_spec(spec(), output=OUT, workers=2,
                       time_budget_s=None if a.budget_min is None else 60 * a.budget_min)
    if pending:
        sys.exit(3)
    df = pd.read_csv(OUT)
    df["obj"] = df["t_last"] / df["max_steps"]
    best = {}
    for label, g in df.groupby("label"):
        s = g.groupby("kwargs").agg(obj=("obj", "mean"), success=("success", "mean"), n=("obj", "size"))
        s = s.sort_values("obj")
        print(f"\n== {label}: top 5 of {len(s)} ==\n{s.head(5).to_string()}")
        best[label] = yaml.safe_load(s.index[0])
    with open("results/tuning_best.yaml", "w") as f:
        yaml.safe_dump(best, f)
    print("\nbest:", best)


if __name__ == "__main__":
    main()
