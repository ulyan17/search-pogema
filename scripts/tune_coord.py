"""Select lam_frac / pen_radius / yield_on_conflict of the coordinated frontier on TUNING seeds (10000+).

    python scripts/tune_coord.py
"""
import itertools
import os
import sys

import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pogema_search.experiment import run_spec  # noqa: E402

OUT = "results/tuning_coord.csv"
methods = [{"label": "frontier_comm", "policy": "frontier_comm"}]
for lam, R, y in itertools.product([0.05, 0.1, 0.25], [3, 5, 9], [True, False]):
    methods.append({"label": "frontier_coord", "policy": "frontier_coord",
                    "kwargs": {"lam_frac": lam, "pen_radius": R, "yield_on_conflict": y}})
spec = {"name": "tuning_coord", "seeds": {"start": 10000, "count": 4},
        "base": {"num_targets": 3, "density": 0.2, "comm_radius": 5},
        "grid": {"size": [32, 64], "num_agents": [8, 32]}, "methods": methods}
if run_spec(spec, output=OUT, workers=2, time_budget_s=540):
    sys.exit(3)
df = pd.read_csv(OUT)
df["obj"] = df.t_last / df.max_steps
s = df.groupby(["label", "kwargs"])["obj"].mean().sort_values()
print(s.head(8).to_string())
print("frontier_comm reference:", round(df[df.label == "frontier_comm"]["obj"].mean(), 4))
