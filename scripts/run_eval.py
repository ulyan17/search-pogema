"""Run an experiment spec; resumable (finished episodes are skipped).

    python scripts/run_eval.py configs/stage2_small.yaml --workers 2 [--budget-min 9]
"""
import argparse
import os
import sys

import yaml

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pogema_search.experiment import run_spec  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("spec")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--output", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--budget-min", type=float, default=None, help="stop after this many minutes")
    p.add_argument("--max-seed", type=int, default=None, help="only seeds < max-seed (seed-major passes)")
    a = p.parse_args()
    spec = yaml.safe_load(open(a.spec))
    if a.max_seed is not None:
        st = spec["seeds"]
        spec["seeds"] = {"start": st["start"], "count": min(st["count"], a.max_seed - st["start"])}
    pending = run_spec(spec, output=a.output, workers=a.workers, limit=a.limit,
                       time_budget_s=None if a.budget_min is None else 60 * a.budget_min)
    sys.exit(0 if pending == 0 else 3)


if __name__ == "__main__":
    main()
