"""Experiment specs (yaml) -> list of episodes -> CSV.

One row per (method, task config, seed). Rows are appended as soon as an episode finishes, finished
keys are skipped on restart, so a long run can be split into several shorter ones. All methods get the
same seeds, i.e. the same maps, starts, targets and failure times.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
import time
from multiprocessing import get_context

import numpy as np

from .env import SearchConfig
from .metrics import coverage_curve, episode_metrics

CURVE_POINTS = 21
METRIC_KEYS = ["success", "frac_found", "t_first", "t_last", "coverage_final", "coverage_auc",
               "revisit_ratio", "messages", "cells_tx", "steps", "n_failed"]


def _as_list(v):
    return v if isinstance(v, list) else [v]


def _norm(v):
    if isinstance(v, str) and v.lower() in ("inf", "global"):
        return math.inf
    return v


def expand(spec: dict) -> list[dict]:
    base = dict(spec.get("base", {}))
    grid = spec.get("grid", {})
    seeds = spec.get("seeds", {"start": 0, "count": 10})
    seed_list = list(range(seeds["start"], seeds["start"] + seeds["count"])) if isinstance(seeds, dict) else seeds
    jobs = []
    for m in spec["methods"]:
        g = dict(grid)
        g.update(m.get("task", {}))
        keys = sorted(g)
        for combo in itertools.product(*[_as_list(g[k]) for k in keys]):
            task = dict(base)
            task.update({k: _norm(v) for k, v in zip(keys, combo)})
            for s in seed_list:
                jobs.append({"label": m["label"], "policy": m.get("policy", m["label"]),
                             "kwargs": m.get("kwargs", {}), "task": task, "seed": s})
    return jobs


def job_key(job) -> str:
    t = {k: (str(v) if isinstance(v, float) and math.isinf(v) else v) for k, v in sorted(job["task"].items())}
    return json.dumps([job["label"], job["kwargs"], t, job["seed"]], sort_keys=True, default=str)


def run_job(job) -> dict:
    try:
        return _run_job(job)
    except Exception as e:  # one broken episode must not kill a multi-hour run
        return {"key": job_key(job), "error": repr(e)}


def _run_job(job) -> dict:
    from .policies import make_policy  # local import: workers
    from .runner import run_episode
    t0 = time.perf_counter()
    cfg = SearchConfig.from_dict(job["task"])
    m, env = run_episode(cfg, make_policy(job["policy"], **job["kwargs"]), seed=job["seed"])
    curve = coverage_curve(env)
    idx = np.linspace(0, len(curve) - 1, CURVE_POINTS).round().astype(int)
    row = {"key": job_key(job), "label": job["label"], "policy": job["policy"],
           "kwargs": json.dumps(job["kwargs"], sort_keys=True), "seed": job["seed"]}
    row.update({k: v for k, v in cfg.to_dict().items() if k != "seed"})
    row["fail_window"] = json.dumps(list(cfg.fail_window))
    row.update({k: m[k] for k in METRIC_KEYS})
    row["coverage_curve"] = json.dumps([round(float(curve[i]), 4) for i in idx])
    row["wall_s"] = round(time.perf_counter() - t0, 4)
    return row


def done_keys(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {r["key"] for r in csv.DictReader(f)}


def run_spec(spec: dict, output: str | None = None, workers: int = 2, limit: int | None = None,
             time_budget_s: float | None = None, verbose: bool = True) -> int:
    """Run all pending jobs of a spec. Returns the number of jobs still pending afterwards."""
    output = output or spec["output"]
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    jobs = expand(spec)
    done = done_keys(output)
    todo = [j for j in jobs if job_key(j) not in done]
    # seed-major order: an interrupted run still covers every (method, config) with the first k seeds
    todo.sort(key=lambda j: j["seed"])
    if limit:
        todo = todo[:limit]
    if verbose:
        print(f"[{spec.get('name', '?')}] total {len(jobs)}, done {len(done)}, running {len(todo)}", flush=True)
    if not todo:
        return 0
    t_start = time.time()
    new_file = not os.path.exists(output) or os.path.getsize(output) == 0
    n_done = 0
    ctx = get_context("fork")
    with open(output, "a", newline="") as f, ctx.Pool(workers) as pool:
        writer = None
        for row in pool.imap_unordered(run_job, todo, chunksize=1):
            if "error" in row:
                with open(output + ".errors.log", "a") as ef:
                    ef.write(json.dumps(row) + "\n")
                continue
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new_file:
                    writer.writeheader()
            writer.writerow(row)
            f.flush()
            n_done += 1
            if verbose and (n_done % 50 == 0 or n_done == len(todo)):
                el = time.time() - t_start
                print(f"  {n_done}/{len(todo)}  {el:.0f}s  ({el / n_done:.2f} s/episode)", flush=True)
            if time_budget_s and time.time() - t_start > time_budget_s:
                pool.terminate()
                break
    done_now = done_keys(output)
    pending = sum(job_key(j) not in done_now for j in jobs)
    if verbose:
        print(f"[{spec.get('name', '?')}] pending after this run: {pending}", flush=True)
    return pending
