"""Lower bounds on t_last for 64x64 episodes of the full grid.

travel: the last target has to be reached, max_k min_i dist(start_i, detection zone of target k).
area: each agent adds at most (2 r_d + 1) searched cells per step, so roughly
      K / (K + 1) * (reachable area - initially searched) / ((2 r_d + 1) N) steps.
"""
import os
import sys
from collections import deque

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pogema_search import SearchConfig, SearchPogema  # noqa: E402


def multi_bfs(free, sources):
    dist = np.full(free.shape, -1, dtype=np.int32)
    q = deque()
    for s in sources:
        dist[s] = 0
        q.append(s)
    while q:
        x, y = q.popleft()
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if free[nx, ny] and dist[nx, ny] < 0:
                dist[nx, ny] = dist[x, y] + 1
                q.append((nx, ny))
    return dist


def bounds(row):
    cfg = SearchConfig(size=int(row["size"]), density=float(row["density"]), num_agents=int(row["num_agents"]),
                       num_targets=int(row["num_targets"]), map_type="random", target_dist="uniform")
    env = SearchPogema(cfg)
    env.reset(seed=int(row["seed"]))
    free = env.reachable
    dist = multi_bfs(free, [tuple(p) for p in env.positions()])
    rd = cfg.detect_radius
    lb_t = 0
    for tx, ty in env.targets:
        zone = dist[tx - rd:tx + rd + 1, ty - rd:ty + rd + 1]
        lb_t = max(lb_t, int(zone[zone >= 0].min()))
    a0 = env.searched_global[free].sum()
    k = cfg.num_targets
    lb_a = k / (k + 1) * (free.sum() - a0) / ((2 * rd + 1) * cfg.num_agents)
    return lb_t, lb_a


def main():
    df = pd.read_csv("results/full_grid.csv")
    df = df[(df.fail_frac == 0) & (df["size"] == 64)]
    keys = ["size", "density", "num_agents", "num_targets", "seed"]
    ep = df[keys].drop_duplicates()
    lbs = [bounds(r) for _, r in ep.iterrows()]
    ep["lb_travel"] = [a for a, b in lbs]
    ep["lb_area"] = [b for a, b in lbs]
    ep["lb"] = ep[["lb_travel", "lb_area"]].max(1)
    ep.to_csv("results/lower_bounds_64.csv", index=False)
    print(ep.groupby("num_agents")[["lb_travel", "lb_area", "lb"]].mean().round(1))


if __name__ == "__main__":
    main()
