"""Measure environment throughput (random-walk policy, fixed number of steps)."""
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pogema_search import SearchConfig, SearchPogema  # noqa: E402
from pogema_search.policies import RandomWalk  # noqa: E402


def bench(size, n, comm, steps=200, reps=2):
    cfg = SearchConfig(size=size, density=0.2, num_agents=n, num_targets=3, comm_radius=comm,
                       max_steps=steps, stop_when_all_found=False, start="uniform")
    tot = 0.0
    for s in range(reps):
        env = SearchPogema(cfg)
        pol = RandomWalk()
        t0 = time.perf_counter()
        obs = env.reset(seed=s)
        pol.reset(env.spec(), seed=s)
        while not env.done:
            obs, _, _ = env.step(pol.act(obs))
        tot += time.perf_counter() - t0
    return tot / (reps * steps)


if __name__ == "__main__":
    print("size agents comm  ms/step")
    for size in (16, 32, 64):
        for n in (4, 16, 64):
            for comm in (0, 5, math.inf):
                ms = 1000 * bench(size, n, comm)
                print(f"{size:4d} {n:6d} {str(comm):>4} {ms:8.2f}", flush=True)
