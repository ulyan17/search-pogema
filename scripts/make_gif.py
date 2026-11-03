"""Render one search episode to a GIF.

    python scripts/make_gif.py --policy swarm --size 32 --agents 16 --targets 3 --fail 0.25 --seed 4 \
        --out figures/gifs/swarm.gif
"""
import argparse
import json
import os
import sys

import imageio.v2 as imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pogema_search import SearchConfig, run_episode  # noqa: E402
from pogema_search.policies import make_policy  # noqa: E402
from pogema_search.render import Renderer  # noqa: E402


def render_episode(cfg, policy, kwargs=None, seed=0, every=1, cell_px=14, title=None):
    frames, holder = [], {}

    def cb(env):
        if "r" not in holder:
            holder["r"] = Renderer(env, title=title or policy, cell_px=cell_px)
        holder["r"].update(env)
        if env.t % every == 0 or env.done:
            frames.append(holder["r"].frame(env))

    m, env = run_episode(cfg, make_policy(policy, **(kwargs or {})), seed=seed, frame_cb=cb)
    holder["r"].close()
    return frames, m


def save_gif(frames, path, fps=10, hold_s=1.5):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frames = frames + [frames[-1]] * int(fps * hold_s)
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="swarm")
    p.add_argument("--kwargs", default="{}", help="policy kwargs as JSON")
    p.add_argument("--map", default="random")
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--density", type=float, default=0.2)
    p.add_argument("--agents", type=int, default=16)
    p.add_argument("--targets", type=int, default=3)
    p.add_argument("--target-dist", default="uniform")
    p.add_argument("--comm", default="0")
    p.add_argument("--fail", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--every", type=int, default=1, help="keep every n-th frame")
    p.add_argument("--cell-px", type=int, default=14)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", default="figures/gifs/demo.gif")
    a = p.parse_args()
    cfg = SearchConfig(map_type=a.map, size=a.size, density=a.density, num_agents=a.agents,
                       num_targets=a.targets, target_dist=a.target_dist, comm_radius=a.comm,
                       fail_frac=a.fail, max_steps=a.max_steps, seed=a.seed)
    frames, m = render_episode(cfg, a.policy, json.loads(a.kwargs), a.seed, a.every, a.cell_px)
    save_gif(frames, a.out, a.fps)
    print(f"{a.out}: {len(frames)} frames, " + ", ".join(f"{k}={v:.3g}" for k, v in m.items()))


if __name__ == "__main__":
    main()
