"""All GIFs used in the README.

    python scripts/make_all_gifs.py            # everything
    python scripts/make_all_gifs.py A cmp      # only some groups
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from make_gif import render_episode, save_gif  # noqa: E402
from pogema_search import SearchConfig  # noqa: E402

OUT = "figures/gifs"
IPPO = {"path": "runs/ippo_v1/model_final.pt"}

# A: one episode, every method
BASE_A = dict(size=32, density=0.2, num_agents=16, num_targets=3, fail_frac=0.25)
A = [("random_walk", {}, 0, 2), ("levy", {}, 0, 2), ("frontier", {}, 0, 2), ("frontier_comm", {}, 5, 1),
     ("frontier_coord", {}, "inf", 1), ("swarm", {}, 0, 1), ("ippo", IPPO, 0, 1)]
SEED_A = 4

SCENARIOS = {
    "B_prior": (dict(size=32, density=0.2, num_agents=12, num_targets=3, target_dist="prior"), 2,
                [("swarm", {}, 0), ("frontier", {}, 0), ("ippo", IPPO, 0)]),
    "C_maze": (dict(map_type="maze", size=32, density=0.2, num_agents=16, num_targets=3), 3,
               [("swarm", {}, 0), ("frontier_comm", {}, 5)]),
    "D_warehouse": (dict(map_type="warehouse", size=32, num_agents=16, num_targets=3), 1,
                    [("swarm", {}, 0), ("ippo", IPPO, 0)]),
    "E_64": (dict(size=64, density=0.1, num_agents=32, num_targets=3), 0,
             [("swarm", {}, 0), ("ippo", IPPO, 0), ("frontier_coord", {}, "inf")]),
    "F_fail50": (dict(size=32, density=0.2, num_agents=16, num_targets=3, fail_frac=0.5), 6,
                 [("swarm", {}, 0), ("frontier_comm", {}, 5)]),
}


def label(policy, rc):
    if rc in (0, "0"):
        return policy
    return f"{policy} r_c={rc}" if rc != "inf" else f"{policy} global"


def group_a(keep=None):
    res = {}
    for pol, kw, rc, every in A:
        cfg = SearchConfig(**BASE_A, comm_radius=rc)
        frames, m = render_episode(cfg, pol, kw, SEED_A, every=every, title=label(pol, rc))
        save_gif(frames, f"{OUT}/A_{pol}.gif")
        print(f"A_{pol}: t_last={m['t_last']:.0f} success={m['success']:.0f}", flush=True)
        if keep and pol in keep:
            res[pol] = frames
    return res


def scenario(name):
    base, seed, methods = SCENARIOS[name]
    for pol, kw, rc in methods:
        cfg = SearchConfig(**base, comm_radius=rc)
        every = 2 if base["size"] >= 64 else 1
        frames, m = render_episode(cfg, pol, kw, seed, every=every, title=label(pol, rc),
                                   cell_px=8 if base["size"] >= 64 else 14)
        save_gif(frames, f"{OUT}/{name}_{pol}.gif")
        print(f"{name}_{pol}: t_last={m['t_last']:.0f} success={m['success']:.0f}", flush=True)


def comparison(methods=(("swarm", {}, 0), ("ippo", IPPO, 0), ("frontier", {}, 0), ("frontier_comm", {}, 5)),
               scale=0.45, every=2, max_steps_shown=120):
    """2x2 grid: four methods on the same episode, frames in sync; finished panels hold their last frame."""
    runs = []
    for pol, kw, rc in methods:
        cfg = SearchConfig(**BASE_A, comm_radius=rc)
        frames, _ = render_episode(cfg, pol, kw, SEED_A, every=every, title=label(pol, rc))
        runs.append(frames)
    T = max(len(f) for f in runs)
    T = min(T, max_steps_shown // every)  # slow methods are cut off
    tiles = []
    for t in range(T):
        ims = [Image.fromarray(r[min(t, len(r) - 1)]) for r in runs]
        w, h = ims[0].size
        ims = [im.resize((int(w * scale), int(h * scale)), Image.LANCZOS) for im in ims]
        w, h = ims[0].size
        canvas = Image.new("RGB", (2 * w, 2 * h), "white")
        for k, im in enumerate(ims):
            canvas.paste(im, ((k % 2) * w, (k // 2) * h))
        tiles.append(np.asarray(canvas))
    save_gif(tiles, f"{OUT}/compare_4_methods.gif", fps=6, hold_s=2)
    print("compare_4_methods:", T, "frames", flush=True)


if __name__ == "__main__":
    groups = sys.argv[1:] or ["A", *SCENARIOS, "cmp"]
    for g in groups:
        if g == "A":
            group_a()
        elif g == "cmp":
            comparison()
        else:
            scenario(g)
