"""Observation encoding for the IPPO policy (the same in training and evaluation).

local (7, 15, 15): crop of the agent's own memory around it: known obstacles, unknown, searched,
                   visible agents, visits (min(v, 4) / 4), prior, outside of the map
glob  (4, 8, 8):   whole map pooled to 8x8: unsearched, prior, obstacles, own position
scal  (2,):        t / max_steps, share of targets known to be found
mask  (5,):        valid actions
"""
from __future__ import annotations

import numpy as np

from ..env import ACTIONS

R = 7
C_L, C_G, G = 7, 4, 8
N_SCAL = 2


def crop(a: np.ndarray, x: int, y: int, r: int, fill) -> np.ndarray:
    H, W = a.shape
    out = np.full((2 * r + 1, 2 * r + 1), fill, dtype=np.float32)
    x0, x1, y0, y1 = x - r, x + r + 1, y - r, y + r + 1
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(H, x1), min(W, y1)
    out[sx0 - x0:sx1 - x0, sy0 - y0:sy1 - y0] = a[sx0:sx1, sy0:sy1]
    return out


class ObsEncoder:
    def __init__(self, spec: dict):
        self.spec = spec
        self.r = spec["obs_radius"]
        self.inner = spec["inner"]
        self.size = spec["H"] - 2 * self.r
        assert self.size % G == 0, "map size must be divisible by 8"
        self.b = self.size // G
        p = spec["prior"]
        self.prior_scaled = (p / p.max()).astype(np.float32)
        sl = slice(self.r, self.r + self.size)
        self.sl = sl
        pin = p[sl, sl]
        self.prior_blocks = pin.reshape(G, self.b, G, self.b).sum((1, 3))
        self.prior_blocks = (self.prior_blocks / self.prior_blocks.max()).astype(np.float32)
        self.outside = (~self.inner).astype(np.float32)
        self.T = spec["max_steps"]
        self.K = spec["num_targets"]

    def encode(self, o):
        x, y = o.xy
        r, sl, b = self.r, self.sl, self.b
        loc = np.empty((C_L, 2 * R + 1, 2 * R + 1), dtype=np.float32)
        loc[0] = crop(o.obst, x, y, R, 1)
        loc[1] = crop(~o.known, x, y, R, 1)
        loc[2] = crop(o.searched, x, y, R, 1)
        ag = np.zeros((2 * R + 1, 2 * R + 1), dtype=np.float32)
        ag[R - r:R + r + 1, R - r:R + r + 1] = o.local_agents
        ag[R, R] = 0.0
        loc[3] = ag
        loc[4] = np.minimum(crop(o.visits, x, y, R, 0), 4) / 4.0
        loc[5] = crop(self.prior_scaled, x, y, R, 0)
        loc[6] = crop(self.outside, x, y, R, 1)

        glob = np.empty((C_G, G, G), dtype=np.float32)
        glob[0] = 1.0 - o.searched[sl, sl].reshape(G, b, G, b).mean((1, 3))
        glob[1] = self.prior_blocks
        glob[2] = o.obst[sl, sl].reshape(G, b, G, b).mean((1, 3))
        glob[3] = 0.0
        glob[3, min(G - 1, (x - r) // b), min(G - 1, (y - r) // b)] = 1.0

        scal = np.array([o.t / self.T, o.found_known.mean() if self.K else 0.0], dtype=np.float32)

        mask = np.ones(5, dtype=bool)
        for a in range(1, 5):
            dx, dy = ACTIONS[a]
            if o.local_obstacles[r + dx, r + dy] or o.local_agents[r + dx, r + dy]:
                mask[a] = False
        return loc, glob, scal, mask

    def encode_batch(self, obs_list):
        enc = [self.encode(o) for o in obs_list]
        return (np.stack([e[0] for e in enc]), np.stack([e[1] for e in enc]),
                np.stack([e[2] for e in enc]), np.stack([e[3] for e in enc]))
