"""Local swarm rule, no communication.

Each step the agent picks the free neighbour cell c with the best score

    w_gain * gain(c)              share of unsearched cells in the detection footprint around c
  - w_visit * v(c) / (1 + v(c))   own visit count
  - w_rep * sum_j max(0, 1 - d(c, j) / R)   repulsion from visible agents
  + w_pull' * cos(move, g)        pull towards unsearched prior mass
  + w_mom * [same move as before] + noise

g is the direction of sum_c p(c) exp(-d/lam) (c - x) / d^2 over the agent's unsearched cells
(a plain centroid gets stuck in already searched holes). w_pull' = stuck_boost * w_pull when no move
has any gain. Weights were picked on tuning seeds (scripts/tune.py).
"""
from __future__ import annotations

import numpy as np

from ..env import ACTIONS
from .base import Policy


class SwarmRule(Policy):
    name = "swarm"

    def __init__(self, w_gain=1.0, w_visit=0.3, w_rep=0.6, w_pull=0.05, w_mom=0.15, noise=0.05,
                 rep_radius=None, pull_every=4, lam=None, stuck_boost=4.0):
        self.w = dict(gain=w_gain, visit=w_visit, rep=w_rep, pull=w_pull, mom=w_mom, noise=noise)
        self.rep_radius_arg = rep_radius
        self.lam_arg = lam
        self.stuck_boost = stuck_boost
        self.pull_every = pull_every

    def reset(self, spec, seed=0):
        super().reset(spec, seed)
        n = spec["num_agents"]
        self.rd = spec["detect_radius"]
        self.rep_radius = self.rep_radius_arg or float(self.r)
        self.inner = spec["inner"]
        self.prior = spec["prior"]
        H, W = self.inner.shape
        self.gx, self.gy = np.mgrid[0:H, 0:W]
        self.last = [0] * n
        self.pull = [np.zeros(2) for _ in range(n)]
        self.fp = (2 * self.rd + 1) ** 2
        self.lam = self.lam_arg or 0.25 * (H - 2 * self.r)  # decay length = quarter of the map size

    def _pull_dir(self, o) -> np.ndarray:
        """Direction of sum_c p(c) exp(-d/lam) (c - x) / d^2 over unsearched cells."""
        w = self.prior * (~o.searched) * (~o.obst)
        dx, dy = self.gx - o.xy[0], self.gy - o.xy[1]
        d2 = dx * dx + dy * dy
        d2[o.xy[0], o.xy[1]] = 1
        k = w * np.exp(-np.sqrt(d2) / self.lam) / d2
        v = np.array([(k * dx).sum(), (k * dy).sum()])
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else np.zeros(2)

    def act_one(self, o) -> int:
        i, w, r, rd = o.agent_id, self.w, self.r, self.rd
        free = self.free_moves(o)
        if not free:
            return 0
        if o.t % self.pull_every == 0 or not self.pull[i].any():
            self.pull[i] = self._pull_dir(o)
        g = self.pull[i]
        # visible neighbours (relative coordinates), excluding self
        nb = np.argwhere(o.local_agents > 0) - r
        nb = nb[(nb != 0).any(axis=1)]
        x, y = o.xy
        rng = self.rngs[i]
        gains = []
        for a in free:
            dx, dy = ACTIONS[a]
            cx, cy = x + dx, y + dy
            fs = (slice(cx - rd, cx + rd + 1), slice(cy - rd, cy + rd + 1))
            gains.append(((~o.searched[fs]) & (~o.obst[fs]) & self.inner[fs]).sum() / self.fp)
        # stronger pull when there is nothing new around
        pull_w = w["pull"] * (self.stuck_boost if max(gains) == 0 else 1.0)
        best, best_s = free[0], -np.inf
        for a, gain in zip(free, gains):
            dx, dy = ACTIONS[a]
            cx, cy = x + dx, y + dy
            v = o.visits[cx, cy]
            s = w["gain"] * gain - w["visit"] * v / (1.0 + v)
            if len(nb):
                d = np.sqrt(((nb - np.array([dx, dy])) ** 2).sum(1))
                s -= w["rep"] * np.clip(1.0 - d / self.rep_radius, 0, None).sum()
            s += pull_w * (dx * g[0] + dy * g[1])
            s += w["mom"] * (a == self.last[i]) + w["noise"] * rng.random()
            if s > best_s:
                best, best_s = a, s
        self.last[i] = best
        return best
