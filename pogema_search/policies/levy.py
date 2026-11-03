"""Lattice Levy walk: random direction, flight length P(L) ~ L^-mu, new flight when blocked."""
from __future__ import annotations

import numpy as np

from ..env import ACTIONS
from .base import Policy


class LevyWalk(Policy):
    name = "levy"

    def __init__(self, mu: float = 2.0, l_min: int = 1, l_max: int | None = None):
        assert mu > 1
        self.mu, self.l_min, self.l_max_arg = mu, l_min, l_max

    def reset(self, spec, seed=0):
        super().reset(spec, seed)
        n = spec["num_agents"]
        self.l_max = self.l_max_arg or (spec["H"] - 2 * self.r)  # map size
        self.dir = [0] * n
        self.left = [0] * n

    def _flight_length(self, rng) -> int:
        # inverse-CDF sampling of a truncated power law
        a, b, e = self.l_min, self.l_max, 1.0 - self.mu
        u = rng.random()
        return int(max(a, np.floor((a ** e + u * (b ** e - a ** e)) ** (1.0 / e))))

    def act_one(self, o) -> int:
        i, rng = o.agent_id, self.rngs[o.agent_id]
        free = self.free_moves(o)
        if not free:
            self.left[i] = 0
            return 0
        if self.left[i] <= 0 or self.dir[i] not in free:
            self.dir[i] = int(rng.integers(1, 5))
            self.left[i] = self._flight_length(rng)
            if self.dir[i] not in free:  # new direction blocked: turn to a free one, keep the length
                self.dir[i] = int(rng.choice(free))
        self.left[i] -= 1
        return self.dir[i]


__all__ = ["LevyWalk", "ACTIONS"]
