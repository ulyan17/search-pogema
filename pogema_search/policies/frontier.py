"""Frontier exploration (Yamauchi 1997, 1998) adapted to search.

Frontier cell = known free cell that is not searched yet or borders unknown space. Each agent goes to
the nearest one (BFS over its own known map) and replans when the goal is searched, a new obstacle
blocks the path or another agent blocks it for a few steps. With comm_radius > 0 the maps are merged
by the environment. FrontierCoord adds goal sharing (see below).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..env import ACTIONS
from .base import Policy

DIRS = ((-1, 0, 1), (1, 0, 2), (0, -1, 3), (0, 1, 4))  # dx, dy, action
MOVE_TO_ACTION = {(dx, dy): a for dx, dy, a in DIRS}


class Frontier(Policy):
    name = "frontier"

    def __init__(self, patience: int = 2, max_expand: int | None = None):
        self.patience = patience
        self.max_expand = max_expand

    def reset(self, spec, seed=0):
        super().reset(spec, seed)
        n = spec["num_agents"]
        self.inner = spec["inner"]
        self.path = [[] for _ in range(n)]   # list of cells to visit (next first)
        self.goal = [None] * n
        self.blocked = [0] * n
        self.n_plans = 0

    def _is_candidate(self, o, cell) -> bool:
        x, y = cell
        if not self.inner[x, y] or o.obst[x, y] or not o.known[x, y]:
            return False
        if not o.searched[x, y]:
            return True
        return not (o.known[x - 1, y] and o.known[x + 1, y] and o.known[x, y - 1] and o.known[x, y + 1])

    def _masks(self, o, avoid_agents: bool):
        known, obst, searched, inner = o.known, o.obst, o.searched, self.inner
        passable = known & ~obst
        if avoid_agents:
            r = self.r
            x, y = o.xy
            win = o.local_agents.astype(bool).copy()
            win[r, r] = False
            passable = passable.copy()
            passable[x - r:x + r + 1, y - r:y + r + 1] &= ~win
        unknown = ~known
        adj_unknown = np.zeros_like(known)
        adj_unknown[1:, :] |= unknown[:-1, :]
        adj_unknown[:-1, :] |= unknown[1:, :]
        adj_unknown[:, 1:] |= unknown[:, :-1]
        adj_unknown[:, :-1] |= unknown[:, 1:]
        cand = passable & inner & (~searched | adj_unknown)
        return passable, cand

    def _plan(self, o, avoid_agents: bool):
        """BFS from the agent to the nearest candidate cell; returns a path (excluding start)."""
        self.n_plans += 1
        passable, cand = self._masks(o, avoid_agents)
        known = o.known
        start = tuple(o.xy)
        H, W = known.shape
        parent = {start: None}
        q = deque([start])
        expanded = 0
        while q:
            c = q.popleft()
            if c != start and cand[c]:
                path = []
                while c != start:
                    path.append(c)
                    c = parent[c]
                return path[::-1]
            expanded += 1
            if self.max_expand and expanded > self.max_expand:
                break
            cx, cy = c
            for dx, dy, _ in DIRS:
                nb = (cx + dx, cy + dy)
                if nb not in parent and passable[nb]:
                    parent[nb] = c
                    q.append(nb)
        return []

    def act_one(self, o) -> int:
        i = o.agent_id
        path, goal = self.path[i], self.goal[i]
        if path and path[0] == tuple(o.xy):  # last move succeeded
            path = self.path[i] = path[1:]
        x, y = o.xy
        need = (goal is None or not path or not self._is_candidate(o, goal) or o.obst[path[0]]
                or self.blocked[i] >= self.patience
                or abs(path[0][0] - x) + abs(path[0][1] - y) != 1)
        if need:
            avoid = self.blocked[i] >= self.patience
            path = self._plan(o, avoid_agents=avoid)
            self.path[i] = path
            self.goal[i] = path[-1] if path else None
            self.blocked[i] = 0
        if not path:  # nothing left to explore that I know of: random free move
            free = self.free_moves(o)
            return int(self.rngs[i].choice(free)) if free else 0
        nx, ny = path[0]
        a = MOVE_TO_ACTION[(nx - x, ny - y)]
        r = self.r
        if o.local_agents[r + nx - x, r + ny - y]:  # another agent in the way: wait
            self.blocked[i] += 1
            return 0
        self.blocked[i] = 0
        return a


class FrontierComm(Frontier):
    """Same planner; the name only marks that it is meant to run with comm_radius > 0."""
    name = "frontier_comm"
    uses_comm = True


class FrontierCoord(Frontier):
    """Frontier + goal allocation, a decentralized version of Burgard et al. (2005).

    Agents also send their current goal to neighbours. Agent i minimises
    d(c) + lam * sum_g max(0, 1 - cheb(c, g) / R) over frontier cells, g = goals of neighbours with a
    smaller id. BFS stops as soon as the distance alone is worse than the best score.
    """
    name = "frontier_coord"
    uses_comm = True

    # defaults selected on tuning seeds (scripts/tune_coord.py, results/tuning_coord.csv)
    def __init__(self, lam_frac: float = 0.1, pen_radius: int | None = 9, patience: int = 2,
                 yield_on_conflict: bool = False):
        super().__init__(patience=patience)
        self.lam_frac, self.pen_radius_arg = lam_frac, pen_radius
        self.yield_on_conflict = yield_on_conflict

    def reset(self, spec, seed=0):
        super().reset(spec, seed)
        size = spec["H"] - 2 * self.r
        self.lam = self.lam_frac * size
        self.R = self.pen_radius_arg or (2 * spec["detect_radius"] + 1)
        self.outbox = {}
        self._claimed = []

    def act(self, obs_list):
        self.outbox = {}
        return super().act(obs_list)

    def _pen(self, c):
        p = 0.0
        for g in self._claimed:
            d = max(abs(c[0] - g[0]), abs(c[1] - g[1]))
            if d < self.R:
                p += 1.0 - d / self.R
        return p

    def act_one(self, o) -> int:
        i = o.agent_id
        self._claimed = [tuple(p) for j, p in o.inbox if j < i and p is not None]
        g = self.goal[i]
        if self.yield_on_conflict and g is not None and self._claimed and self._pen(g) > 0:
            self.goal[i] = None  # a higher-priority neighbour took (nearly) my frontier
        a = super().act_one(o)
        self.outbox[i] = self.goal[i]
        return a

    def _plan(self, o, avoid_agents: bool):
        if not self._claimed:
            return super()._plan(o, avoid_agents)
        self.n_plans += 1
        passable, cand = self._masks(o, avoid_agents)
        start = tuple(o.xy)
        parent = {start: None}
        q = deque([(start, 0)])
        best, best_s = None, np.inf
        while q:
            c, d = q.popleft()
            if d >= best_s:  # every remaining cell has score >= its distance >= best
                break
            if c != start and cand[c]:
                s = d + self.lam * self._pen(c)
                if s < best_s:
                    best, best_s = c, s
            cx, cy = c
            for dx, dy, _ in DIRS:
                nb = (cx + dx, cy + dy)
                if nb not in parent and passable[nb]:
                    parent[nb] = c
                    q.append((nb, d + 1))
        if best is None:
            return []
        path, c = [], best
        while c != start:
            path.append(c)
            c = parent[c]
        return path[::-1]


__all__ = ["Frontier", "FrontierComm", "FrontierCoord", "ACTIONS"]
