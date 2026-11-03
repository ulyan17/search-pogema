"""Multi-agent search for hidden targets on top of POGEMA.

POGEMA gives us the grid, maps, movement with collisions and the egocentric view. On top of that:
hidden targets, detection radius, per-agent memory, map exchange within comm_radius and agent failures.

Notes on POGEMA 1.3.1:
* on_target='finish' removes an agent at its own POGEMA goal, so we use 'nothing' (PogemaCoopFinish)
  with dummy goals and ignore POGEMA rewards/termination.
* a failed agent gets grid.is_active = False and keeps blocking its cell (hide_agent() would free it).
* reset(seed=...) does not regenerate the map, so a new GridConfig is built every episode.
* coordinates are POGEMA's padded ones (map shifted by obs_radius); agents know their own position.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

import numpy as np

from pogema import GridConfig
from pogema.envs import PogemaCoopFinish

from .maps import bfs_distances, largest_free_component, make_obstacles

ACTIONS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))  # == GridConfig.MOVES: stay, up, down, left, right


@dataclass
class SearchConfig:
    map_type: str = "random"           # random | maze | warehouse
    size: int = 16
    density: float = 0.2
    num_agents: int = 8
    num_targets: int = 1
    target_dist: str = "uniform"       # uniform | prior
    prior_sigma_frac: float = 0.15     # sigma of the Gaussian prior as a fraction of map size
    start: str = "cluster"             # cluster (deployment from one base) | uniform
    obs_radius: int = 3
    detect_radius: int = 2
    comm_radius: float = 0.0           # 0 = no communication, math.inf = global
    fail_frac: float = 0.0             # fraction of agents that fail during the episode
    fail_window: tuple = (0.0, 0.5)    # failure time ~ U[fail_window] * T_scale
    fail_time_scale: str = "size"      # T_scale: size (same for all methods) or max_steps
    failed_blocks: bool = True         # failed agent keeps occupying its cell
    max_steps: Optional[int] = None    # default: 8 * size
    stop_when_all_found: bool = True
    seed: int = 0

    def __post_init__(self):
        if self.max_steps is None:
            self.max_steps = 8 * self.size
        if isinstance(self.comm_radius, str):
            self.comm_radius = math.inf if self.comm_radius.lower() in ("inf", "global") else float(self.comm_radius)
        self.fail_window = tuple(self.fail_window)
        assert 1 <= self.detect_radius <= self.obs_radius, "need 1 <= detect_radius <= obs_radius"
        assert self.target_dist in ("uniform", "prior")
        assert self.start in ("cluster", "uniform")
        assert 0.0 <= self.fail_frac <= 1.0
        assert self.fail_time_scale in ("size", "max_steps")

    @classmethod
    def from_dict(cls, d: dict) -> "SearchConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise KeyError(f"unknown SearchConfig keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentObs:
    """What agent i can see and remember. Arrays are views of the agent's memory, do not modify."""
    agent_id: int
    active: bool
    t: int
    xy: tuple                    # global (padded) position
    local_obstacles: np.ndarray  # (2r+1, 2r+1) POGEMA obstacle window
    local_agents: np.ndarray     # (2r+1, 2r+1) POGEMA agents window (includes self at the centre)
    known: np.ndarray            # (H, W) bool, cells the agent has seen (own + received)
    obst: np.ndarray             # (H, W) bool, known obstacles
    searched: np.ndarray         # (H, W) bool, cells inside some detection footprint
    visits: np.ndarray           # (H, W) int32 visit counts (own; max-merged with neighbours)
    prior: np.ndarray            # (H, W) float, prior over target location (0 outside the map)
    found_known: np.ndarray      # (K,) bool, targets the agent knows are found
    found_xy: list               # positions of the targets the agent knows are found
    inbox: list = field(default_factory=list)  # [(sender, payload)] policy messages received this step


class SearchPogema:
    def __init__(self, config: SearchConfig):
        self.cfg = config
        self.base: Optional[PogemaCoopFinish] = None

    def _build_layout(self, rng: np.random.Generator):
        c = self.cfg
        for attempt in range(50):
            obst = make_obstacles(c.map_type, c.size, c.density, int(rng.integers(2**31 - 1)))
            comp = largest_free_component(obst)
            if comp.sum() >= max(c.num_agents * 2 + c.num_targets, 0.25 * c.size * c.size * (1 - c.density)):
                break
        else:
            raise RuntimeError("could not generate a map with a large enough free component")
        cells = np.argwhere(comp)
        if c.start == "uniform":
            idx = rng.choice(len(cells), size=c.num_agents, replace=False)
            starts = [tuple(map(int, cells[i])) for i in idx]
        else:  # cluster: the N cells closest (BFS) to a random base cell
            base = tuple(map(int, cells[rng.integers(len(cells))]))
            dist = bfs_distances(comp, base)
            d = dist[comp].astype(float) + rng.random(len(cells)) * 0.5  # random tie-break
            order = np.argsort(d)
            starts = [tuple(map(int, cells[i])) for i in order[:c.num_agents]]

        # prior over target location, defined on the whole inner map (agents do not know obstacles)
        xs, ys = np.mgrid[0:c.size, 0:c.size]
        if c.target_dist == "uniform":
            prior = np.ones((c.size, c.size))
            prior_center = None
        else:
            prior_center = tuple(map(int, cells[rng.integers(len(cells))]))
            s = c.prior_sigma_frac * c.size
            prior = np.exp(-((xs - prior_center[0]) ** 2 + (ys - prior_center[1]) ** 2) / (2 * s * s))
        prior = prior / prior.sum()

        # targets: in the agents' component, not detectable from the start positions
        st = np.array(starts)
        cheb = np.max(np.abs(cells[:, None, :] - st[None, :, :]), axis=2).min(axis=1)
        cand = cells[cheb > c.detect_radius]
        if len(cand) < c.num_targets:
            start_set = set(starts)
            cand = np.array([p for p in cells if tuple(p) not in start_set])
        w = prior[cand[:, 0], cand[:, 1]]
        w = w / w.sum()
        tidx = rng.choice(len(cand), size=c.num_targets, replace=False, p=w)
        targets = [tuple(map(int, cand[i])) for i in tidx]
        return obst, comp, starts, targets, prior, prior_center

    def reset(self, seed: Optional[int] = None):
        c = self.cfg
        self.seed = c.seed if seed is None else seed
        rng = np.random.default_rng(self.seed)
        obst, comp, starts, targets, prior, prior_center = self._build_layout(rng)
        r = c.obs_radius
        self.r = r

        gc = GridConfig(map=obst.tolist(), size=c.size, agents_xy=[list(s) for s in starts],
                        targets_xy=[list(s) for s in starts], obs_radius=r, on_target="nothing",
                        collision_system="priority", max_episode_steps=c.max_steps + 1, seed=self.seed)
        self.base = PogemaCoopFinish(gc)
        self.base.reset()
        self.grid = self.base.grid
        H, W = self.grid.obstacles.shape
        self.H, self.W = H, W
        assert H == c.size + 2 * r

        N, K = c.num_agents, c.num_targets
        pad = lambda a, v=0: np.pad(a, r, constant_values=v)  # noqa: E731
        self.inner = pad(np.ones((c.size, c.size), dtype=bool), False)
        self.reachable = pad(comp, False)          # free cells of the agents' component
        self.prior = pad(prior.astype(np.float64), 0.0)
        self.prior_center = None if prior_center is None else (prior_center[0] + r, prior_center[1] + r)
        self.targets = [(x + r, y + r) for x, y in targets]
        self.target_arr = np.array(self.targets)

        self.known = np.zeros((N, H, W), dtype=bool)
        self.obst = np.zeros((N, H, W), dtype=bool)
        self.searched = np.zeros((N, H, W), dtype=bool)
        self.visits = np.zeros((N, H, W), dtype=np.int32)
        self.found_known = np.zeros((N, K), dtype=bool)

        self.searched_global = np.zeros((H, W), dtype=bool)
        self.searched_t = np.full((H, W), -1, dtype=np.int32)   # step at which a cell was first searched
        self.visited_global = np.zeros((H, W), dtype=bool)
        self.found_t = np.full(K, -1, dtype=np.int64)
        self.found_by = np.full(K, -1, dtype=np.int64)

        self.active = np.ones(N, dtype=bool)
        n_fail = int(round(c.fail_frac * N))
        self.fail_t = np.full(N, -1, dtype=np.int64)
        if n_fail > 0:
            scale = c.size if c.fail_time_scale == "size" else c.max_steps
            lo = max(1, int(c.fail_window[0] * scale))
            hi = max(lo + 1, int(c.fail_window[1] * scale) + 1)
            who = rng.choice(N, size=n_fail, replace=False)
            self.fail_t[who] = rng.integers(lo, hi, size=n_fail)

        self.t = 0
        self.total_visits = 0
        self.area = c.size * c.size
        self.new_mass = np.zeros(N)       # per-agent prior-weighted newly searched cells this step (RL reward)
        self.step_detect = np.zeros(N)    # per-agent number of targets detected this step (RL reward)
        self.hist = {"coverage": [], "messages": [], "cells_tx": [], "n_active": [], "n_found": []}
        self.last_links: list = []
        self._sense()
        self._detect()
        self._communicate()
        self._record()
        return self.observations()

    def step(self, actions, messages: Optional[dict] = None):
        """messages: optional {agent_id: payload}, delivered to neighbours within comm_radius (inbox)."""
        c = self.cfg
        self.outbox = messages or {}
        assert not self.done, "episode is over, call reset()"
        self.t += 1
        # failures happen at the beginning of step t: the agent does not move at t
        for i in np.flatnonzero(self.fail_t == self.t):
            self.active[i] = False
            if c.failed_blocks:
                self.grid.is_active[i] = False
            else:
                self.grid.hide_agent(int(i))
        acts = [int(a) if self.active[i] else 0 for i, a in enumerate(actions)]
        self.base.move_agents(acts)
        self._sense()
        self._detect()
        self._communicate()
        self._record()
        return self.observations(), self.done, self.info()

    @property
    def all_found(self) -> bool:
        return bool((self.found_t >= 0).all())

    @property
    def done(self) -> bool:
        return (self.cfg.stop_when_all_found and self.all_found) or self.t >= self.cfg.max_steps

    def positions(self) -> np.ndarray:
        return np.array(self.grid.positions_xy, dtype=np.int64)

    def _sense(self):
        r, rd = self.r, self.cfg.detect_radius
        go = self.grid.obstacles.astype(bool)
        self.new_mass[:] = 0.0
        self.step_detect[:] = 0.0
        for i in np.flatnonzero(self.active):
            x, y = self.grid.positions_xy[i]
            self.known[i, x - r:x + r + 1, y - r:y + r + 1] = True
            self.obst[i, x - r:x + r + 1, y - r:y + r + 1] = go[x - r:x + r + 1, y - r:y + r + 1]
            self.searched[i, x - rd:x + rd + 1, y - rd:y + rd + 1] = True
            fs = (slice(x - rd, x + rd + 1), slice(y - rd, y + rd + 1))
            new = ~self.searched_global[fs] & self.reachable[fs]
            # uniform prior -> 1 per new reachable cell; first agent in the loop gets the credit
            self.new_mass[i] = float((self.prior[fs] * new).sum() * self.area)
            self.searched_global[x - rd:x + rd + 1, y - rd:y + rd + 1] = True
            st = self.searched_t[x - rd:x + rd + 1, y - rd:y + rd + 1]
            st[st < 0] = self.t
            self.visits[i, x, y] += 1
            self.visited_global[x, y] = True
            self.total_visits += 1

    def _detect(self):
        pos = self.positions()[self.active]
        if len(pos) == 0:
            return
        ids = np.flatnonzero(self.active)
        for k in np.flatnonzero(self.found_t < 0):
            d = np.max(np.abs(pos - self.target_arr[k]), axis=1)
            hit = np.flatnonzero(d <= self.cfg.detect_radius)
            if len(hit):
                self.found_t[k] = self.t
                self.found_by[k] = ids[hit[0]]
                self.found_known[ids[hit], k] = True
                self.step_detect[ids[hit]] += 1.0

    def _deliver(self, ids, adj):
        ob = getattr(self, "outbox", {})
        if not ob:
            return
        for a, i in enumerate(ids):
            for b in np.flatnonzero(adj[a]):
                j = int(ids[b])
                if j in ob:
                    self.inbox[i].append((j, ob[j]))

    def _communicate(self):
        rc = self.cfg.comm_radius
        self.last_links = []
        self._step_msgs, self._step_cells = 0, 0
        self.inbox = [[] for _ in range(self.cfg.num_agents)]
        if rc <= 0:
            return
        ids = np.flatnonzero(self.active)
        if len(ids) < 2:
            return
        pos = self.positions()[ids].astype(float)
        if math.isinf(rc):
            adj = np.ones((len(ids), len(ids)), dtype=bool)
        else:
            d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
            adj = d2 <= rc * rc
        np.fill_diagonal(adj, False)
        if not adj.any():
            return
        self._deliver(ids, adj)
        if math.isinf(rc):
            self._communicate_global(ids)
            return
        known, obst, searched = self.known[ids], self.obst[ids], self.searched[ids]
        visits, fk = self.visits[ids], self.found_known[ids]
        msgs = cells = 0
        for a in range(len(ids)):
            nb = np.flatnonzero(adj[a])
            if len(nb) == 0:
                continue
            i = ids[a]
            msgs += len(nb)
            # information volume: cells (seen or searched) the receiver did not have yet
            cells += int((known[nb] & ~known[a]).sum() + (searched[nb] & ~searched[a]).sum())
            self.known[i] |= known[nb].any(0)
            self.obst[i] |= obst[nb].any(0)
            self.searched[i] |= searched[nb].any(0)
            np.maximum(self.visits[i], visits[nb].max(0), out=self.visits[i])
            self.found_known[i] |= fk[nb].any(0)
        self._step_msgs, self._step_cells = msgs, cells
        iu = np.argwhere(np.triu(adj))
        self.last_links = [(int(ids[a]), int(ids[b])) for a, b in iu]

    def _communicate_global(self, ids):
        """Same as the local exchange with everyone connected, but O(N*H*W) instead of O(N^2*H*W)."""
        n = len(ids)
        known, searched = self.known[ids], self.searched[ids]
        kc, sc = known.sum(0), searched.sum(0)
        cells = int(((~known) * kc[None]).sum() + ((~searched) * sc[None]).sum())
        k_all, s_all, o_all = known.any(0), searched.any(0), self.obst[ids].any(0)
        v_all = self.visits[ids].max(0)
        f_all = self.found_known[ids].any(0)
        for i in ids:
            self.known[i] = k_all
            self.searched[i] = s_all
            self.obst[i] = o_all
            self.visits[i] = v_all
            self.found_known[i] = f_all
        self._step_msgs, self._step_cells = n * (n - 1), cells
        self.last_links = []  # not drawn for global communication

    def _record(self):
        h = self.hist
        cov = (self.searched_global & self.reachable).sum() / self.reachable.sum()
        h["coverage"].append(float(cov))
        h["messages"].append(getattr(self, "_step_msgs", 0))
        h["cells_tx"].append(getattr(self, "_step_cells", 0))
        h["n_active"].append(int(self.active.sum()))
        h["n_found"].append(int((self.found_t >= 0).sum()))

    def observations(self) -> list[AgentObs]:
        obs = []
        for i in range(self.cfg.num_agents):
            found_xy = [self.targets[k] for k in np.flatnonzero(self.found_known[i])]
            obs.append(AgentObs(
                agent_id=i, active=bool(self.active[i]), t=self.t, xy=tuple(self.grid.positions_xy[i]),
                local_obstacles=self.grid.get_obstacles_for_agent(i),
                local_agents=self.grid.get_positions(i),
                known=self.known[i], obst=self.obst[i], searched=self.searched[i], visits=self.visits[i],
                prior=self.prior, found_known=self.found_known[i], found_xy=found_xy,
                inbox=self.inbox[i] if hasattr(self, "inbox") else []))
        return obs

    def info(self) -> dict:
        return {"t": self.t, "n_found": int((self.found_t >= 0).sum()), "n_active": int(self.active.sum()),
                "coverage": self.hist["coverage"][-1]}

    def spec(self) -> dict:
        """Static information every agent knows in advance."""
        c = self.cfg
        return {"num_agents": c.num_agents, "num_targets": c.num_targets, "obs_radius": c.obs_radius,
                "detect_radius": c.detect_radius, "comm_radius": c.comm_radius, "H": self.H, "W": self.W,
                "inner": self.inner, "prior": self.prior, "max_steps": c.max_steps}
