"""Policy interface.

act() gets one AgentObs per agent and returns one action per agent. Agent i may only use obs[i] and its
own state inside the policy; act() is batched just for speed.
"""
from __future__ import annotations

import numpy as np

from ..env import ACTIONS


class Policy:
    name = "base"
    uses_comm = False

    def reset(self, spec: dict, seed: int = 0):
        self.spec = spec
        self.r = spec["obs_radius"]
        self.rngs = [np.random.default_rng([seed, i]) for i in range(spec["num_agents"])]

    def act(self, obs_list) -> list[int]:
        return [self.act_one(o) if o.active else 0 for o in obs_list]

    def act_one(self, o) -> int:
        raise NotImplementedError

    def free_moves(self, o) -> list[int]:
        """Move actions (1..4) into cells that are free of obstacles and agents right now."""
        r = self.r
        res = []
        for a in range(1, 5):
            dx, dy = ACTIONS[a]
            if o.local_obstacles[r + dx, r + dy] == 0 and o.local_agents[r + dx, r + dy] == 0:
                res.append(a)
        return res


class RandomWalk(Policy):
    """Uniformly random move among currently free neighbouring cells (stay if boxed in)."""
    name = "random_walk"

    def act_one(self, o) -> int:
        moves = self.free_moves(o)
        if not moves:
            return 0
        return int(self.rngs[o.agent_id].choice(moves))
