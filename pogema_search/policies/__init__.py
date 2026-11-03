from .base import Policy, RandomWalk
from .frontier import Frontier, FrontierComm, FrontierCoord
from .ippo import IPPOPolicy
from .levy import LevyWalk
from .swarm import SwarmRule

POLICIES = {
    "random_walk": RandomWalk,
    "levy": LevyWalk,
    "frontier": Frontier,
    "frontier_comm": FrontierComm,
    "frontier_coord": FrontierCoord,
    "swarm": SwarmRule,
    "ippo": IPPOPolicy,
}


def make_policy(name: str, **kwargs) -> Policy:
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; available: {sorted(POLICIES)}")
    return POLICIES[name](**kwargs)
