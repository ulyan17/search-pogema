"""Run one episode of a policy and return its metrics."""
from __future__ import annotations

from .env import SearchConfig, SearchPogema
from .metrics import episode_metrics


def run_episode(cfg: SearchConfig, policy, seed: int, frame_cb=None):
    env = SearchPogema(cfg)
    obs = env.reset(seed=seed)
    policy.reset(env.spec(), seed=seed)
    if frame_cb is not None:
        frame_cb(env)
    while not env.done:
        acts = policy.act(obs)
        obs, done, info = env.step(acts, messages=getattr(policy, "outbox", None))
        if frame_cb is not None:
            frame_cb(env)
    return episode_metrics(env), env
