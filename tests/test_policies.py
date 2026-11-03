import math

import numpy as np
import pytest

from pogema_search import SearchConfig, run_episode
from pogema_search.maps import MAP_TYPES
from pogema_search.policies import POLICIES, make_policy
from pogema_search.policies.levy import LevyWalk


@pytest.mark.parametrize("name", sorted(POLICIES))
@pytest.mark.parametrize("mt", MAP_TYPES)
def test_policies_run_everywhere(name, mt):
    cfg = SearchConfig(map_type=mt, size=16, num_agents=6, num_targets=2, max_steps=60,
                       comm_radius=5 if name == "frontier_comm" else 0, fail_frac=0.25)
    m, env = run_episode(cfg, make_policy(name), seed=1)
    assert 0 <= m["coverage_final"] <= 1 and env.t <= 60


def test_frontier_single_agent_covers_empty_map():
    cfg = SearchConfig(size=16, density=0.0, num_agents=1, num_targets=1, max_steps=200,
                       stop_when_all_found=False)
    m, env = run_episode(cfg, make_policy("frontier"), seed=0)
    assert m["coverage_final"] == 1.0


def test_comm_helps_frontier_on_average():
    base = dict(size=24, density=0.1, num_agents=8, num_targets=3)
    t0 = np.mean([run_episode(SearchConfig(**base, comm_radius=0), make_policy("frontier"), s)[0]["t_last"]
                  for s in range(6)])
    t1 = np.mean([run_episode(SearchConfig(**base, comm_radius=math.inf), make_policy("frontier_comm"), s)[0]
                  ["t_last"] for s in range(6)])
    assert t1 < t0


def test_levy_lengths_in_range():
    lw = LevyWalk(mu=2.0, l_min=1, l_max=32)
    lw.l_max = 32
    rng = np.random.default_rng(0)
    L = np.array([lw._flight_length(rng) for _ in range(5000)])
    assert L.min() >= 1 and L.max() <= 32
    assert (L == 1).mean() > 0.3  # heavy mass at short flights for mu = 2


def test_inbox_delivered_only_within_comm_radius():
    from pogema_search import SearchPogema
    for rc, expect in ((0, 0), (math.inf, 5)):
        env = SearchPogema(SearchConfig(size=16, num_agents=6, num_targets=1, comm_radius=rc, start="uniform"))
        env.reset(seed=0)
        obs, _, _ = env.step([0] * 6, messages={0: (5, 5)})
        got = sum(any(j == 0 for j, _ in o.inbox) for o in obs)
        assert got == expect


def test_coordinated_frontier_spreads_goals():
    cfg = SearchConfig(size=32, density=0.1, num_agents=16, num_targets=3, comm_radius=math.inf, max_steps=40,
                       stop_when_all_found=False)
    from pogema_search import SearchPogema
    res = {}
    for name in ("frontier_comm", "frontier_coord"):
        pol = make_policy(name)
        env = SearchPogema(cfg)
        obs = env.reset(seed=1)
        pol.reset(env.spec(), seed=1)
        dup = []
        for _ in range(40):
            acts = pol.act(obs)
            goals = [g for g in pol.goal if g is not None]
            dup.append(len(goals) - len(set(goals)))
            obs, done, _ = env.step(acts, messages=getattr(pol, "outbox", None))
        res[name] = np.mean(dup)
    assert res["frontier_coord"] < res["frontier_comm"]
