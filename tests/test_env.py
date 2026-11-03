import math

import numpy as np
import pytest

from pogema_search import SearchConfig, SearchPogema, bootstrap_ci, episode_metrics, robustness_ci, run_episode
from pogema_search.maps import MAP_TYPES, largest_free_component, make_obstacles
from pogema_search.policies import RandomWalk


def make(**kw):
    base = dict(size=16, density=0.2, num_agents=6, num_targets=2, obs_radius=3, detect_radius=2, seed=0)
    base.update(kw)
    env = SearchPogema(SearchConfig(**base))
    obs = env.reset()
    return env, obs


@pytest.mark.parametrize("mt", MAP_TYPES)
def test_map_shapes(mt):
    for s in (16, 32):
        m = make_obstacles(mt, s, 0.2, seed=1)
        assert m.shape == (s, s) and set(np.unique(m)) <= {0, 1}
        assert largest_free_component(m).sum() > 0


@pytest.mark.parametrize("start", ["cluster", "uniform"])
@pytest.mark.parametrize("dist", ["uniform", "prior"])
def test_layout_validity(start, dist):
    for seed in range(5):
        env, _ = make(start=start, target_dist=dist, seed=seed)
        pos = [tuple(p) for p in env.positions()]
        assert len(set(pos)) == len(pos), "agents overlap"
        for p in pos + env.targets:
            assert env.reachable[p], "start/target outside the agents' component"
        for t in env.targets:  # not detectable at t=0
            assert np.max(np.abs(env.positions() - np.array(t)), axis=1).min() > env.cfg.detect_radius
        assert (env.found_t < 0).all()
        assert math.isclose(env.prior.sum(), 1.0)
        assert env.prior[~env.inner].sum() == 0


def test_observation_matches_pogema_window():
    env, obs = make()
    r = env.r
    for o in obs:
        x, y = o.xy
        assert np.array_equal(o.local_obstacles, env.grid.obstacles[x - r:x + r + 1, y - r:y + r + 1])
        assert o.local_agents[r, r] == 1  # self in the centre
        assert o.known[x - r:x + r + 1, y - r:y + r + 1].all()


def test_dynamics_no_overlap_no_walls():
    env, obs = make(num_agents=10, max_steps=200, stop_when_all_found=False)
    pol = RandomWalk()
    pol.reset(env.spec(), seed=0)
    rng = np.random.default_rng(0)
    while not env.done:
        acts = rng.integers(0, 5, size=10)  # includes moves into walls and other agents
        env.step(acts)
        pos = env.positions()
        assert len({tuple(p) for p in pos}) == len(pos)
        assert not env.grid.obstacles[pos[:, 0], pos[:, 1]].any()
    assert env.t == 200


@pytest.mark.parametrize("blocks", [True, False])
def test_failures(blocks):
    env, obs = make(num_agents=8, fail_frac=0.5, fail_window=(0.0, 0.5), max_steps=60,
                    failed_blocks=blocks, stop_when_all_found=False)
    assert (env.fail_t > 0).sum() == 4
    pol = RandomWalk()
    pol.reset(env.spec())
    frozen = {}
    while not env.done:
        obs, _, _ = env.step(pol.act(obs))
        for i in np.flatnonzero(~env.active):
            p = tuple(env.positions()[i])
            frozen.setdefault(i, (p, env.searched[i].copy(), env.visits[i].sum()))
            assert frozen[i][0] == p, "failed agent moved"
            assert np.array_equal(frozen[i][1], env.searched[i]), "failed agent kept sensing"
            other_active_here = any(tuple(q) == p for j, q in enumerate(env.positions()) if env.active[j])
            assert env.grid.positions[p] == (1 if (blocks or other_active_here) else 0)
    assert (~env.active).sum() == 4


def test_detection():
    env, obs = make(num_targets=1, stop_when_all_found=True)
    x, y = env.grid.positions_xy[0]
    # put the hidden target exactly at detection range of agent 0
    env.targets = [(x, y + env.cfg.detect_radius)]
    env.target_arr = np.array(env.targets)
    env.step([0] * env.cfg.num_agents)
    assert env.found_t[0] == 1 and env.found_by[0] >= 0 and env.found_known[0, 0]
    assert env.done


def test_communication():
    env, obs = make(comm_radius=0)
    env.step([0] * 6)
    assert sum(env.hist["messages"]) == 0
    env, obs = make(comm_radius=math.inf, start="uniform")
    env.step([0] * 6)
    s = env.searched
    assert all(np.array_equal(s[0], s[i]) for i in range(6)), "global comm must merge all maps"
    assert env.hist["messages"][-1] == 6 * 5


def test_global_fast_path_equals_pairwise():
    res = []
    for rc in (math.inf, 1e9):
        cfg = SearchConfig(size=16, num_agents=7, num_targets=2, comm_radius=rc, start="uniform", max_steps=30,
                           stop_when_all_found=False, fail_frac=0.3)
        _, env = run_episode(cfg, RandomWalk(), seed=3)
        res.append((env.hist["messages"], env.hist["cells_tx"], env.searched.copy(), env.visits.copy()))
    assert res[0][0] == res[1][0] and res[0][1] == res[1][1]
    assert np.array_equal(res[0][2], res[1][2]) and np.array_equal(res[0][3], res[1][3])


def test_local_comm_is_single_hop():
    env, obs = make(num_agents=3, comm_radius=1.0, start="uniform", density=0.0)
    # chain 0 - 1 - 2 with spacing 1: 0 and 2 are not neighbours
    r = env.r
    cells = [(r + 5, r + 5), (r + 5, r + 6), (r + 5, r + 7)]
    for i, c in enumerate(cells):
        env.grid.positions[env.grid.positions_xy[i]] = 0
    for i, c in enumerate(cells):
        env.grid.positions_xy[i] = c
        env.grid.positions[c] = 1
    env.searched[:] = False
    env.searched[0, 0, 0] = True  # a fact only agent 0 knows
    env._communicate()
    assert env.searched[1, 0, 0] and not env.searched[2, 0, 0]
    env._communicate()
    assert env.searched[2, 0, 0]


def test_determinism():
    cfg = SearchConfig(size=16, num_agents=6, num_targets=2, comm_radius=3, fail_frac=0.25, seed=0)
    m1, _ = run_episode(cfg, RandomWalk(), seed=7)
    m2, _ = run_episode(cfg, RandomWalk(), seed=7)
    m3, _ = run_episode(cfg, RandomWalk(), seed=8)
    assert m1 == m2 and m1 != m3


def test_metrics_ranges():
    cfg = SearchConfig(size=16, num_agents=6, num_targets=3, max_steps=80)
    m, env = run_episode(cfg, RandomWalk(), seed=1)
    cov = np.array(env.hist["coverage"])
    assert (np.diff(cov) >= -1e-12).all()
    assert 0 <= m["revisit_ratio"] <= 1 and 0 <= m["coverage_auc"] <= m["coverage_final"] <= 1
    assert 0 <= m["t_first"] <= m["t_last"] <= cfg.max_steps
    assert m["success"] in (0.0, 1.0)


def test_bootstrap():
    x = np.random.default_rng(0).normal(5, 1, size=200)
    m, lo, hi = bootstrap_ci(x)
    assert lo < m < hi and hi - lo < 0.5
    r, lo, hi = robustness_ci(x, x)
    assert math.isclose(r, 1.0) and lo < 1 < hi


def test_failures_happen_during_search_for_every_method():
    from pogema_search.policies import make_policy
    cfg = SearchConfig(size=16, num_agents=8, num_targets=1, fail_frac=0.5)
    for name in ("swarm", "frontier", "random_walk"):
        for seed in range(5):
            m, env = run_episode(cfg, make_policy(name), seed=seed)
            assert env.fail_t.max() <= 9  # U[1, 0.5*size]
            if env.t >= 9:
                assert m["n_failed"] == 4
