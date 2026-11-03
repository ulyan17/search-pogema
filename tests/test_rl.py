import os

import numpy as np
import pytest
import torch

from pogema_search import SearchConfig, SearchPogema, run_episode
from pogema_search.rl.model import ActorCritic
from pogema_search.rl.obs import C_G, C_L, G, R, ObsEncoder


@pytest.mark.parametrize("size", [16, 32, 64])
def test_encoder_shapes_and_mask(size):
    env = SearchPogema(SearchConfig(size=size, num_agents=6, num_targets=2, target_dist="prior"))
    obs = env.reset(seed=1)
    enc = ObsEncoder(env.spec())
    loc, glob, scal, mask = enc.encode_batch(obs)
    assert loc.shape == (6, C_L, 2 * R + 1, 2 * R + 1) and glob.shape == (6, C_G, G, G) and mask.shape == (6, 5)
    assert mask[:, 0].all() and np.isfinite(loc).all() and (loc >= 0).all() and (loc <= 1).all()
    r = env.r
    for i, o in enumerate(obs):  # masked moves are exactly the blocked ones
        for a, (dx, dy) in enumerate([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][1:], start=1):
            blocked = o.local_obstacles[r + dx, r + dy] or o.local_agents[r + dx, r + dy]
            assert mask[i, a] == (not blocked)
    assert glob[:, 3].sum(axis=(1, 2)).tolist() == [1.0] * 6


def test_model_forward_respects_mask():
    m = ActorCritic()
    mask = torch.zeros(3, 5, dtype=torch.bool)
    mask[:, 0] = True
    logits, v = m(torch.zeros(3, C_L, 2 * R + 1, 2 * R + 1), torch.zeros(3, C_G, G, G), torch.zeros(3, 2), mask)
    assert (logits.argmax(1) == 0).all() and v.shape == (3,)


@pytest.mark.skipif(not os.path.exists("runs/ippo_v1/model_final.pt"), reason="no trained model")
def test_trained_policy_runs():
    from pogema_search.policies import make_policy
    cfg = SearchConfig(size=16, num_agents=4, num_targets=1, max_steps=60, fail_frac=0.25)
    m, env = run_episode(cfg, make_policy("ippo", path="runs/ippo_v1/model_final.pt"), seed=0)
    assert 0 <= m["coverage_final"] <= 1
