"""Learned IPPO policy as a decentralized Policy: each agent encodes its own AgentObs and the shared network
picks its action (batched over agents only for speed)."""
from __future__ import annotations

import numpy as np
import torch

from ..rl.model import ActorCritic
from ..rl.obs import ObsEncoder
from .base import Policy

_CACHE = {}


class IPPOPolicy(Policy):
    name = "ippo"

    def __init__(self, path: str = "runs/ippo_v1/model_final.pt", greedy: bool = False):
        self.path, self.greedy = path, greedy
        if path not in _CACHE:
            torch.set_num_threads(1)
            m = ActorCritic()
            m.load_state_dict(torch.load(path, map_location="cpu"))
            m.eval()
            _CACHE[path] = m
        self.model = _CACHE[path]

    def reset(self, spec, seed=0):
        super().reset(spec, seed)
        self.enc = ObsEncoder(spec)
        self.gen = torch.Generator().manual_seed(int(seed))

    @torch.no_grad()
    def act(self, obs_list):
        idx = [i for i, o in enumerate(obs_list) if o.active]
        acts = [0] * len(obs_list)
        if not idx:
            return acts
        loc, glob, scal, mask = self.enc.encode_batch([obs_list[i] for i in idx])
        logits, _ = self.model(torch.from_numpy(loc), torch.from_numpy(glob), torch.from_numpy(scal),
                               torch.from_numpy(mask))
        if self.greedy:
            a = logits.argmax(1)
        else:
            a = torch.multinomial(torch.softmax(logits, 1), 1, generator=self.gen).squeeze(1)
        for j, i in enumerate(idx):
            acts[i] = int(a[j])
        return acts
