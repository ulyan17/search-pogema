"""Shared actor-critic for IPPO (parameter sharing across agents, decentralized critic on local input)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .obs import C_G, C_L, N_SCAL


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.local = nn.Sequential(                       # 15x15
            nn.Conv2d(C_L, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # 8x8
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 4x4
            nn.Flatten(), nn.Linear(64 * 16, 256), nn.ReLU())
        self.glob = nn.Sequential(                        # 8x8
            nn.Conv2d(C_G, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),   # 4x4
            nn.Flatten(), nn.Linear(32 * 16, 128), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(256 + 128 + N_SCAL, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, 5)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, loc, glob, scal, mask):
        h = self.trunk(torch.cat([self.local(loc), self.glob(glob), scal], dim=1))
        logits = self.pi(h).masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(-1)
