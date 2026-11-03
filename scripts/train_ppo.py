"""Train the IPPO policy (resumable: rerun the same command to continue from runs/<name>/checkpoint.pt).

    python scripts/train_ppo.py --name ippo_v1 --budget-min 9
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pogema_search.rl.ppo import PPOConfig, Trainer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="ippo_v1")
    p.add_argument("--budget-min", type=float, default=9.0)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    cfg = PPOConfig(n_envs=a.n_envs, rollout_steps=a.rollout_steps, seed=a.seed)
    tr = Trainer(os.path.join("runs", a.name), cfg)
    tr.train(time_budget_s=60 * a.budget_min, max_iters=a.max_iters)


if __name__ == "__main__":
    main()
