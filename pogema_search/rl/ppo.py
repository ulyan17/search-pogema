"""IPPO with a shared actor-critic.

Every active agent is a separate trajectory; it ends at the end of the episode or when the agent fails.
Time-limit ends are terminal, the network sees t / max_steps. Reward: +1 per detected target plus
beta per prior-weighted reachable cell the agent searched first in the team. Curriculum over task
phases (16x16 -> 32x32), checkpoints allow resuming.
"""
from __future__ import annotations

import csv
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch.distributions import Categorical

from ..env import SearchConfig, SearchPogema
from ..metrics import episode_metrics
from .model import ActorCritic
from .obs import ObsEncoder

DEFAULT_PHASES = [
    dict(name="s16", size=[16], num_agents=[4, 8], max_steps=128, min_iters=40, max_iters=150),
    dict(name="s32", size=[16, 32, 32, 32], num_agents=[4, 8, 16], max_steps=None, min_iters=60, max_iters=400),
]
COMMON = dict(density=[0.0, 0.1, 0.2, 0.3], num_targets=[1, 3], target_dist=["uniform", "prior"],
              fail_frac=[0.0, 0.25, 0.5], comm_radius=[0, 0, 5])


@dataclass
class PPOConfig:
    n_envs: int = 8
    rollout_steps: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    epochs: int = 4
    minibatch: int = 2048
    max_grad_norm: float = 0.5
    beta: float = 0.02            # shaping weight per newly searched cell
    advance_success: float = 0.85
    seed: int = 0
    torch_threads: int = 2


class TaskSampler:
    def __init__(self, phases, rng):
        self.phases, self.rng = phases, rng

    def sample(self, phase: int, seed: int) -> SearchConfig:
        ph = self.phases[phase]
        pick = lambda v: v[self.rng.integers(len(v))] if isinstance(v, list) else v  # noqa: E731
        size = int(pick(ph["size"]))
        d = {k: pick(v) for k, v in COMMON.items()}
        d.update(size=size, num_agents=int(pick(ph["num_agents"])), seed=seed,
                 max_steps=ph["max_steps"] or 8 * size)
        return SearchConfig.from_dict(d)


class Trainer:
    def __init__(self, out_dir: str, cfg: PPOConfig = PPOConfig(), phases=None):
        self.out, self.cfg = out_dir, cfg
        self.phases = phases or DEFAULT_PHASES
        os.makedirs(out_dir, exist_ok=True)
        torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(cfg.seed)
        self.model = ActorCritic()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)
        self.rng = np.random.default_rng(cfg.seed)
        self.it, self.phase, self.phase_start, self.samples = 0, 0, 0, 0
        self.ep_seed = 1_000_000  # training seeds never overlap tuning (10000+) or evaluation (0..)
        self.recent = deque(maxlen=200)
        self.sampler = TaskSampler(self.phases, self.rng)
        self.ckpt = os.path.join(out_dir, "checkpoint.pt")
        if os.path.exists(self.ckpt):
            self._load()
        self.envs = [self._new_env() for _ in range(cfg.n_envs)]

    def _load(self):
        st = torch.load(self.ckpt, weights_only=False)
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        self.it, self.phase, self.phase_start = st["it"], st["phase"], st["phase_start"]
        self.samples, self.ep_seed = st["samples"], st["ep_seed"]
        self.recent = deque(st["recent"], maxlen=200)
        self.rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"])
        print(f"resumed: it={self.it} phase={self.phase} samples={self.samples}", flush=True)

    def save(self):
        st = dict(model=self.model.state_dict(), opt=self.opt.state_dict(), it=self.it, phase=self.phase,
                  phase_start=self.phase_start, samples=self.samples, ep_seed=self.ep_seed,
                  recent=list(self.recent), rng=self.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                  ppo=asdict(self.cfg), phases=self.phases)
        tmp = self.ckpt + ".tmp"
        torch.save(st, tmp)
        os.replace(tmp, self.ckpt)
        torch.save(self.model.state_dict(), os.path.join(self.out, f"model_phase{self.phase}.pt"))
        torch.save(self.model.state_dict(), os.path.join(self.out, "model_latest.pt"))

    def _new_env(self):
        self.ep_seed += 1
        cfg = self.sampler.sample(self.phase, self.ep_seed)
        env = SearchPogema(cfg)
        obs = env.reset(seed=self.ep_seed)
        return {"env": env, "obs": obs, "enc": ObsEncoder(env.spec()), "ret": 0.0, "uid": self.ep_seed}

    @torch.no_grad()
    def rollout(self):
        c = self.cfg
        buf = {k: [] for k in ("loc", "glob", "scal", "mask", "act", "logp", "val", "rew", "done", "seq")}
        last_idx = {}   # seq -> index of the last transition of a live sequence
        ep_stats = []
        for _ in range(c.rollout_steps):
            items = []
            for e, slot in enumerate(self.envs):
                for o in slot["obs"]:
                    if o.active:
                        items.append((e, o))
            enc = [self.envs[e]["enc"].encode(o) for e, o in items]
            loc = torch.from_numpy(np.stack([x[0] for x in enc]))
            glob = torch.from_numpy(np.stack([x[1] for x in enc]))
            scal = torch.from_numpy(np.stack([x[2] for x in enc]))
            mask = torch.from_numpy(np.stack([x[3] for x in enc]))
            logits, val = self.model(loc, glob, scal, mask)
            dist = Categorical(logits=logits)
            act = dist.sample()
            logp = dist.log_prob(act)
            actions = {e: [0] * slot["env"].cfg.num_agents for e, slot in enumerate(self.envs)}
            for j, (e, o) in enumerate(items):
                actions[e][o.agent_id] = int(act[j])
            base = len(buf["act"])
            for k, v in (("loc", loc), ("glob", glob), ("scal", scal), ("mask", mask), ("act", act),
                         ("logp", logp), ("val", val)):
                buf[k].extend(v)
            rew = np.zeros(len(items))
            done = np.zeros(len(items), dtype=bool)
            for e, slot in enumerate(self.envs):
                env = slot["env"]
                slot["obs"], ep_done, _ = env.step(actions[e])
                r_env = env.step_detect + c.beta * env.new_mass
                slot["ret"] += float(r_env.sum())
                for j, (ee, o) in enumerate(items):
                    if ee == e:
                        rew[j] = r_env[o.agent_id]
                        done[j] = ep_done or not env.active[o.agent_id]
                if ep_done:
                    m = episode_metrics(env)
                    m.update(ret=slot["ret"], size=env.cfg.size, n=env.cfg.num_agents)
                    ep_stats.append(m)
                    self.recent.append(m["success"])
            for j, (e, o) in enumerate(items):
                seq = (self.envs[e]["uid"], o.agent_id)
                buf["rew"].append(float(rew[j]))
                buf["done"].append(bool(done[j]))
                buf["seq"].append(seq)
                if done[j]:
                    last_idx.pop(seq, None)
                else:
                    last_idx[seq] = base + j
            for e, slot in enumerate(self.envs):
                if slot["env"].done:
                    self.envs[e] = self._new_env()
        # bootstrap values for live sequences
        boot = {}
        items = [(e, o) for e, slot in enumerate(self.envs) for o in slot["obs"] if o.active]
        if items:
            enc = [self.envs[e]["enc"].encode(o) for e, o in items]
            ten = [torch.from_numpy(np.stack([x[i] for x in enc])) for i in range(4)]
            _, v = self.model(*ten)
            for j, (e, o) in enumerate(items):
                boot[(self.envs[e]["uid"], o.agent_id)] = float(v[j])
        return buf, boot, ep_stats

    def gae(self, buf, boot):
        c = self.cfg
        n = len(buf["rew"])
        val = torch.stack(buf["val"]).numpy()
        rew, done = np.array(buf["rew"]), np.array(buf["done"])
        seqs = {}
        for i, s in enumerate(buf["seq"]):
            seqs.setdefault(s, []).append(i)
        adv = np.zeros(n, dtype=np.float32)
        for s, idx in seqs.items():
            nxt_v = 0.0 if done[idx[-1]] else boot.get(s, 0.0)
            g = 0.0
            for i in reversed(idx):
                nonterm = 0.0 if done[i] else 1.0
                delta = rew[i] + c.gamma * nxt_v * nonterm - val[i]
                g = delta + c.gamma * c.lam * nonterm * g
                adv[i] = g
                nxt_v = val[i]
        return adv, adv + val

    def update(self, buf, adv, ret):
        c = self.cfg
        data = {k: torch.stack(buf[k]) for k in ("loc", "glob", "scal", "mask", "act", "logp")}
        adv_t, ret_t = torch.from_numpy(adv), torch.from_numpy(ret.astype(np.float32))
        n = len(adv)
        stats = []
        for _ in range(c.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, c.minibatch):
                mb = perm[s:s + c.minibatch]
                logits, v = self.model(data["loc"][mb], data["glob"][mb], data["scal"][mb], data["mask"][mb])
                dist = Categorical(logits=logits)
                lp = dist.log_prob(data["act"][mb])
                ratio = torch.exp(lp - data["logp"][mb])
                a = adv_t[mb]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pl = -torch.min(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a).mean()
                vl = 0.5 * ((v - ret_t[mb]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pl + c.vf_coef * vl - c.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kl = (data["logp"][mb] - lp).mean().item()
                stats.append((pl.item(), vl.item(), ent.item(), kl))
        return np.mean(stats, axis=0)

    def train(self, time_budget_s: float, max_iters: int | None = None, save_every: int = 5):
        log_path = os.path.join(self.out, "learning_curve.csv")
        new = not os.path.exists(log_path)
        t0 = time.time()
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["it", "phase", "samples", "wall_s", "episodes", "success", "t_last_norm",
                            "coverage_final", "ep_return", "pi_loss", "v_loss", "entropy", "kl", "fps"])
            while time.time() - t0 < time_budget_s and (max_iters is None or self.it < max_iters):
                ti = time.time()
                buf, boot, eps = self.rollout()
                adv, ret = self.gae(buf, boot)
                pl, vl, ent, kl = self.update(buf, adv, ret)
                self.it += 1
                self.samples += len(adv)
                fps = len(adv) / (time.time() - ti)
                succ = np.mean([e["success"] for e in eps]) if eps else float("nan")
                tl = np.mean([e["t_last"] / (8 * e["size"] if self.phases[self.phase]["max_steps"] is None
                                             else self.phases[self.phase]["max_steps"]) for e in eps]) \
                    if eps else float("nan")
                cov = np.mean([e["coverage_final"] for e in eps]) if eps else float("nan")
                er = np.mean([e["ret"] for e in eps]) if eps else float("nan")
                w.writerow([self.it, self.phase, self.samples, round(time.time() - t0, 1), len(eps),
                            succ, tl, cov, er, pl, vl, ent, kl, round(fps)])
                f.flush()
                if self.it % 5 == 0:
                    rs = np.mean(self.recent) if self.recent else float("nan")
                    print(f"it {self.it} ph {self.phase} samples {self.samples} succ(200) {rs:.2f} "
                          f"t_last {tl:.2f} ret {er:.1f} ent {ent:.2f} kl {kl:.4f} {fps:.0f} samples/s", flush=True)
                self._maybe_advance()
                if self.it % save_every == 0:
                    self.save()
        self.save()

    def _maybe_advance(self):
        ph = self.phases[self.phase]
        k = self.it - self.phase_start
        if self.phase + 1 >= len(self.phases):
            return
        ready = k >= ph["min_iters"] and len(self.recent) >= 100 and np.mean(self.recent) >= self.cfg.advance_success
        if ready or k >= ph["max_iters"]:
            self.save()
            self.phase += 1
            self.phase_start = self.it
            self.recent.clear()
            self.envs = [self._new_env() for _ in range(self.cfg.n_envs)]
            print(f"== advance to phase {self.phase} ({self.phases[self.phase]['name']}) at it {self.it}", flush=True)
