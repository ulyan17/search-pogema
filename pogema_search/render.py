"""Episode rendering for GIFs: map with searched area, trails, targets, failures and a live coverage plot."""
from __future__ import annotations

from collections import deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OBST = np.array([0.17, 0.19, 0.24])
FREE = np.array([0.985, 0.985, 0.98])
SEARCH_CMAP = LinearSegmentedColormap.from_list("search", ["#d7ebfa", "#7fb6e6"])
AGENT_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46c0c8", "#f032e6", "#9a9a00",
                "#008080", "#9a6324", "#800000", "#000075", "#808000", "#e6a800", "#5a5aff", "#ff6f91"]


class Renderer:
    def __init__(self, env, title: str = "", trail: int = 14, cell_px: int = 14):
        r, s = env.r, env.cfg.size
        self.sl, self.r, self.size, self.title = slice(r, r + s), r, s, title
        self.trails = [deque(maxlen=trail) for _ in range(env.cfg.num_agents)]
        self.events = []  # (t, k) detection events
        h = max(4.2, s * cell_px / 100)
        self.fig = plt.figure(figsize=(h * 1.62, h), dpi=100)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.0, 0.62], height_ratios=[1, 1],
                                   wspace=0.12, hspace=0.35, left=0.02, right=0.97, top=0.92, bottom=0.08)
        self.ax = self.fig.add_subplot(gs[:, 0])
        self.ax_cov = self.fig.add_subplot(gs[0, 1])
        self.ax_info = self.fig.add_subplot(gs[1, 1])

    def update(self, env):
        """Call every step (also for frames that are not drawn) to keep trails and events."""
        pos = env.positions() - self.r
        for i, p in enumerate(pos):
            if env.active[i]:
                self.trails[i].append(tuple(p))
        for k in np.flatnonzero(env.found_t == env.t):
            self.events.append((env.t, int(k)))

    def frame(self, env) -> np.ndarray:
        ax, sl, r, s = self.ax, self.sl, self.r, self.size
        ax.clear()
        obst = env.grid.obstacles[sl, sl].astype(bool)
        st = env.searched_t[sl, sl].astype(float)
        img = np.tile(FREE, (s, s, 1))
        sea = (st >= 0) & ~obst
        if sea.any():
            tn = st / max(env.cfg.max_steps, 1)
            img[sea] = SEARCH_CMAP(np.clip(tn[sea] * 2.0, 0, 1))[:, :3]
        img[obst] = OBST
        ax.imshow(img, interpolation="nearest")
        if env.prior_center is not None:
            ax.contour(env.prior[sl, sl], levels=4, colors="#f0a030", linewidths=1.0, alpha=0.9)
        pos = env.positions() - r
        for a, b in env.last_links:
            ax.plot([pos[a, 1], pos[b, 1]], [pos[a, 0], pos[b, 0]], color="#666", lw=0.6, alpha=0.45, zorder=3)
        for i, tr in enumerate(self.trails):
            if len(tr) > 1:
                pts = np.array(tr)
                n = len(pts)
                for j in range(n - 1):
                    ax.plot(pts[j:j + 2, 1], pts[j:j + 2, 0], color=AGENT_COLORS[i % 16],
                            lw=1.6, alpha=0.15 + 0.6 * j / n, solid_capstyle="round", zorder=4)
        for k, (tx, ty) in enumerate(env.targets):
            found = env.found_t[k] >= 0
            if found:
                ax.scatter(ty - r, tx - r, s=420, facecolors="none", edgecolors="#2ca02c", linewidths=1.5, zorder=5)
            ax.scatter(ty - r, tx - r, marker="*", s=260, c="#2ca02c" if found else "#d62728",
                       edgecolors="white", linewidths=0.8, zorder=6)
        act = env.active
        ia = np.flatnonzero(act)
        ax.scatter(pos[ia, 1], pos[ia, 0], s=max(18, 900 / s), c=[AGENT_COLORS[i % 16] for i in ia],
                   edgecolors="white", linewidths=0.9, zorder=7)
        if (~act).any():
            ax.scatter(pos[~act, 1], pos[~act, 0], s=max(18, 900 / s), marker="X", c="#8a8a8a",
                       edgecolors="white", linewidths=0.6, zorder=7)
        ax.set_xlim(-0.5, s - 0.5), ax.set_ylim(s - 0.5, -0.5)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#999")

        # coverage panel
        T = env.cfg.max_steps
        cov = env.hist["coverage"]
        a2 = self.ax_cov
        a2.clear()
        a2.plot(range(len(cov)), cov, color="#2f7fc1", lw=2)
        for t, k in self.events:
            a2.axvline(t, color="#2ca02c", lw=1, alpha=0.7)
        a2.set_xlim(0, T), a2.set_ylim(0, 1.02)
        a2.set_title("searched area (green: detections)", fontsize=9)
        a2.tick_params(labelsize=7)
        a2.grid(alpha=0.25)
        # info panel
        a3 = self.ax_info
        a3.clear()
        a3.axis("off")
        nf = int((env.found_t >= 0).sum())
        c = env.cfg
        comm = "none" if c.comm_radius == 0 else ("global" if np.isinf(c.comm_radius) else f"r_c = {c.comm_radius:g}")
        lines = [f"policy      {self.title}", f"step        {env.t} / {T}",
                 f"targets     {nf} / {len(env.targets)} found",
                 f"agents      {int(act.sum())} / {len(act)} active",
                 f"comm        {comm}", f"prior       {c.target_dist}",
                 f"map         {c.map_type} {s}x{s}"]
        a3.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=8.5, color="#222")
        self.fig.suptitle("Search-POGEMA: decentralized search for hidden targets", fontsize=10.5, x=0.02,
                          ha="left", color="#222")
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        plt.close(self.fig)
