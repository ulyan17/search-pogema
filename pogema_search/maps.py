"""Obstacle maps: POGEMA random maps, pogema-toolbox mazes and warehouses.

All maps are (size, size) int arrays, 1 = obstacle, without POGEMA's border.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from pogema import GridConfig
from pogema.generator import generate_obstacles
from pogema_toolbox.generators.maze_generator import MazeGenerator
from pogema_toolbox.generators.warehouse_generator import WarehouseConfig, generate_warehouse

MAP_TYPES = ("random", "maze", "warehouse")

FOUR_CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _fit(grid: np.ndarray, size: int) -> np.ndarray:
    """Crop or zero-pad (free cells) a grid to exactly ``size x size``."""
    out = np.zeros((size, size), dtype=np.int32)
    h, w = min(size, grid.shape[0]), min(size, grid.shape[1])
    out[:h, :w] = grid[:h, :w]
    return out


def _str_to_array(s: str) -> np.ndarray:
    return np.array([[1 if c == "#" else 0 for c in line] for line in s.split()], dtype=np.int32)


def random_map(size: int, density: float, seed: int) -> np.ndarray:
    return generate_obstacles(GridConfig(size=size, density=density, seed=seed)).astype(np.int32)


def maze_map(size: int, density: float, seed: int, wall_components: int = 6,
             go_straight: float = 0.8) -> np.ndarray:
    s = MazeGenerator.generate_maze(width=size, height=size, obstacle_density=density,
                                    wall_components=wall_components, go_straight=go_straight, seed=seed)
    return _fit(_str_to_array(s), size)


def warehouse_map(size: int, seed: int | None = None) -> np.ndarray:
    """4x2 shelves with narrow aisles, fitted into size. Deterministic, seed is ignored."""
    ww, wh, hg, vg, bg = 4, 2, 1, 2, 2
    rows = max(1, (size - vg) // (vg + wh))
    cols = max(1, (size - 2 * bg + hg) // (ww + hg))
    cfg = WarehouseConfig(wall_width=ww, wall_height=wh, walls_in_row=cols, walls_rows=rows,
                          bottom_gap=bg, horizontal_gap=hg, vertical_gap=vg)
    s = generate_warehouse(cfg).replace("!", ".")
    return _fit(_str_to_array(s), size)


def make_obstacles(map_type: str, size: int, density: float, seed: int) -> np.ndarray:
    if map_type == "random":
        return random_map(size, density, seed)
    if map_type == "maze":
        return maze_map(size, density, seed)
    if map_type == "warehouse":
        return warehouse_map(size, seed)
    raise ValueError(f"unknown map_type {map_type!r}, expected one of {MAP_TYPES}")


def largest_free_component(obstacles: np.ndarray) -> np.ndarray:
    """Boolean mask of the largest 4-connected component of free cells."""
    labels, n = ndimage.label(obstacles == 0, structure=FOUR_CONN)
    if n == 0:
        raise ValueError("map has no free cells")
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def bfs_distances(free: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """4-connected BFS distances from ``start`` over ``free`` cells (-1 = unreachable)."""
    h, w = free.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    sx, sy = start
    dist[sx, sy] = 0
    frontier = [(sx, sy)]
    d = 0
    while frontier:
        d += 1
        nxt = []
        for x, y in frontier:
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and free[nx, ny] and dist[nx, ny] < 0:
                    dist[nx, ny] = d
                    nxt.append((nx, ny))
        frontier = nxt
    return dist
