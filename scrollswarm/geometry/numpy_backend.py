"""
Pure NumPy reference implementation of 2D analytical geometry for ScrollSwarm.

Provides baseline implementations for:
1. Multi-ray LiDAR intersection against corridor boundaries, polygon edges, and targets.
2. Point-in-polygon containment and point-to-segment Euclidean distance.
3. Physical agent-obstacle collision testing.

Zero C-compiler or JIT dependencies; guaranteed to run in any Python environment.
"""

from typing import List, Tuple
import numpy as np


def cast_rays_numpy(
    origin_x: float,
    origin_y: float,
    angles: np.ndarray,
    max_range: float,
    ceil_y: float,
    floor_y: float,
    obs_meta: np.ndarray,      # shape (N_obs, 4): [x, y, radius, max_dim]
    obs_edges: np.ndarray,     # shape (N_edges, 4): [ax, ay, bx, by]
    obs_edge_ranges: np.ndarray, # shape (N_obs, 2): [start_idx, end_idx]
    target_data: np.ndarray,   # shape (N_tgts, 3): [x, y, radius]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Casts LiDAR rays from (origin_x, origin_y) across given angles using NumPy.

    Returns:
        obs_dists: (K,) float32 - Distance to nearest obstacle/boundary hit [0, max_range].
        obs_types: (K,) int32   - 0: none, 1: boundary, 2: obstacle.
        obs_ends:  (K, 2) float32 - Absolute (x, y) coordinates of ray hit terminus.
        tgt_dists: (K,) float32 - Distance to nearest target hit [0, max_range].
        tgt_types: (K,) int32   - 0: none, 1: target.
        tgt_ends:  (K, 2) float32 - Absolute (x, y) coordinates of target terminus.
    """
    num_rays = len(angles)
    obs_dists = np.full(num_rays, max_range, dtype=np.float32)
    obs_types = np.zeros(num_rays, dtype=np.int32)
    obs_ends = np.zeros((num_rays, 2), dtype=np.float32)

    tgt_dists = np.full(num_rays, max_range, dtype=np.float32)
    tgt_types = np.zeros(num_rays, dtype=np.int32)
    tgt_ends = np.zeros((num_rays, 2), dtype=np.float32)

    num_obs = len(obs_meta)
    num_tgts = len(target_data)

    for k in range(num_rays):
        angle = angles[k]
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))

        closest_obs = max_range
        hit_type = 0

        # 1. Analytical ceiling and floor intersections
        if sin_a < -1e-5:
            t_ceil = (ceil_y - origin_y) / sin_a
            if 0.0 <= t_ceil < closest_obs:
                closest_obs = t_ceil
                hit_type = 1  # boundary
        elif sin_a > 1e-5:
            t_floor = (floor_y - origin_y) / sin_a
            if 0.0 <= t_floor < closest_obs:
                closest_obs = t_floor
                hit_type = 1  # boundary

        # 2. Polygon obstacle intersections
        for i in range(num_obs):
            ox = obs_meta[i, 0]
            oy = obs_meta[i, 1]
            max_dim = obs_meta[i, 3]

            vx = ox - origin_x
            vy = oy - origin_y
            t_proj = vx * cos_a + vy * sin_a

            # Bounding circle rejection
            if t_proj < -max_dim or t_proj > closest_obs + max_dim:
                continue

            start_e = int(obs_edge_ranges[i, 0])
            end_e = int(obs_edge_ranges[i, 1])

            for e in range(start_e, end_e):
                ax = obs_edges[e, 0]
                ay = obs_edges[e, 1]
                bx = obs_edges[e, 2]
                by = obs_edges[e, 3]

                ex = bx - ax
                ey = by - ay
                denom = cos_a * ey - sin_a * ex
                if abs(denom) < 1e-6:
                    continue

                dx = ax - origin_x
                dy = ay - origin_y
                t = (dx * ey - dy * ex) / denom
                u = (dx * sin_a - dy * cos_a) / denom

                if 0.0 <= t < closest_obs and 0.0 <= u <= 1.0:
                    closest_obs = t
                    hit_type = 2  # obstacle

        closest_obs = float(np.clip(closest_obs, 0.0, max_range))
        obs_dists[k] = closest_obs
        obs_types[k] = hit_type
        obs_ends[k, 0] = origin_x + closest_obs * cos_a
        obs_ends[k, 1] = origin_y + closest_obs * sin_a

        # 3. Ray-target intersections
        closest_tgt = max_range
        tgt_hit = 0

        for j in range(num_tgts):
            tx = target_data[j, 0]
            ty = target_data[j, 1]
            tr = target_data[j, 2]

            vx = tx - origin_x
            vy = ty - origin_y
            t_proj = vx * cos_a + vy * sin_a

            if t_proj <= 0.0 or t_proj >= closest_tgt + tr:
                continue

            perp_sq = (vx * vx + vy * vy) - (t_proj * t_proj)
            r_sq = tr * tr

            if perp_sq < r_sq:
                half_chord = np.sqrt(max(0.0, r_sq - perp_sq))
                entry_dist = t_proj - half_chord
                if 0.0 <= entry_dist < closest_tgt:
                    closest_tgt = entry_dist
                    tgt_hit = 1

        closest_tgt = float(np.clip(closest_tgt, 0.0, max_range))
        tgt_dists[k] = closest_tgt
        tgt_types[k] = tgt_hit
        tgt_ends[k, 0] = origin_x + closest_tgt * cos_a
        tgt_ends[k, 1] = origin_y + closest_tgt * sin_a

    return obs_dists, obs_types, obs_ends, tgt_dists, tgt_types, tgt_ends


def point_in_polygon_numpy(px: float, py: float, vertices: np.ndarray) -> bool:
    """Ray-casting test for point in 2D polygon with vertices shape (V, 2)."""
    n = len(vertices)
    if n == 0:
        return False
    inside = False
    p1x = vertices[0, 0]
    p1y = vertices[0, 1]
    for i in range(n + 1):
        idx = i % n
        p2x = vertices[idx, 0]
        p2y = vertices[idx, 1]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def distance_to_polygon_numpy(
    px: float,
    py: float,
    vertices: np.ndarray,
    edges: np.ndarray,
) -> float:
    """Computes minimum Euclidean distance from point (px, py) to polygon boundary."""
    if point_in_polygon_numpy(px, py, vertices):
        return 0.0

    min_dist_sq = float("inf")
    n_edges = len(edges)
    for e in range(n_edges):
        ax = edges[e, 0]
        ay = edges[e, 1]
        bx = edges[e, 2]
        by = edges[e, 3]

        vx = bx - ax
        vy = by - ay
        wx = px - ax
        wy = py - ay
        c1 = wx * vx + wy * vy
        if c1 <= 0.0:
            dist_sq = (px - ax) ** 2 + (py - ay) ** 2
        else:
            c2 = vx * vx + vy * vy
            if c2 <= c1:
                dist_sq = (px - bx) ** 2 + (py - by) ** 2
            else:
                b = c1 / c2
                proj_x = ax + b * vx
                proj_y = ay + b * vy
                dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq

    return float(np.sqrt(min_dist_sq))
