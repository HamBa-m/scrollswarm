"""
Raycasting sensor module for ScrollSwarm (Saturn Ring Surfing).

Implements a 7-ray forward rangefinder:
- 7 rays spanning the upcoming half of the screen upfront (angles in [-60°, +60°]).
- Intersects ring sheath boundaries (ceiling and floor), dynamic polygon asteroids,
  and collection targets.
- Merges boundary clearance and asteroid warnings into a continuous, fixed-size vector.
- Vector shape: [y_norm, 7 obstacle ray distances, 7 target ray distances] (15 floats).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from scrollswarm.map import ScrollingMap
from scrollswarm.items import Asteroid, Target


@dataclass
class RayHit:
    """Stores the result of a single ray cast."""
    distance: float            # In pixels [0, max_range]
    norm_distance: float       # Normalized [0.0, 1.0]
    hit_type: str              # "boundary", "asteroid", "target", or "none"
    endpoint: Tuple[float, float]


class RaySensor:
    """
    7-ray LiDAR-style sensor projecting forward into the oncoming half of the screen.
    """

    def __init__(
        self,
        num_rays: int = 7,
        max_range: float = 400.0,      # Upcoming half of 800px screen
        fov_degrees: float = 120.0,     # Spans -60° (up-forward) to +60° (down-forward)
    ):
        self.num_rays = num_rays
        self.max_range = float(max_range)
        self.fov_degrees = float(fov_degrees)

        # Compute ray angles (in radians), from top-forward to bottom-forward
        half_fov = np.radians(self.fov_degrees / 2.0)
        self.angles = np.linspace(-half_fov, half_fov, self.num_rays, dtype=np.float32)

    def cast_rays(
        self,
        origin_x: float,
        origin_y: float,
        scrolling_map: "ScrollingMap",
    ) -> Tuple[List[RayHit], List[RayHit]]:
        """
        Casts 7 rays from (origin_x, origin_y):
        - Returns (obstacle_hits, target_hits).
        - Obstacle hits check ring sheath boundaries and dynamic polygon asteroids.
        - Target hits check collection targets.
        """
        obs_hits: List[RayHit] = []
        tgt_hits: List[RayHit] = []

        h = float(scrolling_map.height)
        ceil_y = float(scrolling_map.config.ceiling_y)
        floor_y = h - float(scrolling_map.config.floor_margin)

        # Retrieve asteroids (or legacy trees)
        asteroids: List[Asteroid] = getattr(scrolling_map, "asteroids", [])
        if not asteroids and hasattr(scrolling_map, "trees"):
            asteroids = scrolling_map.trees

        for angle in self.angles:
            cos_a = float(np.cos(angle))
            sin_a = float(np.sin(angle))

            # -------------------------------------------------------------
            # 1. OBSTACLE INTERSECTIONS (Ring Boundaries & Polygon Asteroids)
            # -------------------------------------------------------------
            closest_obs_dist = self.max_range
            obs_hit_type = "none"

            # A. Analytical ceiling and floor intersections
            if sin_a < -1e-5:
                # Pointing upwards: intersects ceiling y = ceil_y
                t_ceil = (ceil_y - origin_y) / sin_a
                if 0.0 <= t_ceil < closest_obs_dist:
                    closest_obs_dist = t_ceil
                    obs_hit_type = "boundary"
            elif sin_a > 1e-5:
                # Pointing downwards: intersects floor y = floor_y
                t_floor = (floor_y - origin_y) / sin_a
                if 0.0 <= t_floor < closest_obs_dist:
                    closest_obs_dist = t_floor
                    obs_hit_type = "boundary"

            # B. Check dynamic polygon asteroids
            for ast in asteroids:
                vx = ast.x - origin_x
                vy = ast.y - origin_y
                t_proj = vx * cos_a + vy * sin_a

                # Bounding sphere rejection test
                if t_proj < -ast.radius or t_proj > closest_obs_dist + ast.radius:
                    continue
                perp_sq = (vx * vx + vy * vy) - (t_proj * t_proj)
                if perp_sq > ast.radius * ast.radius:
                    continue

                # Exact 2D ray-segment determinant checks against polygon edges
                edges = ast.edges if hasattr(ast, "edges") else []
                if not edges and hasattr(ast, "outer_radius"):
                    # Fallback for circular entities
                    r_sq = ast.outer_radius * ast.outer_radius
                    if perp_sq < r_sq:
                        half_chord = np.sqrt(max(0.0, r_sq - perp_sq))
                        entry_dist = t_proj - half_chord
                        if 0.0 <= entry_dist < closest_obs_dist:
                            closest_obs_dist = entry_dist
                            obs_hit_type = "asteroid"
                    continue

                for (ax, ay), (bx, by) in edges:
                    ex = bx - ax
                    ey = by - ay
                    denom = cos_a * ey - sin_a * ex
                    if abs(denom) < 1e-6:
                        continue  # Ray and edge are parallel

                    dx = ax - origin_x
                    dy = ay - origin_y
                    t = (dx * ey - dy * ex) / denom
                    u = (dx * sin_a - dy * cos_a) / denom

                    if 0.0 <= t < closest_obs_dist and 0.0 <= u <= 1.0:
                        closest_obs_dist = t
                        obs_hit_type = "asteroid"

            # Clamp obstacle hit
            closest_obs_dist = float(np.clip(closest_obs_dist, 0.0, self.max_range))
            end_x = origin_x + closest_obs_dist * cos_a
            end_y = origin_y + closest_obs_dist * sin_a

            obs_hits.append(RayHit(
                distance=closest_obs_dist,
                norm_distance=closest_obs_dist / self.max_range,
                hit_type=obs_hit_type,
                endpoint=(end_x, end_y),
            ))

            # -------------------------------------------------------------
            # 2. TARGET INTERSECTIONS (Analytical Ray-Circle)
            # -------------------------------------------------------------
            closest_tgt_dist = self.max_range
            tgt_hit_type = "none"

            for target in scrolling_map.targets:
                if target.collected:
                    continue

                vx = target.x - origin_x
                vy = target.y - origin_y
                t_proj = vx * cos_a + vy * sin_a

                if t_proj <= 0 or t_proj >= closest_tgt_dist + target.size:
                    continue

                perp_sq = (vx * vx + vy * vy) - (t_proj * t_proj)
                r_sq = target.size * target.size

                if perp_sq < r_sq:
                    half_chord = np.sqrt(max(0.0, r_sq - perp_sq))
                    entry_dist = t_proj - half_chord
                    if 0.0 <= entry_dist < closest_tgt_dist:
                        closest_tgt_dist = entry_dist
                        tgt_hit_type = "target"

            closest_tgt_dist = float(np.clip(closest_tgt_dist, 0.0, self.max_range))
            tgt_end_x = origin_x + closest_tgt_dist * cos_a
            tgt_end_y = origin_y + closest_tgt_dist * sin_a

            tgt_hits.append(RayHit(
                distance=closest_tgt_dist,
                norm_distance=closest_tgt_dist / self.max_range,
                hit_type=tgt_hit_type,
                endpoint=(tgt_end_x, tgt_end_y),
            ))

        return obs_hits, tgt_hits

    def observe(
        self,
        origin_x: float,
        origin_y: float,
        scrolling_map: "ScrollingMap",
    ) -> np.ndarray:
        """
        Generates the 15-dimensional observation vector:
        [normalized_y, 7 obstacle ray distances, 7 target ray distances].
        """
        obs_hits, tgt_hits = self.cast_rays(origin_x, origin_y, scrolling_map)
        h = float(scrolling_map.height)
        y_norm = float(np.clip(origin_y / h, 0.0, 1.0))

        obs_dists = [hit.norm_distance for hit in obs_hits]
        tgt_dists = [hit.norm_distance for hit in tgt_hits]

        vec = np.array([y_norm] + obs_dists + tgt_dists, dtype=np.float32)
        return np.clip(vec, 0.0, 1.0)
