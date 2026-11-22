"""
Raycasting sensor module for ScrollSwarm agents.

Implements a 7-ray forward rangefinder:
- 7 rays spanning the upcoming half of the screen upfront (angles in [-60°, +60°]).
- Rays naturally intersect boundary mountain ridges (top and bottom), circular trees,
  and collection targets.
- Merges boundary clearance and obstacle detection into a continuous, fixed-size vector.
- Vector shape: [y_norm, 7 obstacle ray distances, 7 target ray distances] (15 floats).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from scrollswarm.map import ScrollingMap
from scrollswarm.items import Tree, Target


@dataclass
class RayHit:
    """Stores the result of a single ray cast."""
    distance: float            # In pixels [0, max_range]
    norm_distance: float       # Normalized [0.0, 1.0]
    hit_type: str              # "mountain", "tree", "target", or "none"
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
        scrolling_map: ScrollingMap,
    ) -> Tuple[List[RayHit], List[RayHit]]:
        """
        Casts 7 rays from (origin_x, origin_y):
        - Returns (obstacle_hits, target_hits).
        - Obstacle hits check mountain boundaries and trees.
        - Target hits check collection targets.
        """
        obs_hits: List[RayHit] = []
        tgt_hits: List[RayHit] = []

        w = scrolling_map.width
        h = scrolling_map.height

        for angle in self.angles:
            cos_a = float(np.cos(angle))
            sin_a = float(np.sin(angle))

            # -------------------------------------------------------------
            # 1. OBSTACLE INTERSECTIONS (Mountain Ridges & Trees)
            # -------------------------------------------------------------
            closest_obs_dist = self.max_range
            obs_hit_type = "none"

            # A. Check mountain boundaries along the ray
            # Column-by-column marching forward
            start_col = max(0, int(origin_x))
            max_col = min(w - 1, int(origin_x + self.max_range * cos_a))

            if cos_a > 1e-5:
                # March through columns x in [start_col, max_col]
                for col in range(start_col, max_col + 1):
                    # Ray distance s at this column
                    s = (col - origin_x) / cos_a
                    if s > closest_obs_dist:
                        break

                    y_ray = origin_y + s * sin_a

                    # Check collision with top or bottom mountain
                    top_bound = scrolling_map.top_heights[col]
                    bot_bound = h - scrolling_map.bot_heights[col]

                    if y_ray <= top_bound or y_ray >= bot_bound or y_ray <= 0 or y_ray >= h:
                        closest_obs_dist = s
                        obs_hit_type = "mountain"
                        break

            # B. Check trees (analytical ray-circle intersection)
            for tree in scrolling_map.trees:
                # Vector from ray origin to tree center
                vx = tree.x - origin_x
                vy = tree.y - origin_y

                # Projection along ray
                t_proj = vx * cos_a + vy * sin_a
                if t_proj <= 0 or t_proj >= closest_obs_dist + tree.outer_radius:
                    continue

                # Perpendicular distance squared
                perp_sq = (vx * vx + vy * vy) - (t_proj * t_proj)
                r_sq = tree.outer_radius * tree.outer_radius

                if perp_sq < r_sq:
                    # Ray intersects the circle
                    half_chord = np.sqrt(max(0.0, r_sq - perp_sq))
                    entry_dist = t_proj - half_chord
                    if 0 <= entry_dist < closest_obs_dist:
                        closest_obs_dist = entry_dist
                        obs_hit_type = "tree"

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
            # 2. TARGET INTERSECTIONS
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
                    if 0 <= entry_dist < closest_tgt_dist:
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
        scrolling_map: ScrollingMap,
    ) -> np.ndarray:
        """
        Returns a compact 1D observation vector of shape (15,):
        [
          self_y / H,
          7 normalized obstacle ray distances (0.0=contact, 1.0=clear),
          7 normalized target ray distances (0.0=contact, 1.0=no target)
        ]
        """
        obs_hits, tgt_hits = self.cast_rays(origin_x, origin_y, scrolling_map)

        y_norm = origin_y / float(scrolling_map.height)
        obs_dists = [h.norm_distance for h in obs_hits]
        tgt_dists = [h.norm_distance for h in tgt_hits]

        obs_vector = np.array([y_norm] + obs_dists + tgt_dists, dtype=np.float32)
        return obs_vector
