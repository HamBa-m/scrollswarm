"""
Raycasting and perception sensor module for ScrollSwarm.

Provides a multi-modal 16-dimensional observation vector for each UAV:
1. Normalized altitude (1 float): y / H in [0.0, 1.0]
2. 7-ray forward LiDAR rangefinder (7 floats): normalized obstacle ray distances in [0.0, 1.0]
3. Nearest target relative bearing & proximity (4 floats):
   [dx / W, dy / H, dist / diag, in_photo_range]
4. Nearest living neighbor drone relative bearing & communication link (4 floats):
   [dx / W, dy / H, dist / diag, in_comm_range]
Total: 16 floats, bounded in [-1.0, 1.0].
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from scrollswarm.map import ScrollingMap
    from scrollswarm.agent import Agent
from scrollswarm.items import Obstacle, Target
from scrollswarm.geometry import (
    cast_rays_numpy,
    cast_rays_numba,
    cast_rays_cpp,
    extract_geometry_buffers,
    resolve_backend,
    is_numba_available,
    is_cpp_available,
)


@dataclass
class RayHit:
    """
    Result of a single ray cast from the LiDAR rangefinder.

    Attributes:
        distance: Absolute ray travel distance in pixels, clamped to
            ``[0, max_range]``.
        norm_distance: Normalized distance in ``[0.0, 1.0]``
            (``distance / max_range``).
        hit_type: Nature of the nearest intersection:
            ``"boundary"`` (ceiling or floor), ``"obstacle"`` (polygon
            structure), or ``"none"`` (ray reached max range without contact).
        endpoint: Absolute ``(x, y)`` coordinate of the ray terminus in pixels.
    """

    distance: float  # In pixels [0, max_range]
    norm_distance: float  # Normalized [0.0, 1.0]
    hit_type: str  # "boundary", "obstacle", or "none"
    endpoint: Tuple[float, float]


class RaySensor:
    """
    Multi-modal perception sensor for a ScrollSwarm UAV.

    Combines three sensing modalities into one 16-dimensional observation:

    - **LiDAR rangefinder:** 7 rays spanning a 120° forward FOV, detecting
      corridor boundaries and polygon obstacles via exact analytic
      ray–edge intersection (vector cross-product).
    - **Target tracker:** Relative bearing and distance to the nearest
      uninspected waypoint target.
    - **Neighbor proximity sensor:** Relative bearing and distance to the
      nearest living teammate drone, plus a binary communication-range flag.
    """

    def __init__(
        self,
        num_rays: int = 7,
        max_range: float = 400.0,  # Default: forward half of 800px screen
        fov_degrees: float = 120.0,  # Spans -60° (up-forward) to +60° (down-forward)
        photo_radius: float = 45.0,
        comm_range: float = 180.0,
        safe_proximity_dist: float = 28.0,
        backend: str = "auto",
    ):
        """
        Initializes the RaySensor.

        Args:
            num_rays: Number of LiDAR rays cast per observation step.
                Default 7 covers the full 120° FOV at 20° increments.
            max_range: Maximum ray travel distance in pixels.  Rays that reach
                this limit without a hit report ``norm_distance = 1.0``.
            fov_degrees: Total angular span of the ray fan in degrees,
                centred on the forward (rightward) direction.  Default 120°
                spans −60° (up-forward) to +60° (down-forward).
            photo_radius: Maximum pixel distance at which a drone can photograph
                a target, used to compute the ``in_range`` flag in the
                observation vector.
            comm_range: Maximum pixel distance at which two drones are
                considered to be within communication range, used to compute
                the ``in_comm`` flag in the observation vector.
            safe_proximity_dist: Legacy parameter retained for backward
                compatibility with proximity-penalty code paths.
            backend: Acceleration backend for geometric raycasting. Options:
                ``"auto"`` (fastest available), ``"numpy"`` (baseline),
                ``"numba"`` (LLVM JIT), or ``"cpp"`` (native C++20).
        """
        self.num_rays = num_rays
        self.max_range = float(max_range)
        self.fov_degrees = float(fov_degrees)
        self.photo_radius = float(photo_radius)
        self.comm_range = float(comm_range)
        self.safe_proximity_dist = float(safe_proximity_dist)
        self.backend = backend

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
        Casts LiDAR rays from (origin_x, origin_y) using the configured geometry backend.
        Returns (obstacle_hits, target_hits).
        """
        obs_hits: List[RayHit] = []
        tgt_hits: List[RayHit] = []

        ceil_y = float(scrolling_map.ceiling_y)
        floor_y = float(scrolling_map.floor_y)

        obstacles = getattr(scrolling_map, "obstacles", [])
        if not obstacles and hasattr(scrolling_map, "asteroids"):
            obstacles = scrolling_map.asteroids
        targets = getattr(scrolling_map, "targets", [])

        if hasattr(scrolling_map, "get_geometry_buffers"):
            obs_meta, obs_edges, obs_edge_ranges, target_data = (
                scrolling_map.get_geometry_buffers()
            )
        else:
            obs_meta, obs_edges, obs_edge_ranges, target_data = (
                extract_geometry_buffers(obstacles, targets)
            )

        backend_name = resolve_backend(self.backend)
        if backend_name == "cpp" and is_cpp_available():
            obs_dists, obs_types, obs_ends, tgt_dists, tgt_types, tgt_ends = (
                cast_rays_cpp(
                    origin_x,
                    origin_y,
                    self.angles,
                    self.max_range,
                    ceil_y,
                    floor_y,
                    obs_meta,
                    obs_edges,
                    obs_edge_ranges,
                    target_data,
                )
            )
        elif backend_name == "numba" and is_numba_available():
            obs_dists, obs_types, obs_ends, tgt_dists, tgt_types, tgt_ends = (
                cast_rays_numba(
                    origin_x,
                    origin_y,
                    self.angles,
                    self.max_range,
                    ceil_y,
                    floor_y,
                    obs_meta,
                    obs_edges,
                    obs_edge_ranges,
                    target_data,
                )
            )
        else:
            obs_dists, obs_types, obs_ends, tgt_dists, tgt_types, tgt_ends = (
                cast_rays_numpy(
                    origin_x,
                    origin_y,
                    self.angles,
                    self.max_range,
                    ceil_y,
                    floor_y,
                    obs_meta,
                    obs_edges,
                    obs_edge_ranges,
                    target_data,
                )
            )

        obs_type_labels = {0: "none", 1: "boundary", 2: "obstacle"}
        tgt_type_labels = {0: "none", 1: "target"}

        for k in range(self.num_rays):
            d = float(obs_dists[k])
            obs_hits.append(
                RayHit(
                    distance=d,
                    norm_distance=d / self.max_range,
                    hit_type=obs_type_labels.get(int(obs_types[k]), "none"),
                    endpoint=(float(obs_ends[k, 0]), float(obs_ends[k, 1])),
                )
            )

            td = float(tgt_dists[k])
            tgt_hits.append(
                RayHit(
                    distance=td,
                    norm_distance=td / self.max_range,
                    hit_type=tgt_type_labels.get(int(tgt_types[k]), "none"),
                    endpoint=(float(tgt_ends[k, 0]), float(tgt_ends[k, 1])),
                )
            )

        return obs_hits, tgt_hits

    def observe(
        self,
        origin_x: float,
        origin_y: float,
        scrolling_map: "ScrollingMap",
        other_agents: Optional[List["Agent"]] = None,
        agent_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Constructs the complete 16-dimensional observation vector:
        - 1 float: self_y / H in [0.0, 1.0]
        - 7 floats: LiDAR obstacle normalized distances in [0.0, 1.0]
        - 4 floats: Nearest target [dx/W, dy/H, dist/diag, in_range]
        - 4 floats: Nearest neighbor [dx/W, dy/H, dist/diag, proximity_alert]
        """
        w = float(scrolling_map.width)
        h = float(scrolling_map.height)
        diag = float(np.sqrt(w * w + h * h))

        # 1. Normalized altitude
        self_y_norm = float(np.clip(origin_y / h, 0.0, 1.0))

        # 2. LiDAR rays
        obs_hits, _ = self.cast_rays(origin_x, origin_y, scrolling_map)
        obs_dists = [hit.norm_distance for hit in obs_hits]

        # 3. Nearest uninspected target vector
        nearest_tgt_dist = float("inf")
        target_vec = [0.0, 0.0, 1.0, 0.0]

        for tgt in scrolling_map.targets:
            if tgt.collected:
                continue
            dx = tgt.x - origin_x
            dy = tgt.y - origin_y
            d = float(np.sqrt(dx * dx + dy * dy))
            if d < nearest_tgt_dist:
                nearest_tgt_dist = d
                in_range = 1.0 if d <= self.photo_radius else 0.0
                target_vec = [
                    float(np.clip(dx / w, -1.0, 1.0)),
                    float(np.clip(dy / h, -1.0, 1.0)),
                    float(np.clip(d / diag, 0.0, 1.0)),
                    in_range,
                ]

        # 4. Nearest living neighbor drone vector
        nearest_nbr_dist = float("inf")
        nbr_vec = [0.0, 0.0, 1.0, 0.0]

        if other_agents is not None:
            for other in other_agents:
                if not other.alive:
                    continue
                if agent_id is not None and other.agent_id == agent_id:
                    continue
                dx = other.x - origin_x
                dy = other.y - origin_y
                if abs(dx) < 1e-4 and abs(dy) < 1e-4:
                    continue
                d = float(np.sqrt(dx * dx + dy * dy))
                if d < nearest_nbr_dist:
                    nearest_nbr_dist = d
                    in_comm = 1.0 if d <= self.comm_range else 0.0
                    nbr_vec = [
                        float(np.clip(dx / w, -1.0, 1.0)),
                        float(np.clip(dy / h, -1.0, 1.0)),
                        float(np.clip(d / diag, 0.0, 1.0)),
                        in_comm,
                    ]

        obs_array = np.array(
            [self_y_norm] + obs_dists + target_vec + nbr_vec, dtype=np.float32
        )
        return obs_array
