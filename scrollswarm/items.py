"""
Entity definitions for ScrollSwarm (UAV Swarm Reconnaissance).
Data structures for multi-scale obstacles and inspection targets.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from scrollswarm.agent import Agent


class ObstacleType(str, Enum):
    """Categorization of urban and aerial hazards."""
    STRUCTURE = "structure"    # Ground-standing building / tower
    OVERHEAD = "overhead"      # Suspended overhead structure / cable
    FLOATING = "floating"      # Mid-air debris / airborne obstacle


@dataclass
class Obstacle:
    """
    A multi-scale 2D obstacle (ground structure, overhead barrier, or floating hazard).
    Modeled as an arbitrary convex polygon with drift velocity and optional oscillation:
        x(t) = x_0 - v_x * t
        y(t) = y_base + A * sin(omega * t + phase)
    """
    x: float
    y: float
    base_y: float = 0.0
    radius: float = 20.0
    width: float = 40.0
    height: float = 60.0
    obstacle_type: ObstacleType = ObstacleType.STRUCTURE
    vertices_rel: List[Tuple[float, float]] = field(default_factory=list)
    vx: float = 1.0
    vy_amplitude: float = 0.0
    omega: float = 0.0
    phase: float = 0.0
    step_count: int = 0

    def __post_init__(self):
        if self.base_y == 0.0:
            self.base_y = self.y

        if not self.vertices_rel:
            if self.obstacle_type == ObstacleType.STRUCTURE or self.obstacle_type == ObstacleType.OVERHEAD:
                # Rectangular structure
                hw = self.width / 2.0
                hh = self.height / 2.0
                self.vertices_rel = [
                    (-hw, -hh),
                    (hw, -hh),
                    (hw, hh),
                    (-hw, hh),
                ]
                self.radius = float(np.sqrt(hw * hw + hh * hh))
            else:
                # Regular / irregular polygon for floating debris
                angles = np.linspace(0, 2 * np.pi, 7)[:-1]
                self.vertices_rel = [
                    (float(self.radius * np.cos(a)), float(self.radius * np.sin(a)))
                    for a in angles
                ]

    def step(self) -> None:
        """Advances the obstacle's horizontal drift and vertical oscillation."""
        self.x -= self.vx
        self.step_count += 1
        if self.vy_amplitude > 0.0:
            self.y = self.base_y + self.vy_amplitude * np.sin(self.omega * self.step_count + self.phase)

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """Returns the absolute 2D coordinates of the polygon's vertices."""
        return [(self.x + dx, self.y + dy) for dx, dy in self.vertices_rel]

    @property
    def edges(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Returns the consecutive edge line segments [(v0, v1), (v1, v2), ...]."""
        verts = self.vertices
        n = len(verts)
        return [(verts[i], verts[(i + 1) % n]) for i in range(n)]

    @property
    def is_offscreen(self) -> bool:
        """Returns True if the entire obstacle has scrolled past the left viewport edge."""
        if self.vertices_rel:
            max_rel_x = max(dx for dx, _ in self.vertices_rel)
            return self.x + max_rel_x < 0.0
        return self.x + self.radius < 0.0

    def contains_point(self, px: float, py: float) -> bool:
        """Determines if point (px, py) lies inside the polygon using ray-casting."""
        verts = self.vertices
        n = len(verts)
        if n == 0:
            return False
        inside = False
        p1x, p1y = verts[0]
        for i in range(n + 1):
            p2x, p2y = verts[i % n]
            if py > min(p1y, p2y):
                if py <= max(p1y, p2y):
                    if px <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or px <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def distance_to_point(self, px: float, py: float) -> float:
        """Computes minimum Euclidean distance from (px, py) to the polygon boundary."""
        if self.contains_point(px, py):
            return 0.0

        if not self.edges:
            dx = px - self.x
            dy = py - self.y
            return float(max(0.0, np.sqrt(dx * dx + dy * dy) - self.radius))

        min_dist_sq = float("inf")
        for (ax, ay), (bx, by) in self.edges:
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


# Backward compatibility aliases
Asteroid = Obstacle


@dataclass
class Tree(Obstacle):
    """Backward compatibility alias for legacy Tree references."""
    outer_radius: int = 14
    inner_radius: int = 5

    def __post_init__(self):
        self.radius = float(self.outer_radius)
        self.obstacle_type = ObstacleType.FLOATING
        super().__post_init__()


@dataclass
class Target:
    """
    An inspection target entity requiring aerial photography within an inspection radius.
    Action uncertainty is modeled via stochastic capture probability:
        P_j = (1 - d_j / D_photo) * sensor_reliability
        P_team = 1 - Prod_j (1 - P_j)
    """
    x: float
    y: float
    size: int = 8
    inspection_radius: float = 45.0
    sensor_reliability: float = 0.85
    collected: bool = False
    vx: float = 1.0

    def step(self) -> None:
        """Drifts with the base aerial corridor scroll."""
        self.x -= self.vx

    @property
    def is_offscreen(self) -> bool:
        """Returns True if the target has scrolled off the left edge."""
        return self.x + self.size < 0.0

    def sample_inspection(
        self,
        drones: List["Agent"],
        rng: np.random.Generator,
    ) -> bool:
        """
        Evaluates stochastic photo acquisition by nearby living drones within inspection radius.
        Returns True if a clear photograph is successfully captured.
        """
        if self.collected:
            return False

        # Gather nearby living drones
        inspecting_drones = []
        for drone in drones:
            if not drone.alive:
                continue
            dx = drone.x - self.x
            dy = drone.y - self.y
            dist = np.sqrt(dx * dx + dy * dy)
            if dist <= self.inspection_radius:
                inspecting_drones.append(dist)

        if not inspecting_drones:
            return False

        # Probability of team success under action uncertainty
        p_fail_team = 1.0
        for dist in inspecting_drones:
            # Proximity factor: closer drones have higher capture probability
            p_drone = (1.0 - (dist / self.inspection_radius)) * self.sensor_reliability
            p_drone = float(np.clip(p_drone, 0.05, 0.95))
            p_fail_team *= (1.0 - p_drone)

        p_success_team = 1.0 - p_fail_team
        if rng.random() < p_success_team:
            self.collected = True
            return True

        return False
