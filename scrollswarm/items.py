"""
Entity definitions for ScrollSwarm (Saturn Ring Surfing).
Data structures for dynamic asteroids, ring debris, and telemetry target beacons.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class Asteroid:
    """
    A dynamic icy asteroid / moonlet in Saturn's rings.
    Modeled as a convex polygon executing Keplerian drift and Hill's epicyclic vertical oscillation:
        x(t) = x_0 - v_x * t
        y(t) = y_base + A * sin(omega * t + phase)
    """
    x: float
    y: float
    base_y: float = 0.0
    radius: float = 20.0
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
            angles = np.linspace(0, 2 * np.pi, 7)[:-1]
            self.vertices_rel = [
                (float(self.radius * np.cos(a)), float(self.radius * np.sin(a)))
                for a in angles
            ]

    def step(self) -> None:
        """Advances the asteroid's horizontal drift and vertical epicyclic oscillation."""
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
        """Returns True if the entire asteroid has scrolled past the left viewport edge."""
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


# Backward compatibility alias
@dataclass
class Tree(Asteroid):
    """Backward compatibility alias for legacy Tree references."""
    outer_radius: int = 14
    inner_radius: int = 5

    def __post_init__(self):
        if not self.vertices_rel:
            # Approximate circular tree as a regular octagon
            angles = np.linspace(0, 2 * np.pi, 9)[:-1]
            self.vertices_rel = [
                (float(self.outer_radius * np.cos(a)), float(self.outer_radius * np.sin(a)))
                for a in angles
            ]
            self.radius = float(self.outer_radius)
            self.base_y = self.y


@dataclass
class Target:
    """A collection target: scientific telemetry beacon or volatile ice sample."""
    x: float
    y: float
    size: int = 8
    collected: bool = False
    vx: float = 1.0

    def step(self) -> None:
        """Drifts with the base ring scroll."""
        self.x -= self.vx

    @property
    def is_offscreen(self) -> bool:
        """Returns True if the target has scrolled past the left edge."""
        return self.x + self.size < 0.0
