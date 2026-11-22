"""
Agent entity for ScrollSwarm.

Agents occupy fixed horizontal columns and navigate vertically (UP, STAY, DOWN)
with step size k = 5 pixels. Collision with mountain boundaries or trees causes
instant termination (alive = False).
"""

from enum import IntEnum
from typing import Any, List, Optional, Tuple
import numpy as np

from scrollswarm.items import Asteroid, Tree, Target
from scrollswarm.map import ScrollingMap
from scrollswarm.sensors import RaySensor


class Action(IntEnum):
    """Discrete vertical movement actions."""
    UP = 0
    STAY = 1
    DOWN = 2


# Color palette for up to 10 distinct agents (RGB)
AGENT_PALETTE: List[Tuple[int, int, int]] = [
    (30, 144, 255),   # Dodger Blue
    (255, 127, 80),   # Coral
    (147, 112, 219),  # Medium Purple
    (255, 215, 0),    # Gold
    (0, 206, 209),    # Dark Turquoise
    (255, 105, 180),  # Hot Pink
    (50, 205, 50),    # Lime Green
    (255, 165, 0),    # Orange
    (70, 130, 180),   # Steel Blue
    (218, 112, 214),  # Orchid
]

COLOR_AGENT_DEFAULT = AGENT_PALETTE[0]
COLOR_AGENT_DEAD = (100, 100, 100)


class Agent:
    """
    Represents an agent in ScrollSwarm.
    Geometry: A triangle pointing RIGHT (away from left edge, toward the field).
    """

    def __init__(
        self,
        agent_id: int,
        x: float,
        y: float,
        step_size: int = 5,
        width: float = 16.0,
        height: float = 14.0,
        color: Optional[Tuple[int, int, int]] = None,
    ):
        self.agent_id = agent_id
        self.x = float(x)
        self.y = float(y)
        self.step_size = step_size
        self.width = float(width)
        self.height = float(height)
        self.color = color if color is not None else AGENT_PALETTE[agent_id % len(AGENT_PALETTE)]

        # Operational state
        self.alive: bool = True
        self.score: int = 0
        self.steps_alive: int = 0

        # Collision radius (circumscribed circle approximation for entity checks)
        self.collision_radius: float = max(self.width, self.height) / 2.0

        # Perception sensor
        self.sensor: Optional[RaySensor] = None

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """
        Calculates the 3 vertices of the triangle pointing RIGHT:
        - Tip: (x + width / 2, y)
        - Top-Left: (x - width / 2, y - height / 2)
        - Bottom-Left: (x - width / 2, y + height / 2)
        """
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        tip = (self.x + half_w, self.y)
        top_left = (self.x - half_w, self.y - half_h)
        bot_left = (self.x - half_w, self.y + half_h)
        return [tip, top_left, bot_left]

    def step(self, action: int, map_height: int) -> None:
        """
        Executes vertical movement:
        - UP: y decreases by step_size (k = 5 px)
        - DOWN: y increases by step_size (k = 5 px)
        - STAY: y remains unchanged
        """
        if not self.alive:
            return

        self.steps_alive += 1

        if action == Action.UP:
            self.y -= self.step_size
        elif action == Action.DOWN:
            self.y += self.step_size

        # Clamp within the vertical bounds of the screen
        half_h = self.height / 2.0
        self.y = float(np.clip(self.y, half_h, map_height - half_h))

    def check_collision_mountain(self, scrolling_map: ScrollingMap) -> bool:
        """
        Checks if any vertex of the agent penetrates the top or bottom mountain ridges.
        Instant termination occurs on collision.
        """
        if not self.alive:
            return False

        for vx, vy in self.vertices:
            col = int(np.clip(np.round(vx), 0, scrolling_map.width - 1))
            top_bound = scrolling_map.top_heights[col]
            bot_bound = scrolling_map.height - scrolling_map.bot_heights[col]

            # Inside top mountain or inside bottom mountain
            if vy <= top_bound or vy >= bot_bound:
                self.alive = False
                return True

        return False

    def check_collision_obstacles(self, obstacles: List[Any]) -> bool:
        """
        Checks collision with dynamic polygon asteroids (or circular obstacles).
        Instant termination occurs on contact.
        """
        if not self.alive:
            return False

        for obs in obstacles:
            if hasattr(obs, "distance_to_point"):
                dist = obs.distance_to_point(self.x, self.y)
                if dist <= self.collision_radius or obs.contains_point(self.x, self.y):
                    self.alive = False
                    return True
            elif hasattr(obs, "outer_radius"):
                dx = self.x - obs.x
                dy = self.y - obs.y
                threshold = self.collision_radius + obs.outer_radius
                if dx * dx + dy * dy <= threshold * threshold:
                    self.alive = False
                    return True

        return False

    def check_collision_trees(self, trees: List[Any]) -> bool:
        """Backward compatibility alias for check_collision_obstacles."""
        return self.check_collision_obstacles(trees)

    def check_target_collection(self, targets: List[Target]) -> int:
        """
        Checks if the agent overlaps with any uncollected target.
        Increments score and marks target as collected.
        Returns number of targets collected in this check.
        """
        if not self.alive:
            return 0

        collected_count = 0
        for target in targets:
            if target.collected:
                continue

            dx = self.x - target.x
            dy = self.y - target.y
            dist_sq = dx * dx + dy * dy
            threshold = self.collision_radius + target.size
            if dist_sq <= threshold * threshold:
                target.collected = True
                self.score += 1
                collected_count += 1

        return collected_count

    def reset(self, initial_y: float) -> None:
        """Resets the agent to an initial vertical position and alive state."""
        self.y = float(initial_y)
        self.alive = True
        self.score = 0
        self.steps_alive = 0

    def get_observation(self, scrolling_map: ScrollingMap) -> np.ndarray:
        """
        Computes the 15-dimensional continuous observation vector:
        [self_y / H, 7 obstacle ray distances, 7 target ray distances].
        """
        if self.sensor is None:
            self.sensor = RaySensor(max_range=float(scrolling_map.width) / 2.0)
        return self.sensor.observe(self.x, self.y, scrolling_map)


def create_agents(
    num_agents: int = 5,
    map_width: int = 800,
    scrolling_map: Optional[ScrollingMap] = None,
    step_size: int = 5,
) -> List[Agent]:
    """
    Creates a team of N agents (1 <= N <= 10, default 5).
    Agents occupy fixed, evenly spaced horizontal columns on the left portion of the map.
    """
    if not (1 <= num_agents <= 10):
        raise ValueError(f"num_agents must be between 1 and 10, got {num_agents}")

    agents: List[Agent] = []
    x_min = 80.0
    x_max = min(320.0, float(map_width) * 0.45)

    for i in range(num_agents):
        if num_agents == 1:
            x_pos = (x_min + x_max) / 2.0
        else:
            x_pos = x_min + i * (x_max - x_min) / (num_agents - 1)

        # Center agent vertically in corridor if scrolling_map is available
        if scrolling_map is not None:
            col = int(np.clip(np.round(x_pos), 0, scrolling_map.width - 1))
            top_bound = scrolling_map.top_heights[col]
            bot_bound = scrolling_map.height - scrolling_map.bot_heights[col]
            initial_y = (top_bound + bot_bound) / 2.0
        else:
            initial_y = 200.0

        color = AGENT_PALETTE[i % len(AGENT_PALETTE)]
        agents.append(
            Agent(
                agent_id=i,
                x=x_pos,
                y=initial_y,
                step_size=step_size,
                color=color,
            )
        )

    return agents
