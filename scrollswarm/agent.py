"""
Agent entity for ScrollSwarm (UAV Swarm Reconnaissance).

UAV quadcopters occupy horizontal columns and navigate vertically (UP, STAY, DOWN)
with step size k = 5 pixels.
Obstacle or boundary collision causes structural termination (alive = False).
"""

from enum import IntEnum
from typing import Any, List, Optional, Tuple
import numpy as np

from scrollswarm.items import Obstacle, Target, Asteroid, Tree
from scrollswarm.map import ScrollingMap
from scrollswarm.sensors import RaySensor


class Action(IntEnum):
    """
    Discrete vertical movement actions for a ScrollSwarm UAV.

    Attributes:
        UP: Climb — decreases ``y`` by ``step_size`` pixels (index ``0``).
        STAY: Hold altitude — ``y`` unchanged (index ``1``).
        DOWN: Dive — increases ``y`` by ``step_size`` pixels (index ``2``).
    """
    UP = 0
    STAY = 1
    DOWN = 2


# Color palette for up to 10 distinct UAVs (RGB)
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
    Autonomous survey UAV (quadcopter) entity in ScrollSwarm.

    Each agent occupies a fixed horizontal column and navigates vertically
    (UP / STAY / DOWN) with a configurable step size.  The agent is represented
    as a forward-facing triangle for collision geometry and rendered as a
    quadcopter sprite in ``render_mode="human"``.

    Attributes:
        agent_id: Zero-based integer index of this drone in the swarm.
        x: Fixed horizontal position in pixels (column station-keeping).
        y: Current vertical position in pixels (mutable each step).
        step_size: Vertical displacement per UP/DOWN action in pixels.
        width: Horizontal collision bounding box extent in pixels.
        height: Vertical collision bounding box extent in pixels.
        color: RGB tuple used for rendering this drone.
        alive: ``False`` after any boundary or obstacle collision.
        score: Cumulative count of targets successfully photographed.
        steps_alive: Number of steps the drone has survived.
        collision_radius: Effective radius for obstacle proximity checks.
    """

    def __init__(
        self,
        agent_id: int,
        x: float,
        y: float,
        step_size: int = 5,
        width: float = 20.0,
        height: float = 16.0,
        color: Optional[Tuple[int, int, int]] = None,
        safe_proximity_dist: float = 28.0,
    ):
        """
        Initializes the UAV agent.

        Args:
            agent_id: Zero-based drone index; also selects the color palette
                entry and is used to build the PettingZoo agent ID string.
            x: Initial horizontal position in pixels (fixed for the episode).
            y: Initial vertical position in pixels (dynamic).
            step_size: Vertical displacement per UP / DOWN action in pixels.
                Valid range: [1, 20].
            width: Physical horizontal bounding extent of the quadcopter body
                in pixels (used for collision triangle geometry).
            height: Physical vertical bounding extent in pixels.
            color: RGB tuple override.  Defaults to the palette entry for
                ``agent_id``.
            safe_proximity_dist: Minimum safe separation distance from other
                drones in pixels (retained for legacy proximity checks).
        """
        self.agent_id = agent_id
        self.x = float(x)
        self.y = float(y)
        self.step_size = step_size
        self.width = float(width)
        self.height = float(height)
        self.color = color if color is not None else AGENT_PALETTE[agent_id % len(AGENT_PALETTE)]
        self.safe_proximity_dist = float(safe_proximity_dist)

        # Operational state
        self.alive: bool = True
        self.score: int = 0
        self.steps_alive: int = 0

        # Collision radius for physical obstacle/boundary contact
        self.collision_radius: float = max(self.width, self.height) / 2.0

        # Perception sensor
        self.sensor: Optional[RaySensor] = None

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """
        Calculates the 3 vertices of the quadcopter facing forward (RIGHT):

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

        # Clamp within screen bounds
        half_h = self.height / 2.0
        self.y = float(np.clip(self.y, half_h, map_height - half_h))

    def check_collision_boundary(self, scrolling_map: ScrollingMap) -> bool:
        """
        Checks if any vertex of the UAV penetrates the upper ceiling or ground boundaries.
        Instant termination occurs on collision.
        """
        if not self.alive:
            return False

        for vx, vy in self.vertices:
            col = int(np.clip(np.round(vx), 0, scrolling_map.width - 1))
            top_bound = scrolling_map.top_heights[col]
            bot_bound = scrolling_map.height - scrolling_map.bot_heights[col]

            if vy <= top_bound or vy >= bot_bound:
                self.alive = False
                return True

        return False

    def check_collision_mountain(self, scrolling_map: ScrollingMap) -> bool:
        """Backward compatibility alias for check_collision_boundary."""
        return self.check_collision_boundary(scrolling_map)

    def check_collision_obstacles(self, obstacles: List[Any]) -> bool:
        """
        Checks physical collision with obstacles (buildings, overhead towers, floating debris).
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

    def compute_proximity_penalty(
        self,
        other_agents: List["Agent"],
        safe_dist: Optional[float] = None,
        penalty_coef: float = 0.5,
    ) -> float:
        """Backward compatibility stub (proximity penalty disabled in this scenario)."""
        return 0.0

    def check_target_inspection(
        self,
        targets: List[Target],
        rng: Optional[np.random.Generator] = None,
        team_drones: Optional[List["Agent"]] = None,
    ) -> int:
        """
        Inspects targets within photo radius using stochastic action uncertainty.
        Increments score and marks target collected upon successful photo.
        Returns number of targets successfully photographed.
        """
        if not self.alive:
            return 0

        generator = rng if rng is not None else np.random.default_rng()
        drones_pool = team_drones if team_drones is not None else [self]

        collected_count = 0
        for target in targets:
            if target.collected:
                continue

            dx = self.x - target.x
            dy = self.y - target.y
            dist = np.sqrt(dx * dx + dy * dy)

            # Check if this UAV is within inspection range
            if dist <= target.inspection_radius:
                success = target.sample_inspection(drones_pool, generator)
                if success:
                    self.score += 1
                    collected_count += 1

        return collected_count

    def check_target_collection(self, targets: List[Target]) -> int:
        """Backward compatibility alias for deterministic / direct target pickup."""
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
        """Resets the UAV to an initial vertical position and alive state."""
        self.y = float(initial_y)
        self.alive = True
        self.score = 0
        self.steps_alive = 0

    def get_observation(
        self,
        scrolling_map: ScrollingMap,
        other_agents: Optional[List["Agent"]] = None,
    ) -> np.ndarray:
        """
        Computes the 16-dimensional continuous observation vector:
        [self_y / H, 7 obstacle ray distances, 4 target features, 4 neighbor features].
        """
        if self.sensor is None:
            self.sensor = RaySensor(
                max_range=float(scrolling_map.width) / 2.0,
                safe_proximity_dist=self.safe_proximity_dist,
            )
        return self.sensor.observe(
            origin_x=self.x,
            origin_y=self.y,
            scrolling_map=scrolling_map,
            other_agents=other_agents,
            agent_id=self.agent_id,
        )


def create_agents(
    num_agents: int = 5,
    map_width: int = 800,
    scrolling_map: Optional[ScrollingMap] = None,
    step_size: int = 5,
) -> List[Agent]:
    """
    Creates a team of N UAVs (1 <= N <= 10, default 5).
    UAVs occupy fixed, evenly spaced horizontal columns on the left portion of the corridor.
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

        # Center agent vertically in corridor
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
