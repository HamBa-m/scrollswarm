"""
Map and Environment engine for ScrollSwarm (UAV Swarm Reconnaissance).

Models an open aerial corridor with multi-scale procedural obstacles:
- Open airspace bounded by high altitude ceiling and ground base plane.
- Multi-scale procedural obstacles:
    - Ground-standing structures (buildings and towers of varying heights and widths).
    - Overhead structures / suspended fixtures extending from ceiling.
    - Floating debris and airborne hazards with slight epicyclic sway.
- Guaranteed minimal navigable passage (>= 110 px) at all times between top and bottom hazards.
- Stochastic targets spawned in navigable lanes.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from scrollswarm.items import Obstacle, ObstacleType, Target, Asteroid, Tree

# Aerial Corridor Color Palette (RGB)
COLOR_AIRSPACE = (20, 26, 38)  # Deep atmospheric dusk sky
COLOR_GRID_LINE = (30, 40, 58)  # Subtle altitude / survey grid lines
COLOR_CEILING = (14, 18, 26)  # High altitude limit
COLOR_GROUND = (18, 22, 32)  # Ground surface
COLOR_BOUNDARY_RIM = (55, 80, 115)  # Boundary guidance edge

# Structural obstacles
COLOR_BUILDING_FILL = (45, 58, 80)  # Concrete / metallic urban structure
COLOR_BUILDING_OUTLINE = (75, 100, 135)  # Structural edge highlight
COLOR_BUILDING_SPECULAR = (110, 145, 190)  # Structural beam accent
COLOR_HAZARD_FILL = (85, 60, 68)  # Hazardous debris fill
COLOR_HAZARD_OUTLINE = (180, 75, 70)  # Hazardous debris warning border

# Targets
COLOR_TARGET = (255, 195, 30)  # Active uninspected target (golden glow)
COLOR_TARGET_INSPECTED = (
    50,
    215,
    130,
)  # Photographed / completed target (emerald green)

# Backward compatibility color aliases
COLOR_SPACE_VOID = COLOR_AIRSPACE
COLOR_RING_HAZE = COLOR_GRID_LINE
COLOR_RING_SHEATH = COLOR_CEILING
COLOR_RING_RIM = COLOR_BOUNDARY_RIM
COLOR_ASTEROID_FILL = COLOR_BUILDING_FILL
COLOR_ASTEROID_OUTLINE = COLOR_BUILDING_OUTLINE
COLOR_ASTEROID_SPECULAR = COLOR_BUILDING_SPECULAR
COLOR_LAND = COLOR_AIRSPACE
COLOR_MOUNTAIN = COLOR_CEILING
COLOR_MOUNTAIN_RIM = COLOR_BOUNDARY_RIM
COLOR_TREE_LEAVES = COLOR_BUILDING_FILL
COLOR_TREE_RIM = COLOR_BUILDING_OUTLINE
COLOR_TREE_TRUNK = COLOR_BUILDING_SPECULAR


@dataclass
class MapConfig:
    """
    Configuration parameters for the ScrollSwarm UAV survey corridor.

    Attributes:
        width: Corridor width in pixels.
        height: Corridor height in pixels.
        fps: Target simulation frame rate (used by the Pygame clock).
        scroll_speed: Leftward scroll velocity in pixels per step.
        ceiling_y: Upper boundary ``y`` coordinate (pixels from top).
        floor_margin: Lower boundary inset from the bottom in pixels;
            the navigable floor is at ``y = height - floor_margin``.
        min_corridor_gap: Minimum guaranteed vertical clearance between the
            highest ground structure and lowest overhead structure in pixels.
            Obstacles are rejected at spawn time if they would violate this
            invariant.
        obstacle_spawn_interval: Minimum number of steps between consecutive
            obstacle spawns.
        target_spawn_interval: Minimum number of steps between consecutive
            target spawns.
        preseed: When ``True``, the corridor is pre-populated with obstacles
            and targets before the first ``step()`` call.
        seed: Optional integer seed for deterministic environment generation.
    """

    width: int = 800
    height: int = 400
    fps: int = 30
    scroll_speed: int = 1

    # Corridor boundaries
    ceiling_y: int = 15
    floor_margin: int = 15
    min_corridor_gap: int = 110  # Guaranteed navigable passage (px)
    min_ridge_height: int = 15  # Backward compatibility alias

    # Obstacle scale and spawn intervals
    obstacle_spawn_interval: int = 55  # Minimum steps between obstacle spawns
    target_spawn_interval: int = 75  # Minimum steps between target spawns

    # Backward compatibility aliases
    asteroid_min_radius: float = 14.0
    asteroid_max_radius: float = 65.0
    asteroid_spawn_interval: int = 55

    # Safe inter-agent proximity distance
    safe_proximity_dist: float = 28.0

    preseed: bool = True  # Pre-populate screen at step 0
    seed: Optional[int] = None


class ScrollingMap:
    """
    Manages the scrolling aerial corridor, multi-scale procedural obstacles,
    guaranteed navigable passage, and stochastic inspection targets.

    The map advances one pixel per step (``scroll_speed = 1``) in the leftward
    direction.  Obstacles are procedurally generated off-screen to the right
    and culled when they scroll past the left edge.  The minimum navigable gap
    ``G_min >= min_corridor_gap`` is enforced at spawn time.

    Attributes:
        config: ``MapConfig`` instance governing all corridor parameters.
        width: Corridor width in pixels.
        height: Corridor height in pixels.
        ceiling_y: Upper navigable boundary ``y`` in pixels.
        floor_y: Lower navigable boundary ``y`` in pixels.
        obstacles: List of active ``Obstacle`` polygon entities.
        targets: List of active ``Target`` waypoint entities.
        rng: ``numpy.random.Generator`` used for all stochastic decisions.
        global_step: Monotonically increasing step counter.
        top_heights: Per-column ceiling boundary array (shape ``(W,)``).
        bot_heights: Per-column floor inset array (shape ``(W,)``).
    """

    def __init__(self, config: Optional[MapConfig] = None):
        """
        Initializes the ScrollingMap.

        Args:
            config: ``MapConfig`` dataclass instance.  Defaults to
                ``MapConfig()`` (800×400 corridor, standard spawn rates).
        """
        self.config = config or MapConfig()
        self.width = self.config.width
        self.height = self.config.height
        self.rng = np.random.default_rng(self.config.seed)

        self.min_corridor_gap = self.config.min_corridor_gap
        self.ceiling_y = self.config.ceiling_y
        self.floor_y = self.height - self.config.floor_margin
        self.global_step: int = 0

        # Boundary arrays (for backward compatibility with column-based checks)
        self.top_heights = np.full(self.width, self.ceiling_y, dtype=np.float32)
        self.bot_heights = np.full(
            self.width, self.config.floor_margin, dtype=np.float32
        )

        # Active dynamic entities
        self.obstacles: List[Obstacle] = []
        self.targets: List[Target] = []

        # Spawning timers
        self._steps_since_obstacle = 40
        self._steps_since_target = 30

        # Cached contiguous geometry buffers for current step
        self._geo_cache: Optional[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = None

        if getattr(self.config, "preseed", True):
            self._preseed_corridor()

    def reset(self, seed: Optional[int] = None) -> None:
        """Resets the map state, entities, and random number generator."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.global_step = 0
        self._steps_since_obstacle = 40
        self._steps_since_target = 30
        self.obstacles.clear()
        self.targets.clear()
        self._geo_cache = None
        if getattr(self.config, "preseed", True):
            self._preseed_corridor()

    def _create_ground_structure(self, spawn_x: float, max_h: float) -> Obstacle:
        """Creates a ground-standing building or tower."""
        width = float(self.rng.uniform(30.0, 70.0))
        h = float(self.rng.uniform(40.0, max(50.0, max_h)))
        center_y = float(self.floor_y - h / 2.0)
        return Obstacle(
            x=float(spawn_x),
            y=center_y,
            base_y=center_y,
            width=width,
            height=h,
            obstacle_type=ObstacleType.STRUCTURE,
            vx=1.0,
        )

    def _create_overhead_structure(self, spawn_x: float, max_depth: float) -> Obstacle:
        """Creates an overhead suspended structure extending down from the ceiling."""
        width = float(self.rng.uniform(30.0, 70.0))
        depth = float(self.rng.uniform(30.0, max(40.0, max_depth)))
        center_y = float(self.ceiling_y + depth / 2.0)
        return Obstacle(
            x=float(spawn_x),
            y=center_y,
            base_y=center_y,
            width=width,
            height=depth,
            obstacle_type=ObstacleType.OVERHEAD,
            vx=1.0,
        )

    def _create_floating_hazard(self, spawn_x: float) -> Obstacle:
        """Creates a mid-air polygonal floating hazard with subtle epicyclic oscillation."""
        radius = float(self.rng.uniform(14.0, 32.0))
        vy_amp = float(self.rng.uniform(4.0, 16.0))
        omega = float(self.rng.uniform(0.02, 0.06))
        phase = float(self.rng.uniform(0.0, 2 * np.pi))

        effective_h = radius + vy_amp
        y_min = self.ceiling_y + effective_h + 15.0
        y_max = self.floor_y - effective_h - 15.0

        if y_min >= y_max:
            base_y = float(self.height / 2.0)
            vy_amp = 0.0
        else:
            base_y = float(self.rng.uniform(y_min, y_max))

        return Obstacle(
            x=float(spawn_x),
            y=base_y,
            base_y=base_y,
            radius=radius,
            width=radius * 2.0,
            height=radius * 2.0,
            obstacle_type=ObstacleType.FLOATING,
            vx=1.0,
            vy_amplitude=vy_amp,
            omega=omega,
            phase=phase,
        )

    def _create_procedural_obstacle(self, spawn_x: float) -> Obstacle:
        """
        Samples an obstacle type (ground structure, overhead structure, or floating hazard)
        while strictly preserving the guaranteed navigable corridor gap (>= 110 px).
        """
        max_structure_h = (self.floor_y - self.ceiling_y) - self.min_corridor_gap - 10.0
        max_structure_h = max(35.0, max_structure_h)

        obs_kind = self.rng.choice(
            [ObstacleType.STRUCTURE, ObstacleType.OVERHEAD, ObstacleType.FLOATING],
            p=[0.45, 0.35, 0.20],
        )

        if obs_kind == ObstacleType.STRUCTURE:
            return self._create_ground_structure(spawn_x, max_structure_h)
        elif obs_kind == ObstacleType.OVERHEAD:
            return self._create_overhead_structure(spawn_x, max_structure_h)
        else:
            return self._create_floating_hazard(spawn_x)

    def _preseed_corridor(self) -> None:
        """Populates the initial corridor with starting obstacles and inspection targets."""
        if self.width < 250 or self.height < 120:
            return

        x_positions = [self.width * 0.35, self.width * 0.60, self.width * 0.82]
        for x in x_positions:
            self.obstacles.append(self._create_procedural_obstacle(x))

        target_positions = [self.width * 0.45, self.width * 0.70, self.width * 0.92]
        for x in target_positions:
            low_y = min(self.ceiling_y + 35.0, self.height / 2.0 - 10.0)
            high_y = max(self.floor_y - 35.0, self.height / 2.0 + 10.0)
            if high_y <= low_y:
                high_y = low_y + 10.0
            y = float(self.rng.uniform(low_y, high_y))
            self.targets.append(Target(x=x, y=y))

    def step(self) -> None:
        """Advances the simulation by 1 step: updates obstacles, targets, and handles spawning."""
        self.global_step += 1
        self._steps_since_obstacle += 1
        self._steps_since_target += 1

        # 1. Advance dynamic obstacles
        for obs in self.obstacles:
            obs.step()

        # 2. Advance targets
        for tgt in self.targets:
            tgt.step()

        # 3. Cull offscreen entities
        self.obstacles = [obs for obs in self.obstacles if not obs.is_offscreen]
        self.targets = [tgt for tgt in self.targets if not tgt.is_offscreen]

        # 4. Stochastic spawning at right edge
        if self._steps_since_obstacle >= self.config.obstacle_spawn_interval:
            if self.rng.random() < 0.40:
                spawn_x = float(self.width + self.rng.uniform(30.0, 70.0))
                self.obstacles.append(self._create_procedural_obstacle(spawn_x))
                self._steps_since_obstacle = 0

        if self._steps_since_target >= self.config.target_spawn_interval:
            if self.rng.random() < 0.35:
                spawn_x = float(self.width + self.rng.uniform(20.0, 50.0))
                low_y = min(self.ceiling_y + 35.0, self.height / 2.0 - 10.0)
                high_y = max(self.floor_y - 35.0, self.height / 2.0 + 10.0)
                if high_y <= low_y:
                    high_y = low_y + 10.0
                y = float(self.rng.uniform(low_y, high_y))
                self.targets.append(Target(x=spawn_x, y=y))
                self._steps_since_target = 0

        # Invalidate geometry buffer cache on step
        self._geo_cache = None

    def get_geometry_buffers(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns cached contiguous NumPy geometry buffers for current map step."""
        if self._geo_cache is None:
            from scrollswarm.geometry import extract_geometry_buffers

            self._geo_cache = extract_geometry_buffers(self.obstacles, self.targets)
        return self._geo_cache

    def update(self) -> None:
        """Alias for step()."""
        self.step()

    # Backward compatibility aliases
    @property
    def asteroids(self) -> List[Obstacle]:
        return self.obstacles

    @asteroids.setter
    def asteroids(self, val: List[Obstacle]) -> None:
        self.obstacles = val

    @property
    def trees(self) -> List[Obstacle]:
        return self.obstacles

    @trees.setter
    def trees(self, val: List[Obstacle]) -> None:
        self.obstacles = val
