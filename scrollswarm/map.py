"""
Map and Environment engine for ScrollSwarm (Saturn Ring Surfing).

Models an exploration corridor through Saturn's icy ring system:
- Deep cosmic void with luminous Saturnian ring-plane dust lanes.
- Upper and lower radiation sheath boundary planes (flat boundaries with safety margins).
- Dynamic, multi-scale polygon asteroids and icy moonlets executing Keplerian drift
  and Hill's epicyclic vertical oscillation:
      x(t) = x_0 - v_x * t
      y(t) = y_base + A * sin(omega * t + phase)
- Guaranteed minimal navigable passage (>= 110 px) at all times.
- Stochastic collection targets spawned in open navigation lanes.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from scrollswarm.items import Asteroid, Target


# Saturnian Space Color Palette (RGB)
COLOR_SPACE_VOID = (7, 10, 18)            # Deep cosmic midnight
COLOR_RING_HAZE = (215, 200, 170)         # Soft Saturnian champagne dust
COLOR_RING_SHEATH = (30, 40, 55)          # Boundary radiation sheath
COLOR_RING_RIM = (70, 95, 125)            # Boundary luminous cyan/slate rim
COLOR_ASTEROID_FILL = (75, 95, 120)       # Faceted icy silicate rock
COLOR_ASTEROID_OUTLINE = (30, 42, 58)     # Dark silhouette edge
COLOR_ASTEROID_SPECULAR = (150, 185, 215) # Frosted ice specular highlight
COLOR_TARGET = (255, 195, 30)             # Golden target objective

# Backward compatibility color aliases
COLOR_LAND = COLOR_SPACE_VOID
COLOR_MOUNTAIN = COLOR_RING_SHEATH
COLOR_MOUNTAIN_RIM = COLOR_RING_RIM
COLOR_TREE_LEAVES = COLOR_ASTEROID_FILL
COLOR_TREE_RIM = COLOR_ASTEROID_OUTLINE
COLOR_TREE_TRUNK = COLOR_ASTEROID_SPECULAR


@dataclass
class MapConfig:
    """Configuration parameters for the Saturn ring scrolling corridor."""
    width: int = 800
    height: int = 400
    fps: int = 30
    scroll_speed: int = 1

    # Sheath boundary parameters
    ceiling_y: int = 15
    floor_margin: int = 15
    min_corridor_gap: int = 110          # Guaranteed navigable passage (px)
    min_ridge_height: int = 15          # Backward compatibility alias for ceiling_y

    # Asteroid scale and kinematic distributions
    asteroid_min_radius: float = 14.0    # Small fast shard
    asteroid_max_radius: float = 75.0    # Massive slow moonlet
    asteroid_spawn_interval: int = 65    # Minimum steps between spawns
    target_spawn_interval: int = 85      # Minimum steps between target spawns

    preseed: bool = True                 # Whether to populate screen at initialization
    seed: Optional[int] = None


class ScrollingMap:
    """
    Manages the Saturn ring environment state, dynamic polygon asteroid lifetimes,
    Keplerian/epicyclic velocities, and guaranteed navigable clearance.
    """

    def __init__(self, config: Optional[MapConfig] = None):
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
        self.bot_heights = np.full(self.width, self.config.floor_margin, dtype=np.float32)

        # Active dynamic entities
        self.asteroids: List[Asteroid] = []
        self.targets: List[Target] = []

        # Spawning timers
        self._steps_since_asteroid = 40
        self._steps_since_target = 30

        # Pre-populate the screen with initial obstacles for step 0
        if getattr(self.config, "preseed", True):
            self._preseed_corridor()

    def reset(self, seed: Optional[int] = None) -> None:
        """Resets the map state, entities, and random number generator."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.global_step = 0
        self._steps_since_asteroid = 40
        self._steps_since_target = 30
        self.asteroids.clear()
        self.targets.clear()
        if getattr(self.config, "preseed", True):
            self._preseed_corridor()

    def _generate_asteroid_polygon(self, radius: float) -> List[Tuple[float, float]]:
        """Generates an irregular convex polygon for an icy asteroid."""
        num_vertices = self.rng.integers(5, 8)
        angles = np.sort(self.rng.uniform(0, 2 * np.pi, num_vertices))
        vertices_rel = []
        for angle in angles:
            r_jitter = radius * self.rng.uniform(0.75, 1.20)
            vx = float(r_jitter * np.cos(angle))
            vy = float(r_jitter * np.sin(angle))
            vertices_rel.append((vx, vy))
        return vertices_rel

    def _create_asteroid(self, spawn_x: float) -> Asteroid:
        """
        Creates a dynamic asteroid with size-dependent drift speed and Hill's epicyclic frequency.
        Enforces the guaranteed passage clearance invariant.
        """
        r_min = self.config.asteroid_min_radius
        r_max = self.config.asteroid_max_radius

        # Sample radius using a power-law distribution (favoring small/medium over massive moonlets)
        norm_r = self.rng.power(0.55)  # Skewed towards smaller values
        radius = float(r_min + (r_max - r_min) * (1.0 - norm_r))

        # Size-velocity scaling (smaller = faster approaching)
        size_ratio = float((r_max - radius) / (r_max - r_min))
        vx = float(1.0 + 1.8 * size_ratio)

        # Epicyclic vertical oscillation parameters (smaller = higher frequency)
        omega = float(0.02 + 0.05 * size_ratio)
        max_amplitude = float(min(30.0, 10.0 + 20.0 * size_ratio))
        if max_amplitude <= 5.0:
            vy_amp = 0.0
        else:
            vy_amp = float(self.rng.uniform(5.0, max_amplitude))
        phase = float(self.rng.uniform(0.0, 2 * np.pi))

        # Safe vertical bounds
        effective_h = radius + vy_amp
        y_min_allowed = self.ceiling_y + effective_h + 10.0
        y_max_allowed = self.floor_y - effective_h - 10.0

        if y_min_allowed >= y_max_allowed:
            base_y = float(self.height / 2.0)
            vy_amp = 0.0
        else:
            # Guarantee passage clearance: place asteroid either in upper or lower sector
            # leaving a corridor gap of at least min_corridor_gap (110 px) on the opposite side
            if self.rng.random() < 0.5:
                # Upper lane: safe passage below
                max_upper = self.floor_y - self.min_corridor_gap - effective_h
                if max_upper <= y_min_allowed:
                    base_y = float(y_min_allowed)
                else:
                    base_y = float(self.rng.uniform(y_min_allowed, max_upper))
            else:
                # Lower lane: safe passage above
                min_lower = self.ceiling_y + self.min_corridor_gap + effective_h
                if min_lower >= y_max_allowed:
                    base_y = float(y_max_allowed)
                else:
                    base_y = float(self.rng.uniform(min_lower, y_max_allowed))

        vertices_rel = self._generate_asteroid_polygon(radius)

        return Asteroid(
            x=float(spawn_x),
            y=float(base_y),
            base_y=float(base_y),
            radius=float(radius),
            vertices_rel=vertices_rel,
            vx=vx,
            vy_amplitude=vy_amp,
            omega=omega,
            phase=phase,
            step_count=0,
        )

    def _preseed_corridor(self) -> None:
        """Populates the initial viewport with starting asteroids and targets."""
        if self.width < 250 or self.height < 120:
            return

        x_positions = [self.width * 0.3, self.width * 0.55, self.width * 0.75, self.width * 0.95]
        for x in x_positions:
            self.asteroids.append(self._create_asteroid(x))

        target_positions = [self.width * 0.4, self.width * 0.65, self.width * 0.85]
        for x in target_positions:
            low_y = min(self.ceiling_y + 30.0, self.height / 2.0 - 10.0)
            high_y = max(self.floor_y - 30.0, self.height / 2.0 + 10.0)
            if high_y <= low_y:
                high_y = low_y + 10.0
            y = float(self.rng.uniform(low_y, high_y))
            self.targets.append(Target(x=x, y=y))

    def step(self) -> None:
        """Advances the simulation by 1 step: updates all asteroids, targets, and handles spawning."""
        self.global_step += 1
        self._steps_since_asteroid += 1
        self._steps_since_target += 1

        # 1. Advance dynamic asteroids
        for ast in self.asteroids:
            ast.step()

        # 2. Advance targets
        for tgt in self.targets:
            tgt.step()

        # 3. Cull offscreen entities
        self.asteroids = [ast for ast in self.asteroids if not ast.is_offscreen]
        self.targets = [tgt for tgt in self.targets if not tgt.is_offscreen]

        # 4. Stochastic spawning at right edge
        if self._steps_since_asteroid >= self.config.asteroid_spawn_interval:
            if self.rng.random() < 0.35:
                spawn_x = float(self.width + self.rng.uniform(30.0, 80.0))
                self.asteroids.append(self._create_asteroid(spawn_x))
                self._steps_since_asteroid = 0

        if self._steps_since_target >= self.config.target_spawn_interval:
            if self.rng.random() < 0.30:
                spawn_x = float(self.width + self.rng.uniform(20.0, 60.0))
                low_y = min(self.ceiling_y + 30.0, self.height / 2.0 - 10.0)
                high_y = max(self.floor_y - 30.0, self.height / 2.0 + 10.0)
                if high_y <= low_y:
                    high_y = low_y + 10.0
                y = float(self.rng.uniform(low_y, high_y))
                self.targets.append(Target(x=spawn_x, y=y))
                self._steps_since_target = 0

    def update(self) -> None:
        """Alias for step()."""
        self.step()

    @property
    def trees(self) -> List[Asteroid]:
        """Backward compatibility alias for asteroids."""
        return self.asteroids

    @trees.setter
    def trees(self, val: List[Asteroid]) -> None:
        self.asteroids = val
