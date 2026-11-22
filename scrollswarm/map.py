"""
Map and Topography engine for ScrollSwarm.

Models a 2:1 side-scrolling environment:
- Light green ground plain.
- Multi-scale undulating mountain ridges (meandering centerlines, dynamic chokepoints,
  and rocky crags) providing tactical vertical navigation challenges.
- Guaranteed minimal passage clearance at EVERY column of the screen.
- Scrolling discrete step of 1 pixel per timestep.
- Stochastic tree and target spawning within the navigable corridor.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from scrollswarm.items import Tree, Target


# Enhanced Classic Color Palette (RGB) - Optimized for Contrast & Visibility
COLOR_LAND = (195, 230, 185)         # Soft pastel meadow
COLOR_MOUNTAIN = (65, 75, 85)        # Deep slate rock
COLOR_MOUNTAIN_RIM = (35, 42, 50)    # Dark slate contour rim
COLOR_TREE_LEAVES = (20, 120, 45)    # Crisp forest green
COLOR_TREE_RIM = (10, 60, 25)        # Dark canopy outline
COLOR_TREE_TRUNK = (130, 60, 20)     # Rich cedar core
COLOR_TARGET = (255, 30, 60)         # High-visibility vibrant scarlet


@dataclass
class RockCrag:
    """A protruding rock crag (stalactite from ceiling or stalagmite from floor)."""
    peak_x: float          # x-coordinate of peak center
    width: float           # total horizontal base width
    height: float          # inward peak protrusion
    side: str              # "top" or "bottom"

    def get_protrusion_at(self, x: float) -> float:
        """Computes inward protrusion height at coordinate x using smooth cosine bell."""
        half_w = self.width / 2.0
        dist = abs(x - self.peak_x)
        if dist >= half_w:
            return 0.0
        return float(self.height * (0.5 * (1.0 + np.cos(np.pi * dist / half_w))))


@dataclass
class MapConfig:
    """Configuration parameters for the scrolling map."""
    width: int = 800
    height: int = 400
    fps: int = 30
    scroll_speed: int = 1          # 1 pixel per timestep

    # Multi-Scale Meandering Topography Parameters
    # Centerline waves: C(t) = H/2 + A1 * sin(2pi*t/L1 + p1) + A2 * sin(2pi*t/L2 + p2)
    macro_wave_len: float = 540.0       # Macro sweeping climb/dive wavelength
    macro_wave_amp: float = 65.0        # Macro wave amplitude
    meso_wave_len: float = 190.0        # Medium S-chicane wavelength
    meso_wave_amp: float = 28.0         # Medium wave amplitude

    # Corridor Width Dynamics: G(t) = base_gap + gap_amp * sin(2pi*t/gap_len + pG)
    base_corridor_gap: float = 175.0    # Mean corridor width
    gap_wave_amp: float = 35.0          # Chokepoint modulation amplitude
    gap_wave_len: float = 380.0         # Chokepoint frequency

    # Minimum safety clearance: GUARANTEED minimal open passage for agents at EVERY column
    min_corridor_gap: int = 110         # Agents (height=14px) always have safe passage >= 110px
    min_ridge_height: int = 20          # Minimum mountain thickness

    # Rocky Outcrop Crags (occasional stalactites / stalagmites)
    crag_probability: float = 0.006     # Probability of starting a new crag at incoming edge
    crag_width_range: Tuple[int, int] = (60, 110)
    crag_height_range: Tuple[int, int] = (25, 45)

    # Micro rocky roughness
    micro_roughness_sigma: float = 1.0

    # Spawning probabilities per column arrival
    p_tree: float = 0.015
    p_target: float = 0.012

    # Entity geometry
    tree_radius_range: Tuple[int, int] = (10, 15)
    target_size: int = 8

    seed: Optional[int] = None


class ScrollingMap:
    """
    Manages the 2D world state, right-to-left scrolling, multi-scale topography,
    guaranteed minimal corridor clearance, and stochastic entity lifecycles.
    """

    def __init__(self, config: Optional[MapConfig] = None):
        self.config = config or MapConfig()
        self.width = self.config.width
        self.height = self.config.height
        self.rng = np.random.default_rng(self.config.seed)

        # Enforce valid minimal gap for the chosen height
        self.min_corridor_gap = min(
            self.config.min_corridor_gap,
            max(20, self.height - 2 * self.config.min_ridge_height)
        )

        # Global distance counter (maintains continuous phase during scrolling)
        self.global_step: int = 0

        # Random phase offsets for procedural wave synthesis
        self.phase_macro: float = 0.0
        self.phase_meso: float = 0.0
        self.phase_gap: float = 0.0

        # Topography profile: 1D arrays for top and bottom mountain ridge heights
        # top_heights[x] is the thickness from y=0 down to y=top_heights[x]
        # bot_heights[x] is the thickness from y=H-1 up to y=H - bot_heights[x]
        self.top_heights = np.zeros(self.width, dtype=np.float32)
        self.bot_heights = np.zeros(self.width, dtype=np.float32)

        # Active protruding crags and items
        self.crags: List[RockCrag] = []
        self.trees: List[Tree] = []
        self.targets: List[Target] = []

        self.reset(self.config.seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Resets the map state, generating initial undulating topography across the full width."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.global_step = 0
        self.phase_macro = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.phase_meso = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.phase_gap = float(self.rng.uniform(0.0, 2.0 * np.pi))

        self.crags.clear()
        self.trees.clear()
        self.targets.clear()

        # Generate initial screen columns x in [0, width-1]
        for x in range(self.width):
            h_top, h_bot = self._compute_column_heights(t=float(x), col_x=float(x))
            self.top_heights[x] = h_top
            self.bot_heights[x] = h_bot

        # Assert post-generation guaranteed minimal corridor clearance
        self._verify_corridor_clearance()

    def _compute_column_heights(self, t: float, col_x: float) -> Tuple[float, float]:
        """
        Synthesizes top and bottom mountain boundary heights at distance t:
        1. Centerline meander (macro + meso waves).
        2. Dynamic corridor gap width (gorges vs basins).
        3. Protruding rock crags (stalactites/stalagmites).
        4. Micro rocky roughness.
        5. STRICT GUARANTEE: Invariant gap >= min_corridor_gap is enforced at every column.
        """
        # 1. Centerline meander C(t)
        macro_wave = self.config.macro_wave_amp * np.sin(
            (2.0 * np.pi * t / self.config.macro_wave_len) + self.phase_macro
        )
        meso_wave = self.config.meso_wave_amp * np.sin(
            (2.0 * np.pi * t / self.config.meso_wave_len) + self.phase_meso
        )
        center_y = (self.height / 2.0) + macro_wave + meso_wave

        # 2. Dynamic corridor gap G(t)
        gap_wave = self.config.gap_wave_amp * np.sin(
            (2.0 * np.pi * t / self.config.gap_wave_len) + self.phase_gap
        )
        corridor_gap = self.config.base_corridor_gap + gap_wave

        # 3. Micro roughness
        jitter_top = float(self.rng.normal(0.0, self.config.micro_roughness_sigma))
        jitter_bot = float(self.rng.normal(0.0, self.config.micro_roughness_sigma))

        # 4. Protruding rock crags
        crag_top = sum(c.get_protrusion_at(col_x) for c in self.crags if c.side == "top")
        crag_bot = sum(c.get_protrusion_at(col_x) for c in self.crags if c.side == "bottom")

        # Initial raw heights
        raw_top = center_y - (corridor_gap / 2.0) + jitter_top + crag_top
        raw_bot = self.height - (center_y + (corridor_gap / 2.0)) + jitter_bot + crag_bot

        # Enforce minimum ridge thicknesses
        top_h = max(float(self.config.min_ridge_height), raw_top)
        bot_h = max(float(self.config.min_ridge_height), raw_bot)

        # ------------------------------------------------------------------
        # 5. STRICT INVARIANT: Guaranteed Minimal Corridor Passage
        # ------------------------------------------------------------------
        current_gap = self.height - (top_h + bot_h)
        if current_gap < self.min_corridor_gap:
            deficit = self.min_corridor_gap - current_gap
            # Push ridges back outward away from the corridor center proportionally
            top_h -= deficit / 2.0
            bot_h -= deficit / 2.0

        # Clamping safety check
        top_h = max(float(self.config.min_ridge_height), top_h)
        bot_h = max(float(self.config.min_ridge_height), bot_h)

        # In extreme bounds edge cases, lock strictly to minimal passage
        if (self.height - (top_h + bot_h)) < self.min_corridor_gap:
            available_total = self.height - self.min_corridor_gap
            top_h = available_total / 2.0
            bot_h = available_total / 2.0

        return top_h, bot_h

    def _verify_corridor_clearance(self) -> None:
        """
        Defensive assertion checking that every single column across the screen
        strictly respects the minimal passage guarantee.
        """
        gaps = self.height - (self.top_heights + self.bot_heights)
        min_detected = float(np.min(gaps))
        assert min_detected >= self.min_corridor_gap - 1e-3, (
            f"Minimal corridor passage violated! Detected minimum gap: {min_detected:.2f}px "
            f"(required: {self.min_corridor_gap}px)"
        )

    def get_corridor_bounds(self, x: int) -> Tuple[int, int]:
        """
        Returns (y_min, y_max) inclusive boundaries of the navigable corridor
        at column x.
        """
        clamped_x = int(np.clip(x, 0, self.width - 1))
        y_min = int(self.top_heights[clamped_x])
        y_max = int(self.height - self.bot_heights[clamped_x] - 1)
        return y_min, y_max

    def step(self) -> None:
        """
        Advances the world by 1 timestep:
        1. Scrolls active crags left by 1 pixel and stochastically spawns incoming crags.
        2. Scrolls topography left by 1 pixel and synthesizes incoming column.
        3. Shifts and culls active trees and targets.
        4. Stochastically spawns new trees and targets within the guaranteed open passage.
        5. Verifies minimal corridor clearance across all screen columns.
        """
        self.global_step += 1

        # 1. Scroll active crags left
        for crag in self.crags:
            crag.peak_x -= self.config.scroll_speed
        self.crags = [c for c in self.crags if c.peak_x + (c.width / 2.0) >= 0]

        # Stochastically start a new crag at incoming right boundary
        if self.rng.random() < self.config.crag_probability:
            crag_w = float(self.rng.integers(
                self.config.crag_width_range[0], self.config.crag_width_range[1] + 1
            ))
            crag_h = float(self.rng.integers(
                self.config.crag_height_range[0], self.config.crag_height_range[1] + 1
            ))
            side = "top" if self.rng.random() < 0.5 else "bottom"
            self.crags.append(RockCrag(
                peak_x=float(self.width - 1 + crag_w / 2.0),
                width=crag_w,
                height=crag_h,
                side=side,
            ))

        # 2. Scroll topography left by 1 pixel
        self.top_heights[:-1] = self.top_heights[1:]
        self.bot_heights[:-1] = self.bot_heights[1:]

        # Compute incoming rightmost column (x = width - 1)
        new_top, new_bot = self._compute_column_heights(
            t=float(self.global_step + self.width - 1),
            col_x=float(self.width - 1),
        )
        self.top_heights[-1] = new_top
        self.bot_heights[-1] = new_bot

        # 3. Shift active trees and targets left
        for tree in self.trees:
            tree.x -= self.config.scroll_speed
        for target in self.targets:
            target.x -= self.config.scroll_speed

        # Cull off-screen items
        self.trees = [t for t in self.trees if not t.is_offscreen]
        self.targets = [tgt for tgt in self.targets if not tgt.is_offscreen and not tgt.collected]

        # 4. Stochastic spawning at x = width (right edge)
        corridor_min, corridor_max = self.get_corridor_bounds(self.width - 1)
        corridor_open_space = corridor_max - corridor_min

        # Spawn Tree?
        if self.rng.random() < self.config.p_tree and corridor_open_space >= 80:
            r_out = int(self.rng.integers(
                self.config.tree_radius_range[0],
                self.config.tree_radius_range[1] + 1
            ))
            r_in = max(3, r_out // 3)
            # Ensure tree is spawned with comfortable margins from both mountain borders
            # leaving ample passage for the 14px agent
            spawn_y_min = corridor_min + r_out + 12
            spawn_y_max = corridor_max - r_out - 12
            if spawn_y_min < spawn_y_max:
                tree_y = float(self.rng.integers(spawn_y_min, spawn_y_max))
                self.trees.append(Tree(
                    x=float(self.width + r_out),
                    y=tree_y,
                    outer_radius=r_out,
                    inner_radius=r_in,
                ))

        # Spawn Target?
        if self.rng.random() < self.config.p_target:
            sz = self.config.target_size
            spawn_y_min = corridor_min + sz + 8
            spawn_y_max = corridor_max - sz - 8
            if spawn_y_min < spawn_y_max:
                target_y = float(self.rng.integers(spawn_y_min, spawn_y_max))
                self.targets.append(Target(
                    x=float(self.width + sz),
                    y=target_y,
                    size=sz,
                ))

        # 5. Continuous verification of minimal corridor passage
        self._verify_corridor_clearance()
