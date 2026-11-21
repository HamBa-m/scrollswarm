"""
Map and Topography engine for ScrollSwarm.

Models a 2:1 side-scrolling environment:
- Light green ground plain.
- Gray boundary mountain ridges driven by mean-reverting Brownian motion.
- Scrolling discrete step of 1 pixel per timestep.
- Stochastic tree and target spawning within the navigable corridor.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from objects.items import Tree, Target


# Enhanced Classic Color Palette (RGB) - Optimized for Contrast & Visibility
COLOR_LAND = (195, 230, 185)         # Soft pastel meadow
COLOR_MOUNTAIN = (65, 75, 85)        # Deep slate rock
COLOR_MOUNTAIN_RIM = (35, 42, 50)    # Dark slate contour rim
COLOR_TREE_LEAVES = (20, 120, 45)    # Crisp forest green
COLOR_TREE_RIM = (10, 60, 25)        # Dark canopy outline
COLOR_TREE_TRUNK = (130, 60, 20)     # Rich cedar core
COLOR_TARGET = (255, 30, 60)         # High-visibility vibrant scarlet


@dataclass
class MapConfig:
    """Configuration parameters for the scrolling map."""
    width: int = 800
    height: int = 400
    fps: int = 30
    scroll_speed: int = 1          # 1 pixel per timestep

    # Topography (Mean-Reverting Brownian Motion) parameters
    # dh_t = theta * (mu - h_t) + sigma * epsilon_t
    top_mu: float = 50.0           # Mean height for top mountain ridge
    bot_mu: float = 50.0           # Mean height for bottom mountain ridge
    ou_theta: float = 0.05         # Mean-reversion speed
    ou_sigma: float = 1.8          # Volatility / roughness of terrain
    min_ridge_height: int = 15     # Minimum mountain thickness
    min_corridor_gap: int = 140    # Guaranteed open vertical space between ridges

    # Spawning probabilities per column arrival
    p_tree: float = 0.015          # Probability of spawning a tree at right edge
    p_target: float = 0.010        # Probability of spawning a target at right edge

    # Entity geometry
    tree_radius_range: Tuple[int, int] = (10, 16)
    target_size: int = 8

    seed: Optional[int] = None


class ScrollingMap:
    """
    Manages the 2D world state, right-to-left scrolling, boundary topography,
    and stochastic entity lifecycles.
    """

    def __init__(self, config: Optional[MapConfig] = None):
        self.config = config or MapConfig()
        self.width = self.config.width
        self.height = self.config.height
        self.rng = np.random.default_rng(self.config.seed)

        # Topography profile: 1D arrays for top and bottom mountain ridge heights
        # top_height[x] is the thickness from y=0 down to y=top_height[x]
        # bot_height[x] is the thickness from y=H-1 up to y=H - bot_height[x]
        self.top_heights = np.zeros(self.width, dtype=np.float32)
        self.bot_heights = np.zeros(self.width, dtype=np.float32)

        # Active entities currently in the world
        self.trees: List[Tree] = []
        self.targets: List[Target] = []

        self.reset(self.config.seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Resets the map state, regenerating initial topography across the full width."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.trees.clear()
        self.targets.clear()

        # Initialize the initial full screen of topography using the OU process
        h_top = self.config.top_mu
        h_bot = self.config.bot_mu

        for x in range(self.width):
            h_top, h_bot = self._sample_next_heights(h_top, h_bot)
            self.top_heights[x] = h_top
            self.bot_heights[x] = h_bot

    def _sample_next_heights(self, prev_top: float, prev_bot: float) -> Tuple[float, float]:
        """
        Calculates the next column's top and bottom heights using discretized
        Ornstein-Uhlenbeck (mean-reverting Brownian motion) with corridor safety bounds.
        """
        # Brownian step for top ridge
        eps_top = self.rng.standard_normal()
        delta_top = (
            self.config.ou_theta * (self.config.top_mu - prev_top)
            + self.config.ou_sigma * eps_top
        )
        next_top = prev_top + delta_top

        # Brownian step for bottom ridge
        eps_bot = self.rng.standard_normal()
        delta_bot = (
            self.config.ou_theta * (self.config.bot_mu - prev_bot)
            + self.config.ou_sigma * eps_bot
        )
        next_bot = prev_bot + delta_bot

        # Enforce minimum ridge height
        next_top = max(float(self.config.min_ridge_height), next_top)
        next_bot = max(float(self.config.min_ridge_height), next_bot)

        # Enforce minimum corridor gap so the map remains navigable
        max_total_ridge = self.height - self.config.min_corridor_gap
        if (next_top + next_bot) > max_total_ridge:
            overshoot = (next_top + next_bot) - max_total_ridge
            # Reduce both proportionally
            next_top -= overshoot / 2.0
            next_bot -= overshoot / 2.0

        return next_top, next_bot

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
        1. Scrolls mountain arrays left by 1 pixel and samples a new rightmost column.
        2. Shifts all active entities left by 1 pixel.
        3. Culls off-screen entities.
        4. Stochastically spawns new trees and targets at the rightmost boundary.
        """
        # 1. Scroll topography left
        self.top_heights[:-1] = self.top_heights[1:]
        self.bot_heights[:-1] = self.bot_heights[1:]

        new_top, new_bot = self._sample_next_heights(
            self.top_heights[-2], self.bot_heights[-2]
        )
        self.top_heights[-1] = new_top
        self.bot_heights[-1] = new_bot

        # 2. Shift active trees and targets left
        for tree in self.trees:
            tree.x -= self.config.scroll_speed

        for target in self.targets:
            target.x -= self.config.scroll_speed

        # 3. Cull off-screen items
        self.trees = [t for t in self.trees if not t.is_offscreen]
        self.targets = [tgt for tgt in self.targets if not tgt.is_offscreen and not tgt.collected]

        # 4. Stochastic spawning at x = width (right edge)
        corridor_min, corridor_max = self.get_corridor_bounds(self.width - 1)

        # Spawn Tree?
        if self.rng.random() < self.config.p_tree:
            r_out = int(self.rng.integers(
                self.config.tree_radius_range[0],
                self.config.tree_radius_range[1] + 1
            ))
            r_in = max(3, r_out // 3)
            # Ensure tree is spawned strictly inside the navigable corridor
            spawn_y_min = corridor_min + r_out
            spawn_y_max = corridor_max - r_out
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
            spawn_y_min = corridor_min + sz
            spawn_y_max = corridor_max - sz
            if spawn_y_min < spawn_y_max:
                target_y = float(self.rng.integers(spawn_y_min, spawn_y_max))
                self.targets.append(Target(
                    x=float(self.width + sz),
                    y=target_y,
                    size=sz,
                ))
