"""
Entity definitions for ScrollSwarm.
Data structures for items that scroll from right to left.
"""

from dataclasses import dataclass


@dataclass
class Tree:
    """A tree obstacle: circular canopy with a brown center trunk."""
    x: float
    y: float
    outer_radius: int = 14
    inner_radius: int = 5

    @property
    def is_offscreen(self) -> bool:
        """Returns True if the tree has scrolled past the left edge."""
        return self.x + self.outer_radius < 0


@dataclass
class Target:
    """A collection target: rendered as a red X mark."""
    x: float
    y: float
    size: int = 8
    collected: bool = False

    @property
    def is_offscreen(self) -> bool:
        """Returns True if the target has scrolled past the left edge."""
        return self.x + self.size < 0
