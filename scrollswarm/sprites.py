"""
Sprite rendering utilities for ScrollSwarm.

Provides vector sprite rendering and color-tint caching for the rocket probe agent model.
"""

import io
from pathlib import Path
from typing import Dict, Optional, Tuple
import pygame

# Default rocket visual dimensions (width, height in pixels)
DEFAULT_SPRITE_SIZE = (24, 24)

_SPRITE_CACHE: Dict[Tuple[Tuple[int, int, int], Tuple[int, int]], pygame.Surface] = {}
_SVG_TEMPLATE: Optional[str] = None


def _get_svg_template() -> str:
    global _SVG_TEMPLATE
    if _SVG_TEMPLATE is None:
        # Search package directory first, then workspace root
        pkg_dir = Path(__file__).resolve().parent
        asset_candidates = [
            pkg_dir / "assets" / "rocket.svg",
            pkg_dir.parent / "assets" / "rocket.svg",
            Path("assets/rocket.svg"),
        ]
        svg_path = next((p for p in asset_candidates if p.exists()), None)
        if svg_path is None:
            raise FileNotFoundError("Could not locate assets/rocket.svg")

        with open(svg_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # Ensure width and height match viewBox 24x24 so sprite is centered with no offset
        raw = raw.replace('width="800px" height="800px"', 'width="24px" height="24px"')
        _SVG_TEMPLATE = raw

    return _SVG_TEMPLATE


def get_rocket_sprite(
    color: Tuple[int, int, int],
    size: Tuple[int, int] = DEFAULT_SPRITE_SIZE,
) -> pygame.Surface:
    """
    Returns a cached Pygame Surface of the rocket probe tinted with the given RGB color
    and smoothly scaled to the requested pixel size.
    """
    cache_key = (color, size)
    if cache_key in _SPRITE_CACHE:
        return _SPRITE_CACHE[cache_key]

    template = _get_svg_template()
    hex_color = "#{:02x}{:02x}{:02x}".format(*color)
    colored_svg = template.replace('fill="#030D45"', f'fill="{hex_color}"')

    # Load vector SVG surface and smoothly scale to target pixel size
    raw_surf = pygame.image.load(io.BytesIO(colored_svg.encode("utf-8")))
    scaled_surf = pygame.transform.smoothscale(raw_surf, size)
    _SPRITE_CACHE[cache_key] = scaled_surf
    return scaled_surf


# Alias for backward compatibility
get_quadcopter_sprite = get_rocket_sprite
