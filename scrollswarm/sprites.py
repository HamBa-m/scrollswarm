"""
Sprite rendering utilities for ScrollSwarm.

Provides vector sprite rendering and color-tint caching for the quadcopter agent model.
"""

import io
from pathlib import Path
from typing import Dict, Optional, Tuple
import pygame

# Default quadcopter visual dimensions (width, height in pixels)
DEFAULT_SPRITE_SIZE = (24, 24)

_SPRITE_CACHE: Dict[Tuple[Tuple[int, int, int], Tuple[int, int]], pygame.Surface] = {}
_SVG_TEMPLATE: Optional[str] = None


def _get_svg_template() -> str:
    global _SVG_TEMPLATE
    if _SVG_TEMPLATE is None:
        # Search package directory and workspace assets
        pkg_dir = Path(__file__).resolve().parent
        asset_candidates = [
            pkg_dir.parent / "assets" / "quadrocopte.svg",
            Path("assets/quadrocopte.svg"),
        ]
        svg_path = next((p for p in asset_candidates if p.exists()), None)
        if svg_path is None:
            raise FileNotFoundError("Could not locate assets/quadrocopte.svg")

        with open(svg_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # Ensure width and height match the 470.606 viewBox so sprite is centered with no offset
        raw = raw.replace('height="800px" width="800px"', 'width="470.606px" height="470.606px"')
        _SVG_TEMPLATE = raw

    return _SVG_TEMPLATE


def get_quadcopter_sprite(
    color: Tuple[int, int, int],
    size: Tuple[int, int] = DEFAULT_SPRITE_SIZE,
) -> pygame.Surface:
    """
    Returns a cached Pygame Surface of the quadcopter tinted with the given RGB color
    and smoothly scaled to the requested pixel size.
    """
    cache_key = (color, size)
    if cache_key in _SPRITE_CACHE:
        return _SPRITE_CACHE[cache_key]

    template = _get_svg_template()
    hex_color = "#{:02x}{:02x}{:02x}".format(*color)
    colored_svg = template.replace('fill="#000000"', f'fill="{hex_color}"')

    # Load vector SVG surface and smoothly scale to target pixel size
    raw_surf = pygame.image.load(io.BytesIO(colored_svg.encode("utf-8")))
    scaled_surf = pygame.transform.smoothscale(raw_surf, size)
    _SPRITE_CACHE[cache_key] = scaled_surf
    return scaled_surf
