"""
Sprite rendering utilities for ScrollSwarm (UAV Swarm Reconnaissance).

Provides vector and procedural quadcopter sprite rendering with per-agent color tints.
"""

from typing import Dict, Optional, Tuple
import pygame

# Default quadcopter visual dimensions (width, height in pixels)
DEFAULT_SPRITE_SIZE = (24, 24)

_SPRITE_CACHE: Dict[Tuple[Tuple[int, int, int], Tuple[int, int]], pygame.Surface] = {}


def _create_procedural_quadcopter_surface(
    color: Tuple[int, int, int],
    size: Tuple[int, int] = DEFAULT_SPRITE_SIZE,
) -> pygame.Surface:
    """
    Renders a crisp, high-resolution quadcopter UAV facing right:
    - Sleek aerodynamic central fuselage (drone body).
    - 4 angled structural carbon-fiber rotor arms.
    - 4 translucent rotor discs with spinning blade accents.
    - Front-mounted optical reconnaissance camera pod / sensor lens.
    """
    w, h = size
    # Create surface with per-pixel alpha
    surface = pygame.Surface((w, h), pygame.SRCALPHA)

    cx, cy = w / 2.0, h / 2.0
    arm_span_x = w * 0.38
    arm_span_y = h * 0.36
    rotor_radius = max(3, int(min(w, h) * 0.16))

    # Rotor motor positions (4 corners of quadcopter)
    motors = [
        (cx - arm_span_x, cy - arm_span_y),
        (cx + arm_span_x, cy - arm_span_y),
        (cx - arm_span_x, cy + arm_span_y),
        (cx + arm_span_x, cy + arm_span_y),
    ]

    # 1. Carbon-fiber structural arms (dark metallic cross)
    arm_color = (40, 50, 65)
    for mx, my in motors:
        pygame.draw.line(
            surface, arm_color, (int(cx), int(cy)), (int(mx), int(my)), width=2
        )

    # 2. Spinning rotor discs (translucent spinning blur)
    rotor_disc_color = (200, 230, 255, 120)
    rotor_rim_color = (160, 200, 240, 180)
    for mx, my in motors:
        imx, imy = int(mx), int(my)
        # Outer disc
        pygame.draw.circle(surface, rotor_disc_color, (imx, imy), rotor_radius)
        pygame.draw.circle(surface, rotor_rim_color, (imx, imy), rotor_radius, width=1)
        # Motor hub
        pygame.draw.circle(surface, (30, 35, 45), (imx, imy), 2)

    # 3. Main fuselage body (elliptical aerodynamic pod tinted with agent color)
    body_rect = pygame.Rect(
        int(cx - w * 0.28),
        int(cy - h * 0.20),
        int(w * 0.56),
        int(h * 0.40),
    )
    # Fuselage fill with team color
    pygame.draw.ellipse(surface, color, body_rect)
    # Fuselage outline
    pygame.draw.ellipse(surface, (255, 255, 255), body_rect, width=1)

    # 4. Front optical reconnaissance camera / sensor lens (pointing right)
    cam_x = int(cx + w * 0.26)
    cam_y = int(cy)
    pygame.draw.circle(surface, (0, 230, 255), (cam_x, cam_y), 3)
    pygame.draw.circle(surface, (255, 255, 255), (cam_x, cam_y), 1)

    return surface


def get_quadcopter_sprite(
    color: Tuple[int, int, int],
    size: Tuple[int, int] = DEFAULT_SPRITE_SIZE,
) -> pygame.Surface:
    """
    Returns a cached Pygame Surface of the quadcopter UAV tinted with the given RGB color.
    """
    cache_key = (color, size)
    if cache_key in _SPRITE_CACHE:
        return _SPRITE_CACHE[cache_key]

    surf = _create_procedural_quadcopter_surface(color, size)
    _SPRITE_CACHE[cache_key] = surf
    return surf


# Backward compatibility aliases
get_rocket_sprite = get_quadcopter_sprite
