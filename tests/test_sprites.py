"""
Unit tests for quadcopter sprite loading, recoloring, and caching.
"""

import pytest
import pygame
from scrollswarm.sprites import get_rocket_sprite, get_quadcopter_sprite, DEFAULT_SPRITE_SIZE
from scrollswarm.agent import AGENT_PALETTE, COLOR_AGENT_DEAD


@pytest.fixture(scope="module", autouse=True)
def init_pygame():
    pygame.init()
    yield
    pygame.quit()


def test_get_rocket_sprite_default_size():
    surf = get_rocket_sprite(AGENT_PALETTE[0])
    assert isinstance(surf, pygame.Surface)
    assert surf.get_size() == DEFAULT_SPRITE_SIZE


def test_get_quadcopter_sprite_default_size():
    surf = get_quadcopter_sprite(AGENT_PALETTE[0])
    assert isinstance(surf, pygame.Surface)
    assert surf.get_size() == DEFAULT_SPRITE_SIZE


def test_get_quadcopter_sprite_custom_size():
    custom_size = (32, 32)
    surf = get_quadcopter_sprite(AGENT_PALETTE[1], size=custom_size)
    assert surf.get_size() == custom_size


def test_get_quadcopter_sprite_caching():
    surf1 = get_quadcopter_sprite(AGENT_PALETTE[2], size=(24, 24))
    surf2 = get_quadcopter_sprite(AGENT_PALETTE[2], size=(24, 24))
    assert surf1 is surf2  # Identical cached object


def test_get_quadcopter_sprite_dead_color():
    surf_dead = get_quadcopter_sprite(COLOR_AGENT_DEAD, size=(24, 24))
    assert isinstance(surf_dead, pygame.Surface)
    assert surf_dead.get_size() == (24, 24)
