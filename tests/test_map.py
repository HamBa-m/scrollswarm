"""
Unit tests for ScrollingMap, procedural obstacles, and guaranteed clearance.
"""

import numpy as np
import pytest
from scrollswarm.map import ScrollingMap, MapConfig
from scrollswarm.items import Obstacle, Tree, Target


def test_map_initialization():
    config = MapConfig(width=400, height=200, seed=42, preseed=False)
    m = ScrollingMap(config)

    assert m.top_heights.shape == (400,)
    assert m.bot_heights.shape == (400,)
    assert len(m.obstacles) == 0
    assert len(m.targets) == 0

    # Ensure corridor constraints are respected everywhere
    for x in range(m.width):
        corridor_gap = m.height - (m.top_heights[x] + m.bot_heights[x])
        assert corridor_gap >= m.min_corridor_gap - 1e-3
        assert m.top_heights[x] >= config.ceiling_y
        assert m.bot_heights[x] >= config.floor_margin


def test_map_scrolling_shifts_one_pixel():
    config = MapConfig(width=200, height=100, seed=123, preseed=False)
    m = ScrollingMap(config)

    initial_top = m.top_heights.copy()
    initial_bot = m.bot_heights.copy()

    m.obstacles.append(Obstacle(x=100.0, y=50.0, radius=10))
    m.targets.append(Target(x=150.0, y=50.0, size=8))

    m.step()

    np.testing.assert_allclose(m.top_heights[:-1], initial_top[1:])
    np.testing.assert_allclose(m.bot_heights[:-1], initial_bot[1:])

    # Entity positions must shift left by exactly 1 pixel
    assert m.obstacles[0].x == pytest.approx(99.0)
    assert m.targets[0].x == pytest.approx(149.0)


def test_culling_offscreen_entities():
    config = MapConfig(width=200, height=100, seed=999, preseed=False)
    m = ScrollingMap(config)

    # Place items
    obs = Obstacle(x=-25.0, y=50.0, radius=2, width=4)
    target = Target(x=-9.0, y=50.0, size=8)
    m.obstacles.append(obs)
    m.targets.append(target)

    m.step()
    # Culled offscreen
    assert len(m.obstacles) == 0
    assert len(m.targets) == 0


def test_seed_reproducibility():
    config1 = MapConfig(width=200, height=100, seed=777, preseed=False)
    m1 = ScrollingMap(config1)
    for _ in range(50):
        m1.step()

    config2 = MapConfig(width=200, height=100, seed=777, preseed=False)
    m2 = ScrollingMap(config2)
    for _ in range(50):
        m2.step()

    np.testing.assert_allclose(m1.top_heights, m2.top_heights)
    np.testing.assert_allclose(m1.bot_heights, m2.bot_heights)
    assert len(m1.obstacles) == len(m2.obstacles)
    assert len(m1.targets) == len(m2.targets)


def test_guaranteed_minimal_passage_across_500_steps():
    """
    Rigorously tests that across 500 consecutive scrolling steps,
    corridor gap unconditionally preserves at least min_corridor_gap.
    """
    config = MapConfig(width=800, height=400, min_corridor_gap=110, seed=42)
    m = ScrollingMap(config)

    for step_idx in range(500):
        m.step()
        gaps = m.height - (m.top_heights + m.bot_heights)
        min_gap = float(np.min(gaps))
        assert min_gap >= config.min_corridor_gap - 1e-3


def test_dynamic_floating_hazard_kinematics():
    """
    Tests that floating debris executes drift and oscillation.
    """
    config = MapConfig(width=800, height=400, seed=99)
    m = ScrollingMap(config)
    obs = m._create_floating_hazard(spawn_x=500.0)

    init_x = obs.x
    y_history = []
    for _ in range(50):
        obs.step()
        y_history.append(obs.y)

    assert obs.x < init_x
    if obs.vy_amplitude > 0.0:
        assert len(set(y_history)) > 1
