"""
Unit tests for ScrollingMap, multi-scale topography math, and guaranteed clearance.
"""

import numpy as np
import pytest
from objects.map import ScrollingMap, MapConfig
from objects.items import Tree, Target


def test_map_initialization():
    config = MapConfig(width=400, height=200, seed=42)
    m = ScrollingMap(config)

    assert m.top_heights.shape == (400,)
    assert m.bot_heights.shape == (400,)
    assert len(m.trees) == 0
    assert len(m.targets) == 0

    # Ensure corridor constraints are respected everywhere
    for x in range(m.width):
        corridor_gap = m.height - (m.top_heights[x] + m.bot_heights[x])
        assert corridor_gap >= m.min_corridor_gap - 1e-3
        assert m.top_heights[x] >= config.min_ridge_height
        assert m.bot_heights[x] >= config.min_ridge_height


def test_map_scrolling_shifts_one_pixel():
    config = MapConfig(width=200, height=100, seed=123)
    m = ScrollingMap(config)

    # Save topography before step
    initial_top = m.top_heights.copy()
    initial_bot = m.bot_heights.copy()

    # Place an entity at known coordinates
    m.trees.append(Tree(x=100.0, y=50.0, outer_radius=10, inner_radius=3))
    m.targets.append(Target(x=150.0, y=50.0, size=8))

    m.step()

    # The topography from x=1..W should now be at x=0..W-1
    np.testing.assert_allclose(m.top_heights[:-1], initial_top[1:])
    np.testing.assert_allclose(m.bot_heights[:-1], initial_bot[1:])

    # Entity positions must shift left by exactly 1 pixel
    assert m.trees[0].x == pytest.approx(99.0)
    assert m.targets[0].x == pytest.approx(149.0)


def test_culling_offscreen_entities():
    config = MapConfig(width=200, height=100, seed=999)
    m = ScrollingMap(config)

    # Place item right at the left boundary
    tree = Tree(x=-9.0, y=50.0, outer_radius=10)   # x + radius = 1 (on screen)
    target = Target(x=-7.0, y=50.0, size=8)         # x + size = 1 (on screen)
    m.trees.append(tree)
    m.targets.append(target)

    # Step 1: tree moves to x=-10 (x + radius = 0, still >= 0)
    m.step()
    assert len(m.trees) == 1
    assert len(m.targets) == 1

    # Step 2: tree moves to x=-11 (x + radius = -1 < 0 -> culled)
    m.step()
    assert len(m.trees) == 0
    assert len(m.targets) == 0


def test_seed_reproducibility():
    config1 = MapConfig(width=200, height=100, seed=777)
    m1 = ScrollingMap(config1)
    for _ in range(50):
        m1.step()

    config2 = MapConfig(width=200, height=100, seed=777)
    m2 = ScrollingMap(config2)
    for _ in range(50):
        m2.step()

    np.testing.assert_allclose(m1.top_heights, m2.top_heights)
    np.testing.assert_allclose(m1.bot_heights, m2.bot_heights)
    assert len(m1.trees) == len(m2.trees)
    assert len(m1.targets) == len(m2.targets)


def test_guaranteed_minimal_passage_across_500_steps():
    """
    Rigorously tests that across 500 consecutive scrolling steps,
    EVERY single column on the screen unconditionally preserves
    at least the minimal corridor gap (no bottlenecks ever squeeze below threshold).
    """
    config = MapConfig(width=800, height=400, min_corridor_gap=110, seed=42)
    m = ScrollingMap(config)

    for step_idx in range(500):
        m.step()
        gaps = m.height - (m.top_heights + m.bot_heights)
        min_gap = float(np.min(gaps))
        assert min_gap >= config.min_corridor_gap - 1e-3, (
            f"Step {step_idx}: minimum passage violated! Gap was {min_gap:.2f}px"
        )


def test_topography_dynamic_variance_not_flat():
    """
    Tests that the multi-scale topography exhibits real undulating terrain
    (macro climbs, valleys, chicanes) rather than a flat line.
    """
    config = MapConfig(width=800, height=400, seed=99)
    m = ScrollingMap(config)

    # The top height across columns must have substantial vertical variation (std > 15px)
    std_top = float(np.std(m.top_heights))
    ptp_top = float(np.ptp(m.top_heights))  # peak to peak

    assert std_top > 15.0, f"Topography too flat! Standard deviation was only {std_top:.2f}px"
    assert ptp_top > 50.0, f"Topography peak-to-peak difference too small! PTP was only {ptp_top:.2f}px"
