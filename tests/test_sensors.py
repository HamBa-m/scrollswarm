"""
Unit tests for the RaySensor module and 7-ray forward observation vector.
"""

import numpy as np
import pytest
from scrollswarm.sensors import RaySensor, RayHit
from scrollswarm.map import ScrollingMap, MapConfig
from scrollswarm.items import Tree, Target
from scrollswarm.agent import Agent


def test_raysensor_initialization():
    sensor = RaySensor(num_rays=7, max_range=400.0, fov_degrees=120.0)
    assert sensor.num_rays == 7
    assert sensor.max_range == 400.0
    assert len(sensor.angles) == 7

    # Center ray should point horizontally forward (0 radians)
    assert sensor.angles[3] == pytest.approx(0.0)
    # First ray should point upward-forward (-60 degrees = -pi/3)
    assert sensor.angles[0] == pytest.approx(-np.pi / 3.0)
    # Last ray should point downward-forward (+60 degrees = +pi/3)
    assert sensor.angles[6] == pytest.approx(np.pi / 3.0)


def test_ray_observation_vector_shape_and_bounds():
    config = MapConfig(width=800, height=400, seed=42)
    m = ScrollingMap(config)
    sensor = RaySensor(num_rays=7, max_range=400.0)

    # Cast from corridor center
    obs = sensor.observe(origin_x=100.0, origin_y=200.0, scrolling_map=m)

    # Dimension must be exactly: 1 (self y) + 7 (obs rays) + 7 (tgt rays) = 15
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (15,)
    assert obs.dtype == np.float32

    # All values must be bounded in [0.0, 1.0]
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)
    assert obs[0] == pytest.approx(200.0 / 400.0)


def test_boundary_mountain_detection_by_rays():
    config = MapConfig(width=800, height=400, seed=123)
    m = ScrollingMap(config)
    sensor = RaySensor(num_rays=7, max_range=400.0)

    # Place origin right below the top mountain ridge
    top_h = m.top_heights[100]
    obs_hits, _ = sensor.cast_rays(origin_x=100.0, origin_y=top_h + 10.0, scrolling_map=m)

    # The steepest upward ray (ray 0) should hit the top mountain within a very short distance
    assert obs_hits[0].hit_type in ("boundary", "mountain")
    assert obs_hits[0].distance < 30.0

    # Place origin right above bottom mountain ridge
    bot_y = 400.0 - m.bot_heights[100]
    obs_hits_bot, _ = sensor.cast_rays(origin_x=100.0, origin_y=bot_y - 10.0, scrolling_map=m)

    # The steepest downward ray (ray 6) should hit the bottom mountain within a very short distance
    assert obs_hits_bot[6].hit_type in ("boundary", "mountain")
    assert obs_hits_bot[6].distance < 30.0


def test_tree_detection_by_horizontal_ray():
    config = MapConfig(width=800, height=400, seed=99)
    m = ScrollingMap(config)
    m.trees.clear()

    # Place tree directly ahead along horizontal line (ray 3, angle 0)
    tree = Tree(x=250.0, y=200.0, outer_radius=15)
    m.trees.append(tree)

    sensor = RaySensor(num_rays=7, max_range=400.0)
    obs_hits, _ = sensor.cast_rays(origin_x=100.0, origin_y=200.0, scrolling_map=m)

    center_ray = obs_hits[3]
    assert center_ray.hit_type in ("asteroid", "tree")
    # Expected distance: (250 - 100) - radius = 150 - 15 = 135
    assert center_ray.distance == pytest.approx(135.0, abs=1.0)


def test_target_detection_by_ray():
    config = MapConfig(width=800, height=400, seed=99)
    m = ScrollingMap(config)
    m.targets.clear()

    # Place target directly ahead along ray 3 (angle 0)
    target = Target(x=200.0, y=200.0, size=10)
    m.targets.append(target)

    sensor = RaySensor(num_rays=7, max_range=400.0)
    _, tgt_hits = sensor.cast_rays(origin_x=100.0, origin_y=200.0, scrolling_map=m)

    center_ray = tgt_hits[3]
    assert center_ray.hit_type == "target"
    # Target distance: (200 - 100) - size = 100 - 10 = 90
    assert center_ray.distance == pytest.approx(90.0, abs=1.0)


def test_agent_get_observation_integration():
    config = MapConfig(width=800, height=400, seed=42)
    m = ScrollingMap(config)
    agent = Agent(agent_id=0, x=120.0, y=200.0)

    obs = agent.get_observation(m)
    assert obs.shape == (15,)
    assert obs[0] == pytest.approx(200.0 / 400.0)
