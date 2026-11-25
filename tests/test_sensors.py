"""
Unit tests for the RaySensor module and 16-dimensional observation vector.
"""

import numpy as np
import pytest
from scrollswarm.sensors import RaySensor, RayHit
from scrollswarm.map import ScrollingMap, MapConfig
from scrollswarm.items import Obstacle, Target
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

    # Dimension must be exactly: 1 (self y) + 7 (obs rays) + 4 (tgt features) + 4 (nbr features) = 16
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (16,)
    assert obs.dtype == np.float32

    # All values must be bounded in [-1.0, 1.0]
    assert np.all(obs >= -1.0)
    assert np.all(obs <= 1.0)
    assert obs[0] == pytest.approx(200.0 / 400.0)


def test_boundary_detection_by_rays():
    config = MapConfig(width=800, height=400, seed=123)
    m = ScrollingMap(config)
    sensor = RaySensor(num_rays=7, max_range=400.0)

    # Place origin right below the ceiling
    top_h = m.ceiling_y
    obs_hits, _ = sensor.cast_rays(
        origin_x=100.0, origin_y=top_h + 10.0, scrolling_map=m
    )

    # The steepest upward ray (ray 0) should hit the ceiling within a very short distance
    assert obs_hits[0].hit_type == "boundary"
    assert obs_hits[0].distance < 30.0

    # Place origin right above ground
    bot_y = m.floor_y
    obs_hits_bot, _ = sensor.cast_rays(
        origin_x=100.0, origin_y=bot_y - 10.0, scrolling_map=m
    )

    # The steepest downward ray (ray 6) should hit the ground within a very short distance
    assert obs_hits_bot[6].hit_type == "boundary"
    assert obs_hits_bot[6].distance < 30.0


def test_obstacle_detection_by_horizontal_ray():
    config = MapConfig(width=800, height=400, seed=99)
    m = ScrollingMap(config)
    m.obstacles.clear()

    # Place obstacle directly ahead along horizontal line (ray 3, angle 0)
    obs = Obstacle(x=250.0, y=200.0, radius=15.0, width=30.0, height=30.0)
    m.obstacles.append(obs)

    sensor = RaySensor(num_rays=7, max_range=400.0)
    obs_hits, _ = sensor.cast_rays(origin_x=100.0, origin_y=200.0, scrolling_map=m)

    center_ray = obs_hits[3]
    assert center_ray.hit_type == "obstacle"
    # Expected distance: (250 - 100) - hw = 150 - 15 = 135
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
    assert center_ray.distance == pytest.approx(90.0, abs=1.0)


def test_agent_get_observation_integration():
    config = MapConfig(width=800, height=400, seed=42)
    m = ScrollingMap(config)
    agent = Agent(agent_id=0, x=120.0, y=200.0)

    obs = agent.get_observation(m)
    assert obs.shape == (16,)
    assert obs[0] == pytest.approx(200.0 / 400.0)


def test_observation_target_and_neighbor_feature_slices():
    """Verifies exact encoding of slices [8:12] (nearest target) and [12:16] (nearest living neighbor)."""
    config = MapConfig(width=800, height=400, seed=42, preseed=False)
    m = ScrollingMap(config)
    m.obstacles.clear()
    m.targets.clear()

    # Target placed (+40, +20) from agent -> dist = sqrt(2000) approx 44.72 <= photo_radius (50)
    m.targets.append(Target(x=140.0, y=220.0, size=10))
    # Collected target closer should be ignored
    collected_tgt = Target(x=110.0, y=200.0, size=10)
    collected_tgt.collected = True
    m.targets.append(collected_tgt)

    self_agent = Agent(agent_id=0, x=100.0, y=200.0)
    dead_near = Agent(agent_id=1, x=110.0, y=200.0)
    dead_near.alive = False
    alive_nbr = Agent(
        agent_id=2, x=180.0, y=140.0
    )  # dx=+80, dy=-60, dist=100 <= comm_range (150)

    sensor = RaySensor(num_rays=7, max_range=400.0, photo_radius=50.0, comm_range=150.0)
    obs = sensor.observe(
        origin_x=self_agent.x,
        origin_y=self_agent.y,
        scrolling_map=m,
        other_agents=[self_agent, dead_near, alive_nbr],
        agent_id=0,
    )

    diag = float(np.sqrt(800.0**2 + 400.0**2))
    expected_tgt_dist = float(np.sqrt(40.0**2 + 20.0**2))
    assert obs[8] == pytest.approx(40.0 / 800.0, abs=1e-5)
    assert obs[9] == pytest.approx(20.0 / 400.0, abs=1e-5)
    assert obs[10] == pytest.approx(expected_tgt_dist / diag, abs=1e-5)
    assert obs[11] == 1.0  # in_range

    assert obs[12] == pytest.approx(80.0 / 800.0, abs=1e-5)
    assert obs[13] == pytest.approx(-60.0 / 400.0, abs=1e-5)
    assert obs[14] == pytest.approx(100.0 / diag, abs=1e-5)
    assert obs[15] == 1.0  # in_comm
