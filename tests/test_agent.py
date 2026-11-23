"""
Unit tests for the Agent entity, direction, collision math, soft proximity, and team creation.
"""

import numpy as np
import pytest
from scrollswarm.agent import Agent, Action, create_agents, AGENT_PALETTE
from scrollswarm.items import Obstacle, Tree, Target
from scrollswarm.map import ScrollingMap, MapConfig


def test_agent_initialization():
    agent = Agent(agent_id=0, x=100.0, y=200.0, step_size=5)
    assert agent.agent_id == 0
    assert agent.x == 100.0
    assert agent.y == 200.0
    assert agent.step_size == 5
    assert agent.alive is True
    assert agent.score == 0

    tip, top_left, bot_left = agent.vertices
    assert tip[0] > agent.x
    assert top_left[0] < agent.x
    assert bot_left[0] < agent.x
    assert top_left[1] < agent.y
    assert bot_left[1] > agent.y


def test_agent_movement_k_pixels():
    agent = Agent(agent_id=0, x=100.0, y=200.0, step_size=5)

    # UP
    agent.step(Action.UP, map_height=400)
    assert agent.y == 195.0

    # DOWN
    agent.step(Action.DOWN, map_height=400)
    assert agent.y == 200.0

    # STAY
    agent.step(Action.STAY, map_height=400)
    assert agent.y == 200.0


def test_agent_clamping_at_boundaries():
    agent = Agent(agent_id=0, x=100.0, y=10.0, step_size=10, height=14.0)

    # Try moving UP past top boundary
    agent.step(Action.UP, map_height=400)
    assert agent.y == 7.0

    # Move to bottom boundary
    agent.y = 395.0
    agent.step(Action.DOWN, map_height=400)
    assert agent.y == 393.0


def test_boundary_collision_terminates_agent():
    config = MapConfig(width=400, height=200, seed=42)
    m = ScrollingMap(config)

    mid_y = (m.top_heights[100] + (m.height - m.bot_heights[100])) / 2.0
    agent = Agent(agent_id=0, x=100.0, y=mid_y)
    assert agent.check_collision_boundary(m) is False
    assert agent.alive is True

    # Move agent directly into the top boundary
    agent.y = m.top_heights[100] - 2.0
    assert agent.check_collision_boundary(m) is True
    assert agent.alive is False


def test_obstacle_collision_terminates_agent():
    agent = Agent(agent_id=0, x=100.0, y=200.0)
    obs_far = Obstacle(x=300.0, y=200.0, radius=20.0)
    obs_close = Obstacle(x=105.0, y=200.0, radius=20.0)

    assert agent.check_collision_obstacles([obs_far]) is False
    assert agent.alive is True

    assert agent.check_collision_obstacles([obs_close]) is True
    assert agent.alive is False


def test_no_proximity_penalty_between_agents():
    """
    Verifies that proximity penalty is disabled (returns 0.0) between UAVs.
    """
    agent0 = Agent(agent_id=0, x=100.0, y=200.0)
    agent1 = Agent(agent_id=1, x=100.0, y=214.0)

    penalty0 = agent0.compute_proximity_penalty([agent0, agent1])
    assert penalty0 == 0.0


def test_stochastic_target_inspection_action_uncertainty():
    """
    Verifies stochastic target photography within inspection radius.
    """
    agent = Agent(agent_id=0, x=100.0, y=200.0)
    tgt = Target(x=110.0, y=200.0, inspection_radius=45.0, sensor_reliability=1.0)
    rng = np.random.default_rng(42)

    # Agent is at distance 10 <= 45
    captured = agent.check_target_inspection([tgt], rng=rng)
    assert captured in (0, 1)


def test_target_collection_backward_compat():
    agent = Agent(agent_id=0, x=100.0, y=200.0)
    tgt_far = Target(x=300.0, y=200.0, size=8)
    tgt_overlapping = Target(x=104.0, y=200.0, size=8)

    collected = agent.check_target_collection([tgt_far, tgt_overlapping])
    assert collected == 1
    assert agent.score == 1
    assert tgt_overlapping.collected is True
    assert tgt_far.collected is False


def test_dead_agent_does_not_move():
    agent = Agent(agent_id=0, x=100.0, y=200.0)
    agent.alive = False

    agent.step(Action.UP, map_height=400)
    assert agent.y == 200.0


def test_create_agents_configurable_n():
    agents_default = create_agents()
    assert len(agents_default) == 5

    xs = [a.x for a in agents_default]
    assert sorted(xs) == xs
    assert len(set(xs)) == 5

    agents_1 = create_agents(num_agents=1)
    assert len(agents_1) == 1

    agents_10 = create_agents(num_agents=10)
    assert len(agents_10) == 10

    with pytest.raises(ValueError):
        create_agents(num_agents=0)

    with pytest.raises(ValueError):
        create_agents(num_agents=11)
