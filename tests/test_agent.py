"""
Unit tests for the Agent entity, direction, collision math, and team creation.
"""

import pytest
from scrollswarm.agent import Agent, Action, create_agents, AGENT_PALETTE
from scrollswarm.items import Tree, Target
from scrollswarm.map import ScrollingMap, MapConfig


def test_agent_initialization():
    agent = Agent(agent_id=0, x=100.0, y=200.0, step_size=5)
    assert agent.agent_id == 0
    assert agent.x == 100.0
    assert agent.y == 200.0
    assert agent.step_size == 5
    assert agent.alive is True
    assert agent.score == 0

    # Verify triangle points RIGHT (tip has larger x than base)
    tip, top_left, bot_left = agent.vertices
    assert tip[0] > agent.x
    assert top_left[0] < agent.x
    assert bot_left[0] < agent.x
    assert top_left[1] < agent.y  # above center
    assert bot_left[1] > agent.y  # below center


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
    assert agent.y == 7.0  # Clamped to half_h = 7.0

    # Move to bottom boundary
    agent.y = 395.0
    agent.step(Action.DOWN, map_height=400)
    assert agent.y == 393.0  # Clamped to 400 - 7.0 = 393.0


def test_mountain_collision_terminates_agent():
    config = MapConfig(width=400, height=200, seed=42)
    m = ScrollingMap(config)

    # Place agent safely in the middle of corridor
    mid_y = (m.top_heights[100] + (m.height - m.bot_heights[100])) / 2.0
    agent = Agent(agent_id=0, x=100.0, y=mid_y)
    assert agent.check_collision_mountain(m) is False
    assert agent.alive is True

    # Move agent directly into the top mountain ridge
    agent.y = m.top_heights[100] - 2.0
    assert agent.check_collision_mountain(m) is True
    assert agent.alive is False


def test_tree_collision_terminates_agent():
    agent = Agent(agent_id=0, x=100.0, y=200.0)
    tree_far = Tree(x=300.0, y=200.0, outer_radius=14)
    tree_close = Tree(x=105.0, y=200.0, outer_radius=14)

    assert agent.check_collision_trees([tree_far]) is False
    assert agent.alive is True

    assert agent.check_collision_trees([tree_close]) is True
    assert agent.alive is False


def test_target_collection():
    agent = Agent(agent_id=0, x=100.0, y=200.0)
    tgt_far = Target(x=300.0, y=200.0, size=8)
    tgt_overlapping = Target(x=104.0, y=200.0, size=8)

    collected = agent.check_target_collection([tgt_far, tgt_overlapping])
    assert collected == 1
    assert agent.score == 1
    assert tgt_overlapping.collected is True
    assert tgt_far.collected is False

    # Second check shouldn't collect it again
    assert agent.check_target_collection([tgt_overlapping]) == 0
    assert agent.score == 1


def test_dead_agent_does_not_move():
    agent = Agent(agent_id=0, x=100.0, y=200.0)
    agent.alive = False

    agent.step(Action.UP, map_height=400)
    assert agent.y == 200.0


def test_create_agents_configurable_n():
    # Default 5 agents
    agents_default = create_agents()
    assert len(agents_default) == 5

    # Check distinct columns and monotonically increasing x
    xs = [a.x for a in agents_default]
    assert sorted(xs) == xs
    assert len(set(xs)) == 5

    # Test N = 1
    agents_1 = create_agents(num_agents=1)
    assert len(agents_1) == 1

    # Test N = 10
    agents_10 = create_agents(num_agents=10)
    assert len(agents_10) == 10

    # Bounds check: N < 1 or N > 10 must raise ValueError
    with pytest.raises(ValueError):
        create_agents(num_agents=0)

    with pytest.raises(ValueError):
        create_agents(num_agents=11)
