"""
Unit tests and PettingZoo API compliance tests for ScrollSwarmEnv.
"""

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test
from scrollswarm.env import ScrollSwarmEnv
from scrollswarm.map import MapConfig


def test_env_initialization():
    env = ScrollSwarmEnv(num_agents=5, max_cycles=1000)
    assert len(env.possible_agents) == 5
    assert env.possible_agents == ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"]
    assert env.max_cycles == 1000

    # Test observation and action spaces
    for a in env.possible_agents:
        obs_space = env.observation_space(a)
        act_space = env.action_space(a)
        assert obs_space.shape == (16,)
        assert obs_space.low[0] == -1.0
        assert obs_space.high[0] == 1.0
        assert act_space.n == 3


def test_env_reset():
    env = ScrollSwarmEnv(num_agents=3, max_cycles=500)
    obs, infos = env.reset(seed=42)

    assert len(obs) == 3
    assert len(infos) == 3
    for a in env.agents:
        assert obs[a].shape == (16,)
        assert np.all(obs[a] >= -1.0)
        assert np.all(obs[a] <= 1.0)
        assert infos[a]["targets_photographed"] == 0
        assert infos[a]["collision"] is None


def test_env_step_random_actions():
    env = ScrollSwarmEnv(num_agents=4, max_cycles=100)
    obs, infos = env.reset(seed=123)

    for _ in range(20):
        if not env.agents:
            break
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        for a in actions:
            assert isinstance(rewards[a], float)
            assert isinstance(terminations[a], bool)
            assert isinstance(truncations[a], bool)
            assert isinstance(infos[a], dict)


def test_env_truncation_at_max_cycles():
    env = ScrollSwarmEnv(num_agents=2, max_cycles=15)
    env.reset(seed=999)

    for step_i in range(15):
        actions = {a: 1 for a in env.agents}  # Action.STAY
        obs, rewards, terminations, truncations, infos = env.step(actions)
        if step_i < 14:
            for a in actions:
                assert truncations[a] is False
        else:
            for a in actions:
                assert truncations[a] is True
            assert len(env.agents) == 0


def test_env_rgb_array_render():
    env = ScrollSwarmEnv(num_agents=3, max_cycles=50, render_mode="rgb_array")
    env.reset(seed=42)
    img = env.render()
    assert isinstance(img, np.ndarray)
    assert img.shape == (400, 800, 3)
    assert img.dtype == np.uint8
    env.close()


def test_pettingzoo_parallel_api_compliance():
    """Runs the official PettingZoo parallel API test suite."""
    env = ScrollSwarmEnv(num_agents=3, max_cycles=50)
    parallel_api_test(env, num_cycles=50)
    env.close()


def test_env_communication_graph_integration():
    """Verifies that the environment exposes the dynamic communication graph."""
    env = ScrollSwarmEnv(num_agents=4, comm_topology="chain")
    env.reset(seed=42)

    adj = env.get_adjacency_matrix()
    assert adj.shape == (4, 4)
    assert adj[0, 1] == 1.0
    assert adj[1, 0] == 1.0

    edge_index, edge_attr = env.get_edge_index_and_attributes()
    assert edge_index.shape[0] == 2
    assert edge_attr.shape[1] == 3
    env.close()


def test_env_centralized_state_space():
    """Verifies centralized state() for CTDE algorithms (MAPPO / QMIX)."""
    env = ScrollSwarmEnv(num_agents=3)
    env.reset(seed=42)

    # 16 features per agent * 3 = 48
    assert env.state_space.shape == (48,)
    s = env.state()
    assert s.shape == (48,)
    assert s.dtype == np.float32

    # Manually terminate agent 1 and verify its slice in state() is zero
    env._agent_map["agent_1"].alive = False
    s_dead = env.state()
    assert np.all(s_dead[16:32] == 0.0)
    assert not np.all(s_dead[0:16] == 0.0)
    env.close()


def test_env_no_proximity_penalty_in_reward():
    """Verifies that drones flying close incur no proximity penalty in rewards."""
    env = ScrollSwarmEnv(num_agents=2, step_reward=0.01)
    env.reset(seed=42)

    # Place agent_0 and agent_1 vertically close
    env._agent_map["agent_0"].x = 100.0
    env._agent_map["agent_0"].y = 200.0
    env._agent_map["agent_1"].x = 100.0
    env._agent_map["agent_1"].y = 205.0

    obs, rewards, terms, truncs, infos = env.step({"agent_0": 1, "agent_1": 1})
    # Both agents simply receive standard step survival reward (+0.01)
    assert pytest.approx(rewards["agent_0"], abs=1e-3) == 0.01
    assert pytest.approx(rewards["agent_1"], abs=1e-3) == 0.01
    env.close()
