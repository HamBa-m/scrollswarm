"""
Unit tests for the Graph Communication Layer (objects/comm.py).
Tests topology generation, edge pruning on death, PyG conversion, and message routing.
"""

import numpy as np
import pytest
from objects.comm import CommunicationGraph, TopologyType
from objects.agent import create_agents


def test_chain_topology():
    agents = create_agents(num_agents=5)
    comm = CommunicationGraph(topology=TopologyType.CHAIN)
    adj = comm.get_adjacency_matrix(agents)

    assert adj.shape == (5, 5)
    assert np.all(np.diag(adj) == 0.0)

    # Chain connections: (0,1), (1,2), (2,3), (3,4)
    expected = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=np.float32)
    np.testing.assert_allclose(adj, expected)


def test_ring_topology():
    agents = create_agents(num_agents=5)
    comm = CommunicationGraph(topology=TopologyType.RING)
    adj = comm.get_adjacency_matrix(agents)

    # Ring connections: chain + (0, 4) wrap-around
    expected = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
    ], dtype=np.float32)
    np.testing.assert_allclose(adj, expected)
    # Every node in ring has degree 2
    assert np.all(np.sum(adj, axis=1) == 2.0)


def test_full_topology():
    agents = create_agents(num_agents=4)
    comm = CommunicationGraph(topology=TopologyType.FULL)
    adj = comm.get_adjacency_matrix(agents)

    expected = np.ones((4, 4), dtype=np.float32) - np.eye(4, dtype=np.float32)
    np.testing.assert_allclose(adj, expected)


def test_radius_topology():
    agents = create_agents(num_agents=5, map_width=800)
    # Agent positions are spaced along x in [80, 320]:
    # x = 80, 140, 200, 260, 320 (dx = 60px between consecutive agents)
    comm = CommunicationGraph(topology=TopologyType.RADIUS, radius=70.0)
    adj = comm.get_adjacency_matrix(agents)

    # With radius=70, agents can only reach distance <= 70 (adjacent neighbors at 60px)
    # So radius=70 matches the chain topology
    assert adj[0, 1] == 1.0
    assert adj[0, 2] == 0.0  # distance = 120 > 70


def test_capped_degree_topology():
    agents = create_agents(num_agents=6)
    comm = CommunicationGraph(topology=TopologyType.CAPPED, max_degree=2)
    adj = comm.get_adjacency_matrix(agents)

    # Every agent must have at most 2 outgoing edges
    out_degrees = np.sum(adj, axis=1)
    assert np.all(out_degrees <= 2.0)


def test_dead_agent_pruning():
    agents = create_agents(num_agents=5)
    comm = CommunicationGraph(topology=TopologyType.CHAIN)

    # Kill Agent 2
    agents[2].alive = False

    adj = comm.get_adjacency_matrix(agents)

    # Row 2 and Column 2 must be strictly 0
    assert np.all(adj[2, :] == 0.0)
    assert np.all(adj[:, 2] == 0.0)

    # Edge (1, 2) and (2, 3) must be severed
    assert adj[1, 2] == 0.0
    assert adj[3, 2] == 0.0
    # Other intact edges remain
    assert adj[0, 1] == 1.0
    assert adj[3, 4] == 1.0


def test_pyg_edge_index_and_attributes():
    agents = create_agents(num_agents=4)
    comm = CommunicationGraph(topology=TopologyType.CHAIN)

    edge_index, edge_attr = comm.get_edge_index_and_attributes(agents, map_width=800.0, map_height=400.0)

    # In 4-agent chain: 3 bidirectional edges = 6 directed edges
    assert edge_index.shape == (2, 6)
    assert edge_attr.shape == (6, 3)

    # Relative coordinates and distance bounds
    assert np.all(edge_attr[:, 2] >= 0.0)  # distance >= 0
    assert np.all(edge_attr[:, 2] <= 1.0)  # normalized distance <= 1


def test_message_routing_mean_and_sum():
    agents = create_agents(num_agents=3)
    comm = CommunicationGraph(topology=TopologyType.CHAIN)

    # Messages emitted by each agent:
    # Agent 0 emits [10.0]
    # Agent 1 emits [20.0]
    # Agent 2 emits [30.0]
    messages = {
        0: np.array([10.0], dtype=np.float32),
        1: np.array([20.0], dtype=np.float32),
        2: np.array([30.0], dtype=np.float32),
    }

    # Mean aggregation
    inbox_mean = comm.route_messages(messages, agents, aggregation="mean")

    # In chain 0-1-2:
    # Agent 0 has neighbor 1 -> receives [20.0]
    # Agent 1 has neighbors 0 and 2 -> receives mean([10, 30]) = [20.0]
    # Agent 2 has neighbor 1 -> receives [20.0]
    assert inbox_mean["agent_0"][0] == pytest.approx(20.0)
    assert inbox_mean["agent_1"][0] == pytest.approx(20.0)
    assert inbox_mean["agent_2"][0] == pytest.approx(20.0)

    # Sum aggregation
    inbox_sum = comm.route_messages(messages, agents, aggregation="sum")
    assert inbox_sum["agent_0"][0] == pytest.approx(20.0)
    assert inbox_sum["agent_1"][0] == pytest.approx(40.0)  # 10 + 30 = 40
    assert inbox_sum["agent_2"][0] == pytest.approx(20.0)


def test_message_routing_with_dead_agent():
    agents = create_agents(num_agents=3)
    comm = CommunicationGraph(topology=TopologyType.CHAIN)

    # Kill Agent 1
    agents[1].alive = False

    messages = {
        0: np.array([10.0], dtype=np.float32),
        1: np.array([20.0], dtype=np.float32),
        2: np.array([30.0], dtype=np.float32),
    }

    inbox = comm.route_messages(messages, agents, aggregation="mean")

    # Agent 0 was only connected to 1 (now dead) -> receives [0.0]
    assert inbox["agent_0"][0] == pytest.approx(0.0)
    # Agent 1 is dead -> receives [0.0]
    assert inbox["agent_1"][0] == pytest.approx(0.0)
    # Agent 2 was only connected to 1 (now dead) -> receives [0.0]
    assert inbox["agent_2"][0] == pytest.approx(0.0)
