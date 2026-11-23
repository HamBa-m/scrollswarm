"""
Graph Communication Topology & Message Routing Engine for ScrollSwarm.

Provides formal graph representations for Multi-Agent Reinforcement Learning (MARL):
- Supported Topologies:
    - CHAIN: Adjacent column neighbors (default for ScrollSwarm).
    - RING: Circular cycle graph with end-to-end wrap-around.
    - FULL: Complete graph (all-to-all broadcast baseline).
    - RADIUS: Dynamic Euclidean distance disk graph (||p_i - p_j|| <= R).
    - CAPPED: Degree-capped k-nearest spatial neighbors.
- Automatic Pruning: Dead agents immediately lose all incident edges.
- PyG Compatibility: Computes standard (2, E) edge_index and (E, 3) edge_attr.
- Message Router: Routes messages across edges with 'mean', 'max', 'sum', or 'concat' aggregation.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from scrollswarm.agent import Agent


class TopologyType(str, Enum):
    """
    Supported graph communication topologies for the ScrollSwarm swarm.

    Attributes:
        CHAIN: Linear chain — each drone connects to its immediate left and
            right column neighbors (maximum degree 2).
        RING: Circular ring — same as CHAIN but with wrap-around between the
            first and last drone.
        FULL: Complete broadcast graph — every drone connects to every other
            drone (maximum degree N−1).
        RADIUS: Spatial proximity disk — drones connect if their Euclidean
            distance is at most ``radius`` pixels.
        CAPPED: Degree-capped k-nearest-neighbor — each drone connects to its
            ``max_degree`` closest neighbors by Euclidean distance.
    """
    CHAIN = "chain"
    RING = "ring"
    FULL = "full"
    RADIUS = "radius"
    CAPPED = "capped"


class CommunicationGraph:
    """
    Manages communication topology, dynamic edge pruning, and message passing
    across the ScrollSwarm multi-agent swarm.

    The graph is recomputed dynamically each step based on agent liveness and,
    for spatial topologies (``RADIUS``, ``CAPPED``), agent positions.  Dead
    agents (``alive == False``) are immediately pruned: their rows and columns
    in the adjacency matrix are zeroed.

    Attributes:
        topology: Active ``TopologyType`` governing edge construction.
        radius: Maximum communication range in pixels (used by ``RADIUS``).
        max_degree: Maximum number of neighbors per drone (used by ``CAPPED``).
    """

    def __init__(
        self,
        topology: Union[TopologyType, str] = TopologyType.CHAIN,
        radius: float = 180.0,
        max_degree: int = 2,
    ):
        """
        Initializes the CommunicationGraph.

        Args:
            topology: Graph topology to use.  Pass a ``TopologyType`` enum
                value or its string equivalent (e.g. ``"chain"``).
            radius: Communication radius in pixels for the ``RADIUS`` topology.
                Drones further apart than this do not share edges.
            max_degree: Maximum number of incoming neighbors per drone for the
                ``CAPPED`` topology.
        """
        if isinstance(topology, str):
            topology = TopologyType(topology.lower())
        self.topology = topology
        self.radius = float(radius)
        self.max_degree = max_degree

    def get_adjacency_matrix(self, agents: List[Agent]) -> np.ndarray:
        """
        Computes the binary adjacency matrix for the current swarm state.

        ``A[i, j] = 1`` if a directed edge exists from agent ``j`` to agent
        ``i`` (i.e., agent ``i`` receives a message from agent ``j``).
        Dead agents (``alive == False``) have strictly zero rows and columns.

        Args:
            agents: Ordered list of all ``Agent`` instances in the swarm.

        Returns:
            Binary ``np.ndarray`` of shape ``(N, N)`` and dtype ``float32``.
        """
        n = len(agents)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)

        adj = np.zeros((n, n), dtype=np.float32)
        alive_mask = np.array([a.alive for a in agents], dtype=bool)

        # Positions: (N, 2)
        positions = np.array([[a.x, a.y] for a in agents], dtype=np.float32)

        if self.topology == TopologyType.CHAIN:
            # Agent i connects to adjacent column neighbors i-1 and i+1
            for i in range(n):
                if i > 0:
                    adj[i, i - 1] = 1.0
                if i < n - 1:
                    adj[i, i + 1] = 1.0

        elif self.topology == TopologyType.RING:
            for i in range(n):
                if n > 1:
                    adj[i, (i - 1) % n] = 1.0
                    adj[i, (i + 1) % n] = 1.0

        elif self.topology == TopologyType.FULL:
            # Complete graph (all-to-all, zero on diagonal)
            adj = np.ones((n, n), dtype=np.float32)
            np.fill_diagonal(adj, 0.0)

        elif self.topology == TopologyType.RADIUS:
            # Spatial distance graph
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            adj = (dist <= self.radius) & (dist > 1e-4)
            adj = adj.astype(np.float32)

        elif self.topology == TopologyType.CAPPED:
            # k-nearest spatial neighbors (max_degree)
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dist, np.inf)

            k = min(self.max_degree, max(0, n - 1))
            if k > 0:
                for i in range(n):
                    nearest_indices = np.argsort(dist[i])[:k]
                    for j in nearest_indices:
                        if not np.isinf(dist[i, j]):
                            adj[i, j] = 1.0

        # Enforce dead agent pruning: no edges to or from dead agents
        adj = adj * alive_mask[:, np.newaxis] * alive_mask[np.newaxis, :]
        return adj

    def get_edge_index_and_attributes(
        self,
        agents: List[Agent],
        map_width: float = 800.0,
        map_height: float = 400.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts active edges in PyTorch Geometric (PyG) COO format.

        Args:
            agents: Ordered list of all ``Agent`` instances in the swarm.
            map_width: Corridor width in pixels, used to normalize ``dx``.
            map_height: Corridor height in pixels, used to normalize ``dy``.

        Returns:
            edge_index: Integer array of shape ``(2, E)`` where
                ``edge_index[0]`` contains source node indices (``j``) and
                ``edge_index[1]`` contains target node indices (``i``).
            edge_attr: Float32 array of shape ``(E, 3)`` with normalized
                relative spatial features ``[dx/W, dy/H, dist/diag]`` for
                each directed edge ``j → i``.
        """
        adj = self.get_adjacency_matrix(agents)
        target_indices, source_indices = np.where(adj > 0.5)

        # PyG convention: [source, target] = [j, i]
        edge_index = np.stack([source_indices, target_indices], axis=0).astype(np.int64)
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            return edge_index, np.zeros((0, 3), dtype=np.float32)

        positions = np.array([[a.x, a.y] for a in agents], dtype=np.float32)
        diag = np.sqrt(map_width ** 2 + map_height ** 2)

        src_pos = positions[source_indices]
        tgt_pos = positions[target_indices]

        dx = (src_pos[:, 0] - tgt_pos[:, 0]) / map_width
        dy = (src_pos[:, 1] - tgt_pos[:, 1]) / map_height
        dist = np.sqrt((src_pos[:, 0] - tgt_pos[:, 0]) ** 2 + (src_pos[:, 1] - tgt_pos[:, 1]) ** 2) / diag

        edge_attr = np.stack([dx, dy, dist], axis=-1).astype(np.float32)
        return edge_index, edge_attr

    def route_messages(
        self,
        messages: Dict[Union[int, str], np.ndarray],
        agents: List[Agent],
        aggregation: str = "mean",
        message_dim: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Routes and aggregates neighbor messages across the communication graph.

        Each living agent ``i`` collects the messages of its in-neighbors
        ``j ∈ N(i)`` and aggregates them according to the chosen operator.
        Isolated or dead agents receive a zero vector.

        Args:
            messages: Dict mapping agent ID (``int`` or ``"agent_i"`` string)
                to a 1-D message array of shape ``(M,)``.
            agents: Ordered list of all ``Agent`` instances in the swarm.
            aggregation: Aggregation function applied across in-neighbor
                messages.  One of:

                - ``"mean"`` *(default)* — permutation-invariant mean pooling.
                - ``"sum"`` — cumulative sum (sensitive to degree).
                - ``"max"`` — element-wise maximum (saliency filtering).
                - ``"concat"`` — concatenation up to ``max_degree`` neighbors
                  (fixed-size; order is arbitrary).

            message_dim: Dimensionality ``M`` of each message.  Inferred
                from the first entry of ``messages`` if omitted.

        Returns:
            Dict mapping ``"agent_i"`` to the aggregated neighbor message
            array of shape ``(M,)`` (or ``(M * k,)`` for ``"concat"``).

        Raises:
            ValueError: If ``aggregation`` is not one of the four supported
                operators.
        """
        adj = self.get_adjacency_matrix(agents)
        n = len(agents)

        # Infer message dimension M
        if message_dim is None:
            for val in messages.values():
                message_dim = val.shape[0]
                break
            if message_dim is None:
                message_dim = 1

        # Standardize message lookup
        msg_tensor = np.zeros((n, message_dim), dtype=np.float32)
        for i, a in enumerate(agents):
            if not a.alive:
                continue
            key_int = a.agent_id
            key_str = f"agent_{a.agent_id}"
            if key_int in messages:
                msg_tensor[i] = messages[key_int]
            elif key_str in messages:
                msg_tensor[i] = messages[key_str]

        inbox: Dict[str, np.ndarray] = {}

        for i, agent in enumerate(agents):
            agent_key = f"agent_{agent.agent_id}"
            if not agent.alive:
                inbox[agent_key] = np.zeros(message_dim, dtype=np.float32)
                continue

            # In-neighbors of agent i: indices j where adj[i, j] > 0
            neighbor_indices = np.where(adj[i] > 0.5)[0]

            if len(neighbor_indices) == 0:
                # Isolated node or no alive neighbors
                inbox[agent_key] = np.zeros(message_dim, dtype=np.float32)
            else:
                neighbor_msgs = msg_tensor[neighbor_indices]  # (K, M)

                if aggregation == "mean":
                    inbox[agent_key] = np.mean(neighbor_msgs, axis=0)
                elif aggregation == "sum":
                    inbox[agent_key] = np.sum(neighbor_msgs, axis=0)
                elif aggregation == "max":
                    inbox[agent_key] = np.max(neighbor_msgs, axis=0)
                elif aggregation == "concat":
                    # Flatten up to max_degree neighbors
                    inbox[agent_key] = neighbor_msgs.flatten()
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")

        return inbox

    def get_communication_edges(
        self,
        agents: List[Agent],
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Returns line segment coordinates [(p_source, p_target)] for active edges,
        convenient for visual rendering in Pygame or debugging.
        """
        adj = self.get_adjacency_matrix(agents)
        n = len(agents)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                # If connected in either direction
                if adj[i, j] > 0.5 or adj[j, i] > 0.5:
                    edges.append(((agents[i].x, agents[i].y), (agents[j].x, agents[j].y)))
        return edges
