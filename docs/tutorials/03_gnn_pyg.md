# Tutorial 03: Graph Neural Networks & PyTorch Geometric Integration

Autonomous aerial swarms are inherently relational systems. Rather than treating drones as an unstructured collection of independent agents, modeling the swarm as a **dynamic communication graph** enables Graph Neural Networks (GNNs) to coordinate reconnaissance, route target information, and avoid collisions.

This tutorial guides you through:

1. Formulating the swarm as a dynamic spatial-relational graph $\mathcal{G}(t)$.
2. Extracting PyTorch Geometric (PyG) compatible tensor representations directly from the environment.
3. Performing native, dependency-free message passing using ScrollSwarm's built-in `route_messages` engine.
4. Constructing `torch_geometric.data.Data` objects for training GNN policies (GCN, GAT, GraphConv).
5. Handling dynamic topology changes and edge pruning when agents crash.

---

## 1. Relational Graph Formulation of Aerial Swarms

At any simulation step $t$, the aerial swarm is formally represented as an attributed directed graph:

$$\mathcal{G}(t) = \big( \mathcal{V}(t), \; \mathcal{E}(t), \; \mathbf{X}(t), \; \mathbf{E}(t) \big)$$

Where:

- **Active Nodes $\mathcal{V}(t) \subseteq \{1, \dots, N\}$**: The set of currently operational (alive) UAVs:
  $$\mathcal{V}(t) = \big\{ i \in \{1, \dots, N\} \;\big|\; \text{alive}_i(t) = \text{True} \big\}$$
- **Directed Edges $\mathcal{E}(t) \subseteq \mathcal{V}(t) \times \mathcal{V}(t)$**: Inter-drone communication links. A directed edge $(j \to i) \in \mathcal{E}(t)$ indicates that drone $i$ receives messages from drone $j$.
- **Node Feature Matrix $\mathbf{X}(t) \in \mathbb{R}^{|\mathcal{V}(t)| \times 16}$**: The 16-dimensional observation vectors received by each active agent.
- **Edge Feature Matrix $\mathbf{E}(t) \in \mathbb{R}^{|\mathcal{E}(t)| \times 3}$**: Continuous spatial relationship vectors between connected drone pairs.

### Dynamic Edge Pruning Invariant

In standard multi-agent benchmarks, communication topologies are often static. In ScrollSwarm, **structural agent crashes trigger immediate topological pruning**:

$$\text{alive}_k(t) = \text{False} \implies \forall i: (k, i) \notin \mathcal{E}(t) \quad \text{and} \quad (i, k) \notin \mathcal{E}(t)$$

Both the incoming and outgoing adjacency rows and columns of agent $k$ are zeroed instantaneously. Any PyG tensor extracted after the crash automatically excludes all severed edges, preserving valid relational computations without stale communication.

---

## 2. Graph Topologies in ScrollSwarm

ScrollSwarm provides five built-in communication topologies via `TopologyType`:

```python
from scrollswarm.comm import TopologyType

# Choose from:
# TopologyType.CHAIN  - Linear neighbor graph (default)
# TopologyType.RING   - Circular cycle graph with wrap-around
# TopologyType.FULL   - All-to-all broadcast baseline
# TopologyType.RADIUS - Distance disk graph (||p_i - p_j|| <= R)
# TopologyType.CAPPED - Degree-capped k-nearest spatial neighbors
```

| Topology | Constructor String | Degree Bound $\Delta$ | Edge Complexity | Research Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Chain** | `"chain"` | $\Delta = 2$ | $E = 2(N - 1)$ | Models linear formation flight; low bandwidth |
| **Ring** | `"ring"` | $\Delta = 2$ | $E = 2N$ | Cycle graph with bidirectional redundancy |
| **Full** | `"full"` | $\Delta = N - 1$ | $E = N(N - 1)$ | Upper bound broadcast baseline (all-to-all) |
| **Spatial Radius** | `"radius"` | Unbounded | Dynamic | Realistic ad-hoc RF connectivity ($d \le R_{\text{comm}}$) |
| **Degree-Capped** | `"capped"` | $\Delta = k$ | $E \le kN$ | Hardware-constrained transceivers (at most $k$ links) |

---

## 3. Extracting PyG-Compatible Graph Tensors

`ScrollSwarmEnv` exposes direct methods for retrieving the communication graph in canonical PyTorch Geometric Coordinate (COO) format:

```python
from scrollswarm import ScrollSwarmEnv
from scrollswarm.comm import TopologyType

env = ScrollSwarmEnv(num_agents=5, comm_topology=TopologyType.RADIUS)
obs, infos = env.reset(seed=42)

# Extract PyG-compatible graph connectivity and edge attributes
edge_index, edge_attr = env.get_edge_index_and_attributes()

print(f"edge_index shape: {edge_index.shape}, dtype: {edge_index.dtype}")
print(f"edge_attr shape:  {edge_attr.shape},  dtype: {edge_attr.dtype}")
# Output:
# edge_index shape: (2, E), dtype: int64
# edge_attr shape:  (E, 3), dtype: float32
```

### Tensor Layout

1. **`edge_index`** (`np.ndarray` of shape `(2, E)`, dtype `int64`):
   - `edge_index[0]`: Array of source drone indices ($j$).
   - `edge_index[1]`: Array of target drone indices ($i$).
   - Conforms strictly to PyG convention: a link from $j$ to $i$ is represented as column `[j, i]^\top`.

2. **`edge_attr`** (`np.ndarray` of shape `(E, 3)`, dtype `float32`):
   Normalized spatial displacement vector for each directed edge $(j \to i)$:
   $$\mathbf{e}_{ji} = \left[ \frac{x_j - x_i}{W}, \quad \frac{y_j - y_i}{H}, \quad \frac{\|\mathbf{p}_j - \mathbf{p}_i\|}{\text{diag}} \right]^\top \in [-1.0, 1.0]^2 \times [0.0, 1.0]$$

---

## 4. Pure NumPy Message Routing Engine

ScrollSwarm includes a built-in message passing engine that requires **zero external deep learning frameworks**. You can route arbitrary continuous latent vectors across the communication graph with permutation-invariant aggregation operators:

```python
import numpy as np

# 1. Each drone generates an M-dimensional latent message vector (e.g., M = 4)
# In practice, this could be target detections, velocity intents, or policy embeddings
messages = {
    f"agent_{i}": np.random.randn(4).astype(np.float32)
    for i in range(5)
}

# 2. Access all agent entities and route messages across active graph links
all_agents = list(env._agent_map.values())

# Aggregation options: "mean", "sum", "max", "concat"
inbox = env.comm_graph.route_messages(
    messages=messages,
    agents=all_agents,
    aggregation="mean",
)

# 3. Each living drone receives an aggregated neighbor message of shape (M,)
for agent_id, aggregated_msg in inbox.items():
    print(f"{agent_id} received context: {aggregated_msg}")
```

### Supported Aggregation Operators

| Operator | Mathematical Formulation | Research Use Case |
| :--- | :--- | :--- |
| **`"mean"`** *(Default)* | $\tilde{\mathbf{m}}_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{m}_j$ | Robust spatial consensus; invariant to node degree |
| **`"sum"`** | $\tilde{\mathbf{m}}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{m}_j$ | Counting detected targets; sensitive to swarm density |
| **`"max"`** | $\tilde{\mathbf{m}}_i = \max_{j \in \mathcal{N}(i)} \mathbf{m}_j$ | Saliency detection (e.g., closest obstacle threat) |
| **`"concat"`** | $\tilde{\mathbf{m}}_i = [\mathbf{m}_{j_1}^\top, \dots, \mathbf{m}_{j_k}^\top]^\top$ | Fixed-degree architectures (e.g., `max_degree = 2`) |

> [!NOTE]
> If a drone has no active in-neighbors ($|\mathcal{N}(i)| = 0$) or is dead, `route_messages` safely assigns a zero-filled vector $\mathbf{0}_M$, preventing undefined numerical behavior or NaNs.

---

## 5. PyTorch Geometric (PyG) Pipeline

When training GNN-conditioned MARL policies (e.g., CommNet, TarMAC, DGN, MAGNet), you can convert the environment's NumPy outputs directly into PyG `Data` objects.

### Packaging Environment Outputs into PyG `Data`

```python
import torch
from torch_geometric.data import Data

def env_step_to_pyg_data(env, observations) -> Data:
    """
    Converts ScrollSwarm observations and communication graph to a PyG Data object.
    """
    # 1. Stack observations for all agents: shape (N, 16)
    # Dead agents are zero-masked to maintain fixed node ordering
    node_features = np.zeros((env._num_agents, 16), dtype=np.float32)
    for i in range(env._num_agents):
        agent_key = f"agent_{i}"
        if agent_key in observations:
            node_features[i] = observations[agent_key]

    # 2. Extract edge connectivity and spatial attributes
    edge_index_np, edge_attr_np = env.get_edge_index_and_attributes()

    # 3. Convert to PyTorch tensors
    x = torch.from_numpy(node_features)                     # (N, 16)
    edge_index = torch.from_numpy(edge_index_np).long()     # (2, E)
    edge_attr = torch.from_numpy(edge_attr_np).float()      # (E, 3)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
```

### Implementing a Graph Convolutional Policy

Below is a minimal GNN policy using PyG's `GCNConv` that combines local observations with aggregated neighbor representations:

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class SwarmGNNPolicy(nn.Module):
    """
    Graph Neural Network policy for cooperative multi-agent UAV control.
    """

    def __init__(self, in_channels: int = 16, hidden_dim: int = 64, num_actions: int = 3):
        super().__init__()
        # Graph convolution layers
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

        # Policy action head (outputs logits for UP, STAY, DOWN)
        self.actor_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over the dynamic communication graph.

        Args:
            x: Node feature tensor of shape (N, 16).
            edge_index: COO graph connectivity tensor of shape (2, E).

        Returns:
            Action logits tensor of shape (N, 3).
        """
        # First message-passing hop
        h = self.relu(self.conv1(x, edge_index))

        # Second message-passing hop (aggregates 2-hop neighbor context)
        h = self.relu(self.conv2(h, edge_index))

        # Action distribution logits
        logits = self.actor_head(h)
        return logits
```

---

## 6. Complete End-to-End Workflow

The following script integrates the entire pipeline: stepping the environment, extracting graph tensors, routing latent messages via the native routing engine, and verifying PyG compatibility.

```python
"""
Tutorial 03: Complete Relational Graph & Message Passing Script.
"""

from typing import Dict
import numpy as np
from scrollswarm import ScrollSwarmEnv
from scrollswarm.comm import TopologyType


def run_gnn_simulation(episodes: int = 1, steps_per_episode: int = 100):
    # 1. Initialize environment with spatial radius communication
    env = ScrollSwarmEnv(
        num_agents=5,
        max_cycles=steps_per_episode,
        comm_topology=TopologyType.RADIUS,  # Dynamic distance-based topology
        target_reward=1.0,
        crash_penalty=10.0,
        step_reward=0.01,
        render_mode=None,
    )

    for ep in range(episodes):
        observations, infos = env.reset(seed=42 + ep)
        print(f"\n--- Episode {ep + 1} Initialized ---")

        for step in range(steps_per_episode):
            if not env.agents:
                print("All drones crashed. Episode ended.")
                break

            # 2. Extract current graph tensors
            edge_index, edge_attr = env.get_edge_index_and_attributes()
            num_active_edges = edge_index.shape[1]

            # 3. Generate sample latent communication messages
            # For demonstration, each drone transmits its relative target features (M = 4)
            latent_messages: Dict[str, np.ndarray] = {}
            for agent_id in env.agents:
                obs = observations[agent_id]
                target_features = obs[8:12]  # [dx/W, dy/H, dist/diag, in_range]
                latent_messages[agent_id] = target_features

            # 4. Route messages using native NumPy routing engine
            all_agents = list(env._agent_map.values())
            inbox = env.comm_graph.route_messages(
                messages=latent_messages,
                agents=all_agents,
                aggregation="mean",
            )

            # 5. Cooperative decision making:
            # Combine local observation with aggregated neighbor context
            actions = {}
            for agent_id in env.agents:
                local_obs = observations[agent_id]
                neighbor_context = inbox[agent_id]

                # If neighbor detected target in range (inbox[3] > 0), climb to support
                if neighbor_context[3] > 0.0:
                    actions[agent_id] = 0  # Action.UP
                else:
                    actions[agent_id] = 1  # Action.STAY

            # 6. Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            if step % 25 == 0:
                alive_count = len(env.agents)
                print(f"Step {step:03d} | Active UAVs: {alive_count}/5 | Graph Edges: {num_active_edges}")

    env.close()
    print("\nGNN simulation workflow completed successfully.")


if __name__ == "__main__":
    run_gnn_simulation()
```

---

## Summary of Tutorials

You have completed all three ScrollSwarm tutorials:

1. [Tutorial 01: PettingZoo ParallelEnv Loop](01_parallel_env.md) — Mastered the synchronous Dec-POMDP multi-agent simulation loop and CTDE state extraction.
2. [Tutorial 02: Heuristic Reactive Policy](02_heuristic_baseline.md) — Implemented an explainable 4-tier rule-based baseline using the 7-ray LiDAR rangefinder.
3. [Tutorial 03: PyG & GNN Integration](03_gnn_pyg.md) — Extracted graph tensors and executed permutation-invariant message passing across dynamic communication networks.
