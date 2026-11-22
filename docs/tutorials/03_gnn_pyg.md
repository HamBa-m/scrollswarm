# Tutorial 03: PyTorch Geometric (PyG) Integration

In Graph Multi-Agent Reinforcement Learning (Graph-MARL), agents exchange latent message embeddings over a dynamic communication topology prior to selecting actions. This tutorial demonstrates how to extract PyTorch Geometric (PyG) graphs from ScrollSwarm.

---

## Extracting Graph Data Tensors

At each timestep, `ScrollSwarmEnv` exposes the communication graph in PyG COO format:

```python
import numpy as np
from scrollswarm import ScrollSwarmEnv

env = ScrollSwarmEnv(num_agents=5, comm_topology="radius", render_mode=None)
obs, infos = env.reset(seed=42)

# 1. Retrieve PyG-compatible graph representations
edge_index, edge_attr = env.get_edge_index_and_attributes()

print(f"Edge Index shape: {edge_index.shape}")  # (2, E)
print(f"Edge Attr shape:  {edge_attr.shape}")   # (E, 3) [dx, dy, normalized_dist]
```

---

## Constructing a PyG `Data` Batch

```python
"""
Tutorial 03: Constructing a PyTorch Geometric Data object from ScrollSwarm.
"""

import numpy as np
import torch
from scrollswarm import ScrollSwarmEnv

# Note: requires 'torch' and 'torch_geometric' installed in your training repo
# from torch_geometric.data import Data

env = ScrollSwarmEnv(num_agents=5, comm_topology="chain")
obs, infos = env.reset(seed=101)

# Step 1: Collect node features X of shape (N, 15)
# In standard MARL, each node feature is the agent's observation vector
node_features = np.stack([obs[agent] for agent in env.possible_agents if agent in env.agents])

# Step 2: Retrieve graph edges and edge features
edge_index_np, edge_attr_np = env.get_edge_index_and_attributes()

# Step 3: Convert to PyTorch tensors
x = torch.from_numpy(node_features).float()
edge_index = torch.from_numpy(edge_index_np).long()
edge_attr = torch.from_numpy(edge_attr_np).float()

print(f"Node Features X:   {x.shape}")           # (N_active, 15)
print(f"Edge Index:         {edge_index.shape}")    # (2, E)
print(f"Edge Attributes:    {edge_attr.shape}")     # (E, 3)

# If using torch_geometric:
# graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
# gnn_output = model(graph_data)
```

---

## Downstream GNN Policy Architectures

With `edge_index` and `edge_attr` available at every step, standard graph message-passing layers (such as `GCNConv`, `GATConv`, or custom Edge-Conditioned Convolutions) can be applied directly:

$$\mathbf{h}_i^{(l+1)} = \sigma\left( \mathbf{W}_0 \mathbf{h}_i^{(l)} + \sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W}_1 \mathbf{h}_j^{(l)} \right)$$

Upstream scout quadcopters passing high obstacle hazards along chain edges allow trailing quadcopters to begin pre-emptive climbing maneuvers before obstacles enter their personal LiDAR range.
