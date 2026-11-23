# Communication Graph API

The `comm` module provides graph communication topologies, dynamic dead-agent edge pruning, PyTorch Geometric (PyG) tensor export, and permutation-invariant message routing engines.

---

## Overview

- **Topologies:** Chain (`chain`), Ring (`ring`), Full Broadcast (`full`), Spatial Radius (`radius`), and Degree-Capped K-Nearest (`capped`).
- **Dynamic Edge Pruning:** Zeroes incoming and outgoing adjacency rows/columns instantaneously when a drone crashes.
- **PyG Tensors:** Exports canonical COO `edge_index` $(2, E)$ and normalized spatial `edge_attr` $(E, 3)$.
- **Routing Engine:** Performs synchronous message aggregation using `mean`, `sum`, `max`, or `concat`.

---

## Quick Usage

```python
import numpy as np
from scrollswarm.comm import CommunicationGraph, TopologyType
from scrollswarm.agent import create_agents

agents = create_agents(num_agents=5, map_width=800, map_height=400)
graph = CommunicationGraph(topology=TopologyType.RADIUS, radius=180.0)

# Extract PyG tensors
edge_index, edge_attr = graph.get_edge_index_and_attributes(agents, map_width=800, map_height=400)
print(f"Active links: {edge_index.shape[1]}")

# Route 8-dimensional latent messages
latent_messages = {i: np.random.randn(8).astype(np.float32) for i in range(5)}
aggregated = graph.route_messages(agents, latent_messages, aggregation="mean")
```

---

## Class & Enum Reference

::: scrollswarm.comm.TopologyType

::: scrollswarm.comm.CommunicationGraph
