# Graph Communication & Topologies

ScrollSwarm includes an integrated communication layer (`scrollswarm.CommunicationGraph`) designed for Graph Neural Networks (GNNs) and networked MARL.

---

## Supported Topologies

| Topology | Key | Graph Definition | Research Utility |
| :--- | :---: | :--- | :--- |
| **Chain** (Default) | `chain` | Path graph $P_N$: $A_{ij} = 1 \iff |i - j| = 1$ | Tests predictive communication from upstream scout probes to downstream followers. |
| **Ring** | `ring` | Cycle graph $C_N$: Chain plus wrap-around link $(0 \leftrightarrow N-1)$ | Symmetric circulating message passing benchmark. |
| **Full** | `full` | Complete graph $K_N$: $A_{ij} = 1, \; \forall i \neq j$ | All-to-all broadcast upper bound. |
| **Radius** | `radius` | Spatial $\epsilon$-ball: $A_{ij} = 1 \iff \|\mathbf{p}_i - \mathbf{p}_j\| \le R_{\text{comm}}$ | Distance-limited local ad-hoc wireless mesh. |
| **Capped** | `capped` | Degree-bounded: $k$-nearest spatial neighbors ($k \le \text{max\_degree}$) | Bandwidth-constrained coordination. |

---

## Dynamic Edge Pruning Invariant

Communication requires both transmitter and receiver to be operational:

$$A_{ij}^{(t)} = 1 \implies (\text{alive}_i = \text{True}) \;\land\; (\text{alive}_j = \text{True})$$

When an agent terminates due to collision, all incident incoming and outgoing edges are pruned immediately:

$$\forall j \in \mathcal{N}: \quad A_{ik} = A_{ki} = 0 \quad \text{if agent } i \text{ terminates}$$

This prevents ghost message delivery and simulates link dropouts caused by node failure.

---

## PyTorch Geometric (PyG) Tensor Extraction

The environment exposes graph structures directly formatted for PyTorch Geometric GNN models:

```python
# edge_index: (2, E) uint64 / int64
# edge_attr:  (E, 3) float32 [dx, dy, normalized_distance]
edge_index, edge_attr = env.get_edge_index_and_attributes()
```

- **`edge_index`:** Standard PyG COO format indicating source and target agent indices.
- **`edge_attr`:** 3-dimensional edge feature vector containing relative displacement $(\Delta x, \Delta y)$ and normalized Euclidean separation $\frac{\|\Delta \mathbf{p}\|}{R_{\text{comm}}}$.
