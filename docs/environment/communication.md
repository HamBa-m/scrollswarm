# Communication Graph & Network Topologies

This page details the mathematical graph communication layer in ScrollSwarm: formal graph topology formulations, dynamic dead-agent edge pruning invariants, PyTorch Geometric (PyG) tensor exports, and message passing routing engines.

---

## 1. Graph Communication Model

In multi-agent reinforcement learning, decentralized agents often need to coordinate beyond their local field of view by sharing intermediate representations or messages ([Sukhbaatar et al., 2016](https://proceedings.neurips.cc/paper/2016/hash/55b1927f76d59c74abf742442223c300-Abstract.html); [Das et al., 2019](https://proceedings.mlr.press/v97/das19a.html)).

In ScrollSwarm, the communication network at time $t$ is represented as a directed graph:

$$\mathcal{G}(t) = \big( \mathcal{V}(t), \;\mathcal{E}(t) \big)$$

Where:

- $\mathcal{V}(t) \subseteq \{1, \dots, N\}$ is the active node set consisting of all living UAVs:

$$\mathcal{V}(t) = \big\{ i \in \{1, \dots, N\} \;\big|\; \text{alive}_i(t) = \text{True} \big\}$$

- $\mathcal{E}(t) \subseteq \mathcal{V}(t) \times \mathcal{V}(t)$ is the directed edge set, where $(j, i) \in \mathcal{E}(t)$ denotes an active communication channel from source UAV $j$ to target UAV $i$.
- $\mathbf{A}(t) \in \{0, 1\}^{N \times N}$ is the binary adjacency matrix where $A_{ij}(t) = 1 \iff (j, i) \in \mathcal{E}(t)$.

---

## 2. Supported Topologies & Mathematical Formulations

ScrollSwarm provides five canonical graph communication topologies:

### 2.1 Chain Topology (`TopologyType.CHAIN`, Default)
Models a sequential line-of-sight daisy-chain along horizontal station-keeping columns:

$$A_{ij} = \begin{cases} 1 & \text{if } |i - j| = 1 \\ 0 & \text{otherwise} \end{cases}$$

- Each internal UAV communicates exclusively with its immediate left and right neighbors ($i-1$ and $i+1$).
- Maximum node degree: $\Delta = 2$.

### 2.2 Ring Topology (`TopologyType.RING`)
A 1D periodic cycle graph connecting adjacent neighbors with wrap-around between the first and last drones:

$$A_{ij} = \begin{cases} 1 & \text{if } (i - j) \equiv \pm 1 \pmod N \\ 0 & \text{otherwise} \end{cases}$$

- Eliminates boundary edge effects present in open chains.
- Every node has degree precisely 2 (2-regular graph).

### 2.3 Full Broadcast Topology (`TopologyType.FULL`)
A complete directed graph representing global, unconstrained broadcast communication (all-to-all):

$$A_{ij} = \begin{cases} 1 & \text{if } i \neq j \\ 0 & \text{if } i = j \end{cases}$$

- Number of edges: $E = N(N - 1)$.
- Serves as the theoretical upper bound for collaborative information exchange.

### 2.4 Spatial Radius Disk Graph (`TopologyType.RADIUS`)
A geometric random graph where a communication link exists if and only if the Euclidean distance between two drones is within communication radius $R_{\text{comm}}$:

$$A_{ij} = \begin{cases} 1 & \text{if } 0 < \|\mathbf{p}_i - \mathbf{p}_j\| \le R_{\text{comm}} \\ 0 & \text{otherwise} \end{cases}$$

By default, $R_{\text{comm}} = 180.0\text{ px}$. Links dynamically form and break as drones climb and dive.

### 2.5 Degree-Capped Spatial Graph (`TopologyType.CAPPED`)
Connects each UAV to its $k$-nearest spatial neighbors, enforcing a strict bandwidth/degree cap ($k = \text{max_degree}$, default: $2$):

$$\mathcal{N}_k(i) = \arg\min_{\mathcal{S} \subseteq \mathcal{V} \setminus \{i\}, |\mathcal{S}| \le k} \sum_{j \in \mathcal{S}} \|\mathbf{p}_i - \mathbf{p}_j\|$$

$$A_{ij} = \begin{cases} 1 & \text{if } j \in \mathcal{N}_k(i) \\ 0 & \text{otherwise} \end{cases}$$

---

## 3. Dynamic Dead-Agent Edge Pruning Invariant

In real-world swarm robotics, disabled or crashed agents cannot transmit or receive radio messages. ScrollSwarm enforces strict **dynamic edge pruning** ([Oliehoek & Amato, 2016](https://doi.org/10.1007/978-3-319-28929-8)):

### Invariant
For all pairs $(i, j) \in \{1, \dots, N\}^2$ and all time steps $t$:

$$A_{ij}(t) = A_{ij}^{\text{base}} \cdot \mathbf{1}_{\text{alive}_i(t)} \cdot \mathbf{1}_{\text{alive}_j(t)}$$

If drone $k$ crashes ($\text{alive}_k = \text{False}$), row $k$ and column $k$ of $\mathbf{A}(t)$ are zeroed immediately:

$$\sum_{j=1}^N A_{kj}(t) = 0 \quad \text{and} \quad \sum_{i=1}^N A_{ik}(t) = 0$$

All incident edges vanish from the graph, preventing zombie signal propagation in Graph Neural Networks.

---

## 4. PyTorch Geometric (PyG) Tensor Integration

To facilitate direct integration with Graph Neural Network (GNN) architectures ([Fey & Lenssen, 2019](https://arxiv.org/abs/1903.02428)), `env.get_edge_index_and_attributes()` extracts graph connectivity in canonical PyG format:

```python
edge_index, edge_attr = env.get_edge_index_and_attributes()
```

### 4.1 `edge_index` Tensor: Shape `(2, E)`
- Conforms to PyTorch Geometric COO edge indexing:
  - `edge_index[0]`: Source node indices ($j$)
  - `edge_index[1]`: Target node indices ($i$)
- Data type: `np.int64` (transfers directly to `torch.long`).

### 4.2 `edge_attr` Tensor: Shape `(E, 3)`
Provides normalized relative spatial edge features for every active directed link $(j \to i)$:

$$\mathbf{e}_{ji} = \left[ \frac{x_j - x_i}{W}, \quad \frac{y_j - y_i}{H}, \quad \frac{\|\mathbf{p}_j - \mathbf{p}_i\|}{\sqrt{W^2 + H^2}} \right]^\top \in [-1.0, 1.0]^2 \times [0.0, 1.0]$$

Data type: `np.float32` (transfers directly to `torch.float32`).

---

## 5. Message Passing & Routing Engine

`CommunicationGraph.route_messages()` implements synchronous inter-agent message passing:

$$\tilde{\mathbf{m}}_i = \bigoplus_{j \in \mathcal{N}_{\text{in}}(i)} \mathbf{m}_j$$

Where $\mathbf{m}_j \in \mathbb{R}^M$ is the message emitted by agent $j$, and $\bigoplus$ is a permutation-invariant aggregation operator:

| Aggregation Operator | Mathematical Formulation | Use Case |
| :---: | :---: | :--- |
| **`"mean"`** (Default) | $\tilde{\mathbf{m}}_i = \frac{1}{\|\mathcal{N}(i)\|} \sum_{j \in \mathcal{N}(i)} \mathbf{m}_j$ | Normalized continuous latent feature averaging (CommNet) |
| **`"sum"`** | $\tilde{\mathbf{m}}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{m}_j$ | Cumulative counting and target observation accumulation |
| **`"max"`** | $\tilde{\mathbf{m}}_i = \max_{j \in \mathcal{N}(i)} \mathbf{m}_j$ | Max-pooling saliency filtering |
| **`"concat"`** | $\tilde{\mathbf{m}}_i = \big[ \mathbf{m}_{j_1}^\top, \dots, \mathbf{m}_{j_k}^\top \big]^\top$ | Fixed-degree ordered concatenation |

---

## 6. References

1. **Sukhbaatar, S., Szlam, A., & Fergus, R. (2016).** *Learning multiagent communication with backpropagation.* In Advances in Neural Information Processing Systems (NeurIPS), pp. 2244–2252. [NeurIPS Link](https://proceedings.neurips.cc/paper/2016/hash/55b1927f76d59c74abf742442223c300-Abstract.html).
2. **Das, A., Gervet, T., Romoff, J., Batra, D., Parikh, D., Rabbat, M., & Pineau, J. (2019).** *TarMAC: Targeted multi-agent communication.* In International Conference on Machine Learning (ICML), pp. 1538–1546. [PMLR Link](https://proceedings.mlr.press/v97/das19a.html).
3. **Jiang, J., Dun, C., Huang, T., & Lu, Z. (2020).** *Graph convolutional reinforcement learning.* In International Conference on Learning Representations (ICLR). [OpenReview Link](https://openreview.net/forum?id=HkxdQkSYDB).
4. **Fey, M., & Lenssen, J. E. (2019).** *Fast graph representation learning with PyTorch Geometric.* ICLR Workshop on Representation Learning on Graphs and Manifolds. [arXiv: 1903.02428](https://arxiv.org/abs/1903.02428).
5. **Oliehoek, F. A., & Amato, C. (2016).** *A Concise Introduction to Decentralized POMDPs.* SpringerBriefs in Intelligent Systems, Springer. [DOI: 10.1007/978-3-319-28929-8](https://doi.org/10.1007/978-3-319-28929-8).
