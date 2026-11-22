# ScrollSwarm: Multi-Agent RL Saturn Ring Surfing Testbed

[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PettingZoo Compatible](https://img.shields.io/badge/PettingZoo-ParallelEnv-green.svg)](https://pettingzoo.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](https://github.com/HamBa-m/scrollswarm)

**ScrollSwarm** is a side-scrolling multi-agent reinforcement learning (MARL) testbed engineered for evaluating decentralized coordination, graph communication, and spatial navigation through dynamic planetary rings under strict obstacle constraints.

![ScrollSwarm Simulation Preview](assets/screenshot.png)

---

## Key Research Highlights

- **PettingZoo `ParallelEnv` Standard Compliance:** Fully adheres to Farama Foundation's MARL standards with step transitions, truncation invariants, and CTDE global state vectors.
- **Dynamic Planetary Ring Environment:** Navigational corridor bounded by upper and lower radiation sheaths, populated by irregular polygon asteroids executing Keplerian drift and Hill's epicyclic vertical oscillations with mathematically guaranteed passage clearance ($G(x) \ge 110\text{ px}$).
- **Ego-Centric 7-Ray LiDAR Sensory Suite:** Unified continuous rangefinder ($\mathbf{o}_i \in [0, 1]^{15}$) mapping upcoming boundary sheaths, dynamic polygon asteroid edges via exact 2D determinant ray-segment intersection, and collection targets.
- **Dynamic Communication Graph Layer:** Flexible topology modeling (`chain`, `ring`, `full`, `radius`, `capped`) with real-time dead-agent edge pruning and native PyTorch Geometric (PyG) tensor exports (`edge_index`, `edge_attr`).
- **High-Throughput Headless Execution:** Capable of exceeding **670+ environment steps/sec** per single CPU core ($\approx 3,400$ agent steps/sec), scaling to over **10,000 steps/sec** in standard vectorized rollout runners.

---

## Architecture Overview

```mermaid
graph LR
    subgraph Environment ["ScrollSwarm Environment (PettingZoo)"]
        Map["Procedural Saturn Ring Corridor<br/>(Dynamic Asteroids & Sheath Boundaries)"]
        Agents["Rocket Probes 0..N-1<br/>(k = 5px Dynamics)"]
        LiDAR["7-Ray LiDAR Sensor<br/>(Asteroids & Targets)"]
        Graph["Communication Graph<br/>(Dynamic Edge Pruning)"]
    end

    subgraph Observations ["Agent Interfaces"]
        Obs["Decentralized Obs o_i ∈ [0, 1]¹⁵<br/>(LiDAR + Vertical Elevation)"]
        GlobalState["Centralized State s ∈ [0, 1]¹⁵ᴺ<br/>(CTDE Critic Vector)"]
        PyGData["PyG Graph Tensors<br/>(edge_index, edge_attr)"]
    end

    subgraph MARL ["Downstream Algorithms"]
        Actors["Decentralized Policies<br/>(Actor / GNN)"]
        Critic["Centralized Value Critic<br/>(MAPPO / QMIX)"]
    end

    Map --> LiDAR
    Agents --> LiDAR
    Agents --> Graph
    LiDAR --> Obs
    Obs --> GlobalState
    Graph --> PyGData
    Obs --> Actors
    GlobalState --> Critic
    PyGData --> Actors
```

---

## Research Questions Explored

1. **Single-Agent Survival:** Can an ego-centric rocket probe learn reactive vertical navigation from 7-ray LiDAR rangefinders without global coordinates?
2. **Decentralized Flocking:** How do uncoordinated probes avoid crowding and catastrophic bottlenecks around approaching massive moonlets?
3. **Graph-Assisted Predictive Signaling:** Can upstream scout probes relay advance warnings (approaching fast shards, dense asteroid clusters, targets) to downstream teammates along a graph topology?
4. **Topology Resilience:** How does team performance degrade when communication links fail or probes are dynamically eliminated?
