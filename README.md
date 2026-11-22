# ScrollSwarm

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9.svg?logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![Documentation](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://hamba-m.github.io/scrollswarm/)
[![PettingZoo](https://img.shields.io/badge/pettingzoo-parallel__env-green.svg)](https://pettingzoo.farama.org/)
[![Tests](https://img.shields.io/badge/tests-44%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A 2D multi-agent reinforcement learning (MARL) testbed for cooperative navigation, graph communication, and target acquisition through dynamic planetary rings.**

![ScrollSwarm Environment Preview](assets/screenshot.png)

ScrollSwarm models an exploration swarm of autonomous rocket probes surfing through an active planetary ring corridor. Probes navigate an environment bounded by radiation sheaths and populated by dynamic polygon asteroids executing Keplerian drift and Hill's epicyclic oscillations to collect targets.

> **Full Documentation & Tutorials:** Detailed mathematical derivations, physics formulations, and API reference are available at [https://hamba-m.github.io/scrollswarm/](https://hamba-m.github.io/scrollswarm/).

---

## Features

- **PettingZoo `ParallelEnv` Standard:** Native Farama multi-agent API compliance with vectorized step transitions, truncation invariants, and CTDE global state vectors.
- **Dynamic Planetary Ring Physics:** Bounded by radiation sheaths ($y_{\text{ceil}} = 15\text{ px}, y_{\text{floor}} = 385\text{ px}$) with irregular polygon asteroids governed by Keplerian drift and Hill's epicyclic vertical oscillations with a guaranteed minimal passage clearance ($G_{\min} \ge 110\text{ px}$).
- **Ego-Centric 7-Ray LiDAR:** Continuous rangefinder ($\mathbf{o}_i \in [0, 1]^{15}$) mapping boundary limits, dynamic asteroid edges (via 2D ray-segment determinant intersections), and targets.
- **Dynamic Graph Communication Layer:** Networked topologies (`chain`, `ring`, `full`, `radius`, `capped`) with real-time dead-agent edge pruning and PyTorch Geometric (PyG) tensor exports (`edge_index`, `edge_attr`).
- **High-Throughput Execution:** Exceeds **670+ environment steps/sec** per CPU core ($\approx 3,400$ agent steps/sec), scaling to over **10,000 steps/sec** in vectorized runners.

---

## Quickstart

ScrollSwarm is managed with [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

### Installation
```bash
git clone https://github.com/HamBa-m/scrollswarm.git
cd scrollswarm
uv sync
```

### Interactive Simulation
```bash
# Launch interactive visualizer (default: 5 probes, chain communication)
uv run python main.py

# Custom swarm size and communication topology
uv run python main.py -n 5 -c chain
uv run python main.py -n 8 -c ring
```

| Key | Action |
| :--- | :--- |
| **`W` / `UP`** | Vertical thrust up ($k = 5\text{ px}$) |
| **`S` / `DOWN`** | Vertical dive down ($k = 5\text{ px}$) |
| **`TAB`** / **`1`–`9`** | Switch active focused probe |
| **`A`** | Move all alive probes synchronously |
| **`V`** | Toggle 7-ray LiDAR beam visualization |
| **`C`** | Toggle graph communication data links |
| **`R`** | Restart simulation & reseed corridor |
| **`SPACE`** | Pause / Resume |
| **`ESC`** | Quit |

### Testing & Docs
```bash
# Run unit test suite (44 tests)
uv run pytest

# Serve local documentation site
uv run mkdocs serve
```

---

## Minimal Stepping Example

```python
import numpy as np
from scrollswarm import ScrollSwarmEnv

# 1. Initialize environment with 5 probes
env = ScrollSwarmEnv(num_agents=5, max_cycles=500, render_mode=None)
observations, infos = env.reset(seed=42)

for step in range(200):
    if not env.agents:
        break

    # 2. Sample actions (0: UP, 1: STAY, 2: DOWN)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    # 3. Step environment
    observations, rewards, terminations, truncations, infos = env.step(actions)

    # 4. Extract PyG-compatible communication graph tensors
    edge_index, edge_attr = env.get_edge_index_and_attributes()

env.close()
```

---

## Documentation Guide

Comprehensive specifications and tutorials are hosted on the [Documentation Site](https://hamba-m.github.io/scrollswarm/):

| Documentation Section | Description |
| :--- | :--- |
| **[Formal Environment Model](https://hamba-m.github.io/scrollswarm/environment/overview/)** | Dec-POMDP-NC mathematical tuple, action space, transition dynamics, and rewards. |
| **[Ring Dynamics & Asteroids](https://hamba-m.github.io/scrollswarm/environment/ring_dynamics/)** | Hill's epicyclic equation derivations, size-velocity scaling, and academic citations. |
| **[Observation & LiDAR](https://hamba-m.github.io/scrollswarm/environment/observation/)** | 7-ray rangefinder geometry and exact 2D determinant intersection mathematics. |
| **[Graph Communication](https://hamba-m.github.io/scrollswarm/environment/communication/)** | Topology specifications, edge pruning invariants, and PyG tensor structures. |
| **[Tutorials](https://hamba-m.github.io/scrollswarm/tutorials/01_parallel_env/)** | Stepping guides, heuristic reactive baselines, and GNN integration. |
| **[API Reference](https://hamba-m.github.io/scrollswarm/api/env/)** | Complete docstrings and signatures for all public classes and methods. |

---

## License

MIT License.
