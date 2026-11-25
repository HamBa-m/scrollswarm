# ScrollSwarm

[![Release](https://img.shields.io/badge/release-v0.2.0-blue.svg?logo=github)](https://github.com/HamBa-m/scrollswarm/releases)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg?logo=c%2B%2B&logoColor=white)](scrollswarm/geometry/cpp/)
[![PettingZoo](https://img.shields.io/badge/pettingzoo-parallel__env-green.svg)](https://pettingzoo.farama.org/)
[![PyG](https://img.shields.io/badge/PyG-Compatible-EE4C2C.svg?logo=pytorch&logoColor=white)](https://hambam.top/scrollswarm/tutorials/03_gnn_pyg/)
[![Acceleration](https://img.shields.io/badge/acceleration-Numba%20%7C%20C%2B%2B20-orange.svg)](https://hambam.top/scrollswarm/api/geometry/)
[![Tests](https://img.shields.io/badge/tests-69%20passed-brightgreen.svg)](tests/)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9.svg?logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://hambam.top/scrollswarm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A 2D multi-agent reinforcement learning (MARL) testbed for cooperative UAV swarm reconnaissance, obstacle avoidance, and graph communication.**

![ScrollSwarm Environment Simulation](assets/demo.gif)

ScrollSwarm is a 2D multi-agent simulation where autonomous drones fly forward through an aerial corridor, dodging obstacles while cooperating to photograph targets along their path:

- **Vertical Flight Control:** Drones cruise forward at a constant speed while each chooses one discrete vertical action per step: **climb up**, **dive down**, or **stay level**.
- **Guaranteed Navigable Obstacles:** 2D obstacles of varying shapes and sizes fill the corridor, while a minimum navigable gap is always preserved so paths are never blocked.
- **Probabilistic Photography & Redundancy:** Photo success is probabilistic—the closer the drone, the higher the chance of a clear shot. When targets sit at different altitudes, the team must spread out so at least one drone captures each target.
- **Forward Perception & Communication:** Each drone scans forward with 7 LiDAR rays for obstacle detection, tracks bearing to the nearest target, and exchanges messages with teammates over a configurable communication graph.
- **Crash Handling:** Hitting any obstacle or corridor boundary causes that drone to crash immediately, while surviving teammates continue the mission.

> **Documentation & API Reference:** Detailed guides and specifications are hosted at [https://hambam.top/scrollswarm/](https://hambam.top/scrollswarm/).

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
# Launch interactive visualizer (default: 5 UAVs, chain communication)
uv run python main.py

# Custom swarm size and communication topology
uv run python main.py -n 5 -c chain
uv run python main.py -n 8 -c ring
```

| Key | Action |
| :--- | :--- |
| **`W` / `UP`** | Vertical climb ($k = 5\text{ px}$) |
| **`S` / `DOWN`** | Vertical dive ($k = 5\text{ px}$) |
| **`TAB`** / **`1`–`9`** | Switch active focused UAV |
| **`A`** | Move all alive UAVs synchronously |
| **`V`** | Toggle 7-ray LiDAR beam visualization |
| **`C`** | Toggle graph communication data links |
| **`R`** | Restart simulation & reseed corridor |
| **`SPACE`** | Pause / Resume |
| **`ESC`** | Quit |

### Testing & Docs
```bash
# Run full unit test suite (69 tests)
uv run pytest

# Build documentation strictly
uv run mkdocs build --strict

# Serve local documentation site
uv run mkdocs serve
```

---

## Benchmark Evaluation Scripts

ScrollSwarm provides standalone, CLI-configurable scripts in `scripts/` for evaluating baselines and recording animated GIFs:

```bash
# 1. Run Random Exploration Baseline (with GIF recording)
uv run python scripts/run_random_policy.py --episodes 1 --max-cycles 300 --record-gif

# 2. Run Reactive Heuristic Controller (with GIF recording)
uv run python scripts/run_heuristic_policy.py --episodes 1 --max-cycles 300 --record-gif

# 3. Multi-seed comparative benchmarking (5 episodes, no recording)
uv run python scripts/run_heuristic_policy.py --episodes 5 --max-cycles 500 --no-record-gif

# 4. Multi-backend computational acceleration benchmark
uv run python scripts/benchmark_backends.py --steps 500 --num-agents 4

# 5. Multi-core vectorized scaling benchmark (Sync vs Async across E corridors)
uv run python scripts/benchmark_vector.py --steps 400 --num-agents 4
```

### Empirical Benchmark Baseline (500 Steps, 5 Seeds)

Over 5 seeds (500 steps each), the reactive LiDAR heuristic captures **$3.0\times$ more targets** ($1.80$ vs. $0.60$), survives **$38.3\%$ longer** ($356.7$ vs. $257.9$ steps), and reduces crashes by $24$ percentage points ($60\%$ vs. $84\%$) compared to random actions. See [Tutorial 02: Heuristic Reactive Policy](https://hambam.top/scrollswarm/tutorials/02_heuristic_baseline/) for the full benchmark table and analysis.

### Multi-Backend Computational Acceleration (NumPy vs. C++ vs. Numba)

Profiling revealed that **79.3%** of raw Python stepping is consumed by 2D LiDAR raycasting and polygon intersection math. ScrollSwarm provides a zero-overhead hardware abstraction layer with three interchangeable backends:

- **`numpy` (Baseline):** 100% portable pure Python reference (zero external compiler or build dependencies).
- **`numba` (LLVM CPU JIT):** Runtime JIT compilation via `@njit(fastmath=True, nogil=True)` for $50\times$ faster raycasting.
- **`cpp` (Native C++20 Kernel):** Pre-compiled header-only C++20 geometry engine via `pybind11` with static runtime linkage and zero JIT warmup lag.
- **`auto` (Default):** Dynamically detects and dispatches to the highest-performing available backend (`cpp` $\to$ `numba` $\to$ `numpy`).

#### Head-to-Head Performance (4 UAVs, 500 Steps, Seed 42)

Both accelerated kernels cut isolated raycasting latency by **$33.4\times$ (C++20: $6.41\,\mu\text{s}$)** to **$50.1\times$ (Numba: $4.28\,\mu\text{s}$)** over NumPy ($214.24\,\mu\text{s}$), boosting end-to-end stepping from $691.4\text{ steps/s}$ to $1,652\text{--}1,740\text{ steps/s}$ ($2.39\times\text{--}2.52\times$ speedup; C++20 has zero JIT warmup). See [Tutorial 04: Computational Geometry Acceleration](https://hambam.top/scrollswarm/tutorials/04_acceleration_backends/) for full latency tables, scaling sweeps, and hardware specs.

```bash
# Compile native C++ geometry kernel locally (one-time build)
uv run python scripts/build_geometry_cpp.py

# Run head-to-head acceleration benchmark
uv run python scripts/benchmark_backends.py --steps 500 --num-agents 4
```

---

## Tutorials & Walkthroughs

Step-by-step guides are hosted on the [Documentation Site](https://hambam.top/scrollswarm/):

1. **[Tutorial 01: PettingZoo ParallelEnv Loop](https://hambam.top/scrollswarm/tutorials/01_parallel_env/):** The synchronous Dec-POMDP multi-agent simulation loop, handling agent life/death culling, and CTDE global state extraction (`env.state()`).
2. **[Tutorial 02: Heuristic Reactive Policy](https://hambam.top/scrollswarm/tutorials/02_heuristic_baseline/):** Designing an explainable 4-tier rule-based baseline using the 7-ray LiDAR rangefinder and target tracking.
3. **[Tutorial 03: PyG & GNN Integration](https://hambam.top/scrollswarm/tutorials/03_gnn_pyg/):** Extracting graph tensors (`edge_index`, `edge_attr`), packaging into `torch_geometric.data.Data`, and executing native message passing.
4. **[Tutorial 04: Computational Geometry Acceleration](https://hambam.top/scrollswarm/tutorials/04_acceleration_backends/):** Compiling the native C++20 kernel, enabling Numba JIT, invoking standalone geometry kernels, and benchmarking swarm scaling ($N \in [1, 10]$).
5. **[Tutorial 05: Vectorized MARL & Multi-Core Scaling](https://hambam.top/scrollswarm/tutorials/05_vectorized_marl/):** Running parallel `(E, N)` corridors with `SyncVectorScrollSwarm` / `AsyncVectorScrollSwarm`, auto-reset bootstrapping, and block-diagonal PyG graph batching.

---

## Environment Specification Summary

| Component | Formal Representation | Dimensions & Types |
| :--- | :--- | :--- |
| **Observation Space** | $\mathbf{o}_i \in [-1.0, 1.0]^{16}$ | `Box(-1.0, 1.0, (16,), float32)`: Altitude ($1$), LiDAR ($7$), Target ($4$), Neighbor ($4$) |
| **Action Space** | $a_i \in \{0, 1, 2\}$ | `Discrete(3)`: `0: UP (-5px)`, `1: STAY (0px)`, `2: DOWN (+5px)` |
| **Centralized State** | $\mathbf{s} \in [-1.0, 1.0]^{16N}$ | `Box(-1.0, 1.0, (16*N,), float32)`: Concatenated observations (zero-masked on crash) |
| **Graph Topology** | $\mathcal{G}(t) = (\mathcal{V}(t), \mathcal{E}(t))$ | PyG COO `edge_index` $(2, E)$ & spatial `edge_attr` $(E, 3)$ with dead-agent pruning |
| **Rewards** | $r_i \in \mathbb{R}$ | Photo ($+1.0$), Step survival ($+0.01$), Collision crash ($-10.0$) |
| **Compute Backends** | Pluggable Strategy | `backend="auto" | "cpp" | "numba" | "numpy"`: $33\times\text{--}50\times$ isolated raycasting speedup |

---

## Minimal Stepping Example

```python
import numpy as np
from scrollswarm import ScrollSwarmEnv

# 1. Initialize environment with 5 UAVs
env = ScrollSwarmEnv(num_agents=5, max_cycles=500, render_mode=None)
observations, infos = env.reset(seed=42)

for step in range(200):
    if not env.agents:
        break

    # 2. Sample discrete actions (0: UP, 1: STAY, 2: DOWN)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    # 3. Step environment
    observations, rewards, terminations, truncations, infos = env.step(actions)

    # 4. Extract PyG-compatible communication graph tensors
    edge_index, edge_attr = env.get_edge_index_and_attributes()

env.close()
```

---

## Author & Citation

**Hamza Ba-mohammed**  
College of Computing (CC), Mohammed VI Polytechnic University (UM6P), Rabat, Morocco  
Email: [hamza.bamohammed@um6p.ma](mailto:hamza.bamohammed@um6p.ma)

If you use ScrollSwarm in your research, please cite:

```bibtex
@software{scrollswarm2026,
  author = {Ba-mohammed, Hamza},
  title = {ScrollSwarm: A 2D Multi-Agent Reinforcement Learning Testbed for Cooperative UAV Swarm Reconnaissance},
  url = {https://github.com/HamBa-m/scrollswarm},
  year = {2026}
}
```

---

## License

MIT License.
