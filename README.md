# ScrollSwarm

[![Release](https://img.shields.io/badge/release-v0.1.0-blue.svg?logo=github)](https://github.com/HamBa-m/scrollswarm/releases)
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

ScrollSwarm is a 2D multi-agent simulation where a team of drones flies forward through an aerial corridor, dodging obstacles of different shapes and sizes while working together to take photos of targets along the way.

> **Documentation & API Reference:** Detailed guides and specifications are hosted at [https://hambam.top/scrollswarm/](https://hambam.top/scrollswarm/).

---

## Scenario Description

ScrollSwarm models a swarm of autonomous UAVs (drones) flying forward through an aerial corridor, where the vertical axis represents their flight altitude:

- **Movement & Controls:** The drones cruise forward in a straight line at a steady speed. Each drone simply chooses its vertical action at each step: **climb up**, **dive down**, or **maintain current altitude**.
- **Obstacles of Different Sizes:** 2D obstacles of varying shapes and sizes are distributed across the flight path. An open flight passage is always guaranteed through the corridor so drones never encounter blocked paths.
- **Observation & Target Missions:** Drones have the mission of observing specific target points along the way by flying close enough to photograph them.
- **Photo Quality & Redundancy:** Capturing a clear photo isn't guaranteed every time—we model image clarity with a probability that increases the closer a drone gets to the target. When targets are scattered across different altitudes, the team must coordinate to cover the entire space and provide redundancy, ensuring at least one drone successfully captures each target with sufficient quality.
- **Forward Perception & Communication:** Each drone uses 7 forward rangefinder rays (LiDAR) to spot oncoming obstacles ahead, tracks relative directions to nearby targets, and can share messages with teammates across configurable network graphs (chain, ring, spatial radius, or broadcast).
- **Safety:** Hitting any obstacle or corridor boundary immediately disables that drone, while the remaining team members continue the mission.

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

| Metric | Random Action Baseline | Reactive Swarm Policy | Performance Delta |
| :--- | :---: | :---: | :--- |
| **Mean Episodic Return** | $-28.55 \pm 10.50$ | **$-10.39 \pm 18.00$** | $+18.16$ reduction in crash penalties |
| **Mean Survival Horizon** | $257.9 \pm 66.8\text{ steps}$ | **$356.7 \pm 75.7\text{ steps}$** | $+98.8\text{ steps}$ ($+38.3\%$ longer flight time) |
| **Mean Targets Photographed** | $0.60 \pm 0.49$ | **$1.80 \pm 0.75$** | $\mathbf{3.0\times}$ increase in mission yield |
| **Drone Crash Rate** | $84.0\%$ ($21/25$ dead) | **$60.0\%$** ($15/25$ dead) | $24.0\text{ percentage point}$ reduction |
| **Inference Latency** | $\sim 0.002\text{ ms}$ | $\sim 0.013\text{ ms}$ | Negligible CPU overhead |

### Multi-Backend Computational Acceleration (NumPy vs. C++ vs. Numba)

Profiling revealed that **79.3%** of raw Python stepping is consumed by 2D LiDAR raycasting and polygon intersection math. ScrollSwarm provides a zero-overhead hardware abstraction layer with three interchangeable backends:

- **`numpy` (Baseline):** 100% portable pure Python reference (zero external compiler or build dependencies).
- **`numba` (LLVM CPU JIT):** Runtime JIT compilation via `@njit(fastmath=True, nogil=True)` for $50\times$ faster raycasting.
- **`cpp` (Native C++20 Kernel):** Pre-compiled header-only C++20 geometry engine via `pybind11` with static runtime linkage and zero JIT warmup lag.
- **`auto` (Default):** Dynamically detects and dispatches to the highest-performing available backend (`cpp` $\to$ `numba` $\to$ `numpy`).

#### Head-to-Head Performance (4 UAVs, 500 Steps, Seed 42)

| Metric / Characteristic | 1. NumPy Baseline | 2. Native C++20 Kernel | 3. Numba LLVM JIT |
| :--- | :---: | :---: | :---: |
| **LiDAR Raycasting Latency** | $214.24\,\mu\text{s}$ / call | **$6.41\,\mu\text{s}$ / call** | **$4.28\,\mu\text{s}$ / call** |
| **Isolated Raycasting Speedup** | $1.00\times$ (reference) | **$33.44\times$ faster** | **$50.08\times$ faster** |
| **End-to-End Stepping Throughput** | $691.4\text{ steps/s}$ ($1.446\text{ ms}$) | **$1,652.3\text{ steps/s}$ ($0.605\text{ ms}$)** | **$1,740.2\text{ steps/s}$ ($0.575\text{ ms}$)** |
| **Full Environment Speedup** | $1.00\times$ | **$2.39\times$ faster** | **$2.52\times$ faster** |
| **First-Step JIT Compilation Lag** | $0.0\text{ s}$ | **$0.0\text{ s}$ (instant)** | $\approx 1.5\text{ s}$ (LLVM compile) |
| **Build / Tool Requirements** | None | GCC / Clang / MSVC | `uv sync --extra numba` |

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
