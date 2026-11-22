# ScrollSwarm

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9.svg?logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![Documentation](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://hamba-m.github.io/scrollswarm/)
[![PettingZoo](https://img.shields.io/badge/pettingzoo-parallel__env-green.svg)](https://pettingzoo.farama.org/)
[![Tests](https://img.shields.io/badge/tests-44%20passed-brightgreen.svg)](tests/)
[![Status](https://img.shields.io/badge/status-active%20research-informational.svg)](README.md)

**A side-scrolling multi-agent reinforcement learning (MARL) testbed for cooperative navigation, local graph communication, and objective acquisition through dynamic planetary rings.**

![ScrollSwarm Environment Preview](assets/screenshot.png)

ScrollSwarm models an exploration swarm of autonomous rocket probes surfing through an active planetary ring corridor (such as Saturn's main rings). The corridor is bounded by upper and lower radiation sheath boundary planes and populated by dynamic, multi-scale polygon asteroids and icy moonlets exhibiting orbital-plane perturbations.

The environment serves as a rigorous, visualizable benchmark for studying how agents coordinate under **ego-centric partial observability** (forward LiDAR rangefinders) and **dynamic graph communication** (networked message passing with real-time link pruning on node failure).

---

## Key Research Highlights

- **PettingZoo `ParallelEnv` Standard Compliance:** Fully adheres to Farama Foundation's multi-agent standard with simultaneous stepping, proper truncation handling, and Centralized Training with Decentralized Execution (CTDE) global state vectors.
- **Dynamic Planetary Ring Astrodynamics:** Modeled with Keplerian horizontal drift and Hill's epicyclic vertical oscillations, preserving a mathematically guaranteed minimal passage clearance ($G_{\min} \ge 110\text{ px}$).
- **Ego-Centric 7-Ray LiDAR Sensory Suite:** Unified continuous rangefinder ($\mathbf{o}_i \in [0, 1]^{15}$) mapping boundary limits, dynamic polygon asteroid edges via exact 2D determinant ray-segment intersection, and Cassini telemetry beacons.
- **Dynamic Communication Graph Layer:** Flexible topology modeling (`chain`, `ring`, `full`, `radius`, `capped`) with real-time dead-agent edge pruning and native PyTorch Geometric (PyG) tensor exports (`edge_index`, `edge_attr`).
- **High-Throughput Headless Execution:** Capable of exceeding **670+ environment steps/sec** per single CPU core ($\approx 3,400$ agent steps/sec), scaling to over **10,000 steps/sec** in standard vectorized rollout runners.

---

## Quickstart

ScrollSwarm uses [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency and environment management.

### Installation
```bash
# Clone the repository
git clone https://github.com/HamBa-m/scrollswarm.git
cd scrollswarm

# Install dependencies and sync environment
uv sync
```

### Run Interactive Simulation
```bash
# Launch interactive desktop simulator (default: 5 probes, chain communication)
uv run python main.py

# Launch with custom swarm size (1 to 10) and graph topology
uv run python main.py -n 5 -c chain
uv run python main.py -n 8 -c ring
```

### Interactive Controls
| Key | Action |
| :--- | :--- |
| **`W` / `UP`** | Vertical thrust up ($k = 5\text{ px}$) |
| **`S` / `DOWN`** | Vertical dive down ($k = 5\text{ px}$) |
| **`TAB`** / **`1`–`9`** | Switch active focused probe (highlighted with dual reticle) |
| **`A`** | Command all alive probes synchronously |
| **`V`** | Toggle 7-ray LiDAR beam visualization |
| **`C`** | Toggle graph communication data links |
| **`R`** | Restart simulation & reseed ring corridor |
| **`SPACE`** | Pause / Resume |
| **`ESC`** | Exit |

### Run Test Suite
```bash
uv run pytest
```

### Documentation Site
Build and view the documentation, tutorials, and API reference locally:
```bash
# Serve live-reloading documentation in your browser
uv run mkdocs serve

# Build static production HTML (strict mode)
uv run mkdocs build --strict
```

---

## Ring Astrodynamics & Dynamic Asteroid Kinematics

Rather than static or sinusoidal terrain, ScrollSwarm models an exploration corridor bounded by upper and lower radiation sheath planes ($y_{\text{ceil}} = 15\text{ px}$, $y_{\text{floor}} = 385\text{ px}$) and populated by dynamic polygon asteroids.

### 1. Hill's Epicyclic Vertical Oscillation
In a local rotating coordinate frame centered on a reference Keplerian orbit with orbital mean motion $n$, the linearized out-of-plane relative motion decouples as a simple harmonic oscillator (**Hill's equation**):

$$\ddot{z} + n^2 z = 0 \implies z(t) = z_0 + A \sin(n t + \phi)$$

In ScrollSwarm, each asteroid's vertical position updates according to this exact analytical solution:

$$y(t) = y_{\text{base}} + A(R) \sin\big( \omega(R) \cdot t + \phi \big)$$

where $A(R) \le 30\text{ px}$ is the vertical amplitude, $\omega(R)$ is the size-dependent epicyclic frequency, and $\phi \sim \mathcal{U}(0, 2\pi)$ is the initial orbital inclination phase.

### 2. Size-Velocity Scaling & Thermal Equipartition
In dense collisional ring systems, mutual velocity dispersion is governed by energy exchange toward thermal equipartition:

$$\frac{1}{2} m \sigma_v^2 \approx \text{const} \implies \sigma_v \propto m^{-1/2} \propto R^{-1}$$

Smaller icy shards possess higher velocity dispersions, drifting and oscillating substantially faster than massive moonlets:

$$v_x(R) = v_0 + \Delta v \left( \frac{R_{\max} - R}{R_{\max} - R_{\min}} \right) \in [1.0, 2.8]\text{ px/step}$$

### 3. Guaranteed Minimal Passage Invariant
To ensure the environment remains solvable and fair for learning agents, procedural asteroid generation enforces a minimal clearance invariant across the corridor at all times:

$$\text{Clearance}_{\min} \ge 110\text{ px}$$

Because rocket probes have a collision height of $h_{\text{probe}} = 14\text{ px}$, this invariant mathematically guarantees that at least one unobstructed flight path exists across every column of the screen even at peak epicyclic amplitudes.

---

## Perception & Sensory Model

Each rocket probe perceives oncoming hazards through an ego-centric **7-ray forward LiDAR rangefinder** projecting across a $120^\circ$ field of view ($R_{\max} = 400\text{ px}$):

```text
               \   Ray 0: -60° (Ceiling Sheath / Upper Asteroids)
                \  Ray 1: -40°
                 \ Ray 2: -20°
Probe [>] ======== Ray 3:   0° (Corridor Centerline / Oncoming Shards)
                 / Ray 4: +20°
                /  Ray 5: +40°
               /   Ray 6: +60° (Floor Sheath / Lower Asteroids)
```

### Observation Vector & Global Critic State

At every timestep $t$, each active probe receives a continuous 15-dimensional observation vector:

$$\mathbf{o}_i = \left[ \frac{y_i}{H}, \; d_{\text{obs}, 0}, \dots, d_{\text{obs}, 6}, \; d_{\text{tgt}, 0}, \dots, d_{\text{tgt}, 6} \right] \in [0, 1]^{15}$$

- **Normalized Elevation:** Probe vertical position $y_i / H \in [0, 1]$.
- **Obstacle Distances ($d_{\text{obs}, k}$):** Normalized distance along ray $k$ to the nearest sheath boundary or asteroid edge segment (computed via exact 2D determinant intersection with bounding circle rejection). $1.0$ indicates an entirely clear path.
- **Target Distances ($d_{\text{tgt}, k}$):** Normalized distance to the nearest Cassini telemetry beacon along ray $k$ ($1.0$ = no target).

For centralized critics in CTDE frameworks (e.g., MAPPO, QMIX), `env.state()` concatenates individual observations:

$$\mathbf{s} = \big[ \mathbf{o}_0^\top, \mathbf{o}_1^\top, \dots, \mathbf{o}_{N-1}^\top \big]^\top \in [0, 1]^{15 \cdot N}$$

Dead probes contribute a zero vector of dimension 15, preserving fixed tensor dimensionality.

### Reward Formulation

- **Beacon Acquisition:** $+1.0$ per telemetry target collected.
- **Crash Penalty:** $-10.0$ upon colliding with an asteroid or sheath boundary (triggers instant termination).
- **Survival Incentive:** $+0.01$ per active timestep alive.
- **Cooperative Team Averaging:** When `team_reward=True`, each probe receives the swarm-averaged reward $\bar{r} = \frac{1}{N} \sum_{i=1}^N r_i^{\text{indiv}}$, maintaining invariant reward scale across varying swarm sizes $N$.

---

## Graph Communication & Topologies

ScrollSwarm provides an integrated graph communication layer (`scrollswarm/comm.py`):

- **Supported Topologies:**
  - `chain` (Default): Path graph $P_N$ along probe columns ($A_{ij} = 1 \iff |i - j| = 1$). Ideal for studying predictive advance signaling.
  - `ring`: Cycle graph $C_N$ with wrap-around link $(0 \leftrightarrow N-1)$.
  - `full`: Complete graph $K_N$ (all-to-all broadcast upper bound).
  - `radius`: Dynamic spatial $\epsilon$-disk graph based on Euclidean distance ($\|\mathbf{p}_i - \mathbf{p}_j\| \le R_{\text{comm}}$).
  - `capped`: Degree-bounded $k$-nearest spatial neighbors.
- **Dynamic Edge Pruning:** When a probe terminates due to collision, all incident incoming and outgoing edges are pruned immediately, preventing ghost messages.
- **PyTorch Geometric (PyG) Compatibility:** Directly extracts standard `(2, E)` `edge_index` and `(E, 3)` relative coordinate `edge_attr` tensors.
- **Message Router:** Routes message vectors across active edges with `mean`, `max`, `sum`, or `concat` aggregation.

---

## Minimal Stepping Example

```python
import numpy as np
from scrollswarm import ScrollSwarmEnv

# 1. Initialize environment with 5 rocket probes
env = ScrollSwarmEnv(
    num_agents=5,
    max_cycles=1000,
    target_reward=1.0,
    crash_penalty=10.0,
    step_reward=0.01,
    team_reward=False,
    render_mode=None,  # Headless mode for high-throughput training
)

# 2. Reset the environment (seed for reproducible asteroid belt generation)
observations, infos = env.reset(seed=42)

# 3. Step through episode rollout
total_reward = 0.0
for step in range(200):
    if not env.agents:
        print(f"Swarm eliminated at step {step}!")
        break

    # Sample random actions: 0 = UP, 1 = STAY, 2 = DOWN
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    # Advance simulation by 1 timestep
    observations, rewards, terminations, truncations, infos = env.step(actions)
    total_reward += sum(rewards.values())

    # Extract PyG communication graph tensors
    edge_index, edge_attr = env.get_edge_index_and_attributes()

print(f"Rollout complete. Swarm Total Reward: {total_reward:.2f}")
env.close()
```

---

## Project Structure

```text
scrollswarm/
├── pyproject.toml         # Dependency & packaging configuration (uv/hatchling)
├── uv.lock                # Deterministic lockfile
├── README.md              # Project documentation and specifications
├── main.py                # Interactive 2D visualizer & flight demo runner
├── assets/
│   ├── rocket.svg         # High-resolution vector rocket probe model
│   └── screenshot.png     # Visual simulation preview
├── scrollswarm/
│   ├── assets/
│   │   └── rocket.svg     # Packaged vector rocket sprite (package data)
│   ├── __init__.py        # Public package exports
│   ├── env.py             # PettingZoo ParallelEnv standard environment wrapper
│   ├── map.py             # Ring corridor engine, asteroid lifetimes, and clearance invariant
│   ├── agent.py           # Probe entity, collision models, and vertical actuation (k = 5px)
│   ├── sprites.py         # Vector rocket sprite loading, dynamic SVG recoloring, and caching
│   ├── sensors.py         # 7-ray forward LiDAR rangefinder & observation vector generator
│   ├── comm.py            # Graph communication topologies & PyG tensor extraction
│   └── items.py           # Asteroid polygon and telemetry target beacon dataclasses
└── tests/
    ├── test_map.py        # Sheath limits, scrolling, culling, and clearance invariant tests
    ├── test_agent.py      # Movement (k=5px), collisions, clamping, and team creation tests
    ├── test_sprites.py    # Rocket sprite loading, recoloring, and cache verification tests
    ├── test_sensors.py    # Raycasting math, boundary hits, asteroid edge intersections
    ├── test_env.py        # PettingZoo ParallelEnv compliance & truncation tests
    └── test_comm.py       # Graph topology, dead-agent pruning, and message routing tests
```

---

## Academic References & Theoretical Foundations

1. **Hill, G. W. (1878).** *Researches in the Lunar Theory*. American Journal of Mathematics, 1(1), 5–26.
2. **Murray, C. D., & Dermott, S. F. (1999).** *Solar System Dynamics*. Cambridge University Press (Chapters 3 & 10: "The Three-Body Problem" and "Planetary Ring Dynamics").
3. **Goldreich, P., & Tremaine, S. (1982).** *The dynamics of planetary rings*. Annual Review of Astronomy and Astrophysics, 20(1), 249–283.
4. **Stewart, G. R., Lin, D. N., & Bodenheimer, P. (1984).** *Collision-induced transport processes in planetary rings*. In R. Greenberg & A. Brahic (Eds.), *Planetary Rings* (pp. 447–497). University of Arizona Press.
5. **Salo, H. (1992).** *Numerical simulations of dense collisional systems: An application to Saturn's rings*. The Astronomical Journal, 104, 1964–1972.
6. **Preparata, F. P., & Shamos, M. I. (1985).** *Computational Geometry: An Introduction*. Springer-Verlag.

---

## License

MIT License.
