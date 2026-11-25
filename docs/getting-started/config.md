# Configuration & Technical Reference

Use this reference to configure ScrollSwarm environments, corridor physics, agent parameters, PyG graph tensors, and step telemetry.

For the underlying mathematical formulations and proofs, see the [Environment Specification](../environment/overview.md).

---

## 1. Environment Constructor Arguments (`ScrollSwarmEnv`)

The primary multi-agent environment is instantiated via `ScrollSwarmEnv`:

```python
from scrollswarm import ScrollSwarmEnv
from scrollswarm.map import MapConfig

env = ScrollSwarmEnv(
    num_agents=5,
    max_cycles=1000,
    target_reward=1.0,
    crash_penalty=10.0,
    step_reward=0.01,
    team_reward=False,
    comm_topology="chain",
    render_mode="human",
)
```

| Parameter | Type | Default | Valid Range | Research Rationale / Impact |
| :--- | :---: | :---: | :---: | :--- |
| `num_agents` | `int` | `5` | $[1, 10]$ | Swarm cardinality $N$. Scaling $N$ increases coordination complexity. |
| `max_cycles` | `int` | `1000` | $[50, 10000]$ | Episode horizon $T_{\max}$ before truncation. |
| `map_config` | `Optional[MapConfig]` | `None` | `MapConfig` instance | Customizes corridor dimensions, scroll speed, and obstacle density. |
| `comm_topology` | `Union[TopologyType, str]` | `"chain"` | `chain`, `ring`, `full`, `radius`, `capped` | Determines the graph communication topology for message routing. |
| `target_reward` | `float` | `1.0` | $[0.1, 100.0]$ | Reward scale for successfully photographing a target. |
| `crash_penalty` | `float` | `10.0` | $[0.0, 100.0]$ | Negative penalty upon obstacle or boundary collision. |
| `step_reward` | `float` | `0.01` | $[0.0, 1.0]$ | Dense survival incentive per time step alive. |
| `team_reward` | `bool` | `False` | `True` / `False` | When `True`, target rewards are shared equally among living drones. |
| `render_mode` | `Optional[str]` | `None` | `"human"`, `"rgb_array"`, `None` | Rendering mode for Pygame display or headless image array export. |

---

## 2. Corridor Configuration (`MapConfig`)

The airspace corridor, obstacle generation rates, and passage invariants are managed via `MapConfig`:

```python
from scrollswarm.map import MapConfig

config = MapConfig(
    width=800,
    height=400,
    fps=30,
    scroll_speed=1,
    ceiling_y=15,
    floor_margin=15,
    min_corridor_gap=110,
    obstacle_spawn_interval=55,
    target_spawn_interval=75,
    seed=42,
)
```

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `width` | `int` | `800` | Airspace corridor width in pixels |
| `height` | `int` | `400` | Airspace corridor height in pixels |
| `fps` | `int` | `30` | Simulation frame rate |
| `scroll_speed` | `int` | `1` | Leftward drift velocity ($v_{\text{scroll}}$ in px/step) |
| `ceiling_y` | `int` | `15` | Upper boundary limit in pixels |
| `floor_margin` | `int` | `15` | Lower boundary margin ($H - 15$ in px) |
| `min_corridor_gap` | `int` | `110` | Guaranteed minimal navigable passage ($G_{\min}$ in px) |
| `obstacle_spawn_interval` | `int` | `55` | Minimum simulation steps between obstacle spawns |
| `target_spawn_interval` | `int` | `75` | Minimum simulation steps between target spawns |
| `preseed` | `bool` | `True` | Pre-populates the corridor with obstacles at step 0 |
| `seed` | `Optional[int]` | `None` | Random seed for deterministic reproducibility |

---

## 3. UAV Physical Dimensions & Agility Tuning (`Agent`)

Quadcopter physical dimensions and vertical movement responsiveness are configured in `scrollswarm/agent.py`:

```python
from scrollswarm.agent import Agent, Action

drone = Agent(
    agent_id=0,
    x=100.0,
    y=200.0,
    step_size=5,     # Vertical step k (px)
    width=20.0,      # Horizontal collision span (px)
    height=16.0,     # Vertical collision span (px)
)

# Step UAV with discrete action (0: UP, 1: STAY, 2: DOWN)
drone.step(Action.UP, map_height=400)
assert drone.y == 195.0
```

| Argument | Location | Default | Range | Impact on Dynamics |
| :--- | :--- | :---: | :---: | :--- |
| `step_size` | `Agent.__init__` | `5` | $[1, 20]$ | Vertical step size $k$ (px); larger values allow faster obstacle evasion but coarser vertical positioning |
| `width` | `Agent.__init__` | `20.0` | $[8.0, 40.0]$ | Physical horizontal collision width of the quadcopter |
| `height` | `Agent.__init__` | `16.0` | $[8.0, 32.0]$ | Physical vertical collision height of the quadcopter |
| `num_agents` | `ScrollSwarmEnv.__init__` | `5` | $[1, 10]$ | Swarm cardinality $N$, defining joint action space dimension $3^N$ |

---

## 4. Observation & State Vector Schemas

### 4.1 Local Observation Vector (`16` floats per UAV)
Each active drone receives a 16-dimensional continuous vector $\mathbf{o}_i \in [-1.0, 1.0]^{16}$ (`spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)`) combining normalized altitude (index `0`), 7 forward LiDAR distances (indices `1–7`), nearest-target features (indices `8–11`), and nearest-neighbor features (indices `12–15`). See the [Observation Space Specification](../environment/observation.md#4-complete-16-dimensional-observation-vector) for the complete index table, mathematical definitions, and fallback values.

### 4.2 Centralized State Vector (`16 * N` floats)
For Centralized Training with Decentralized Execution (CTDE), `env.state()` returns the global concatenated vector $\mathbf{s} \in \mathbb{R}^{16N}$:
- Shape: `(16 * num_agents,)`
- Type: `np.float32`
- Dead Agent Masking: When agent $i$ has terminated (`alive = False`), its corresponding 16-element slice is zero-masked ($\mathbf{0}_{16}$).

---

## 5. Graph Neural Network (PyG) Tensors & Routing Engine

To interface with Graph Neural Network (GNN) architectures, `env.get_edge_index_and_attributes()` extracts network connectivity in canonical PyTorch Geometric format:

```python
edge_index, edge_attr = env.get_edge_index_and_attributes()
```

### 5.1 Tensor Formats
- **`edge_index`:** Shape `(2, E)`, dtype `np.int64` (transfers to `torch.long`).
  - `edge_index[0]`: Source node indices ($j$)
  - `edge_index[1]`: Target node indices ($i$)
- **`edge_attr`:** Shape `(E, 3)`, dtype `np.float32` (transfers to `torch.float32`).
  - Normalized relative spatial edge features for directed link $(j \to i)$:
    $$\mathbf{e}_{ji} = \left[ \frac{x_j - x_i}{W}, \quad \frac{y_j - y_i}{H}, \quad \frac{\|\mathbf{p}_j - \mathbf{p}_i\|}{\sqrt{W^2 + H^2}} \right]^\top \in [-1.0, 1.0]^2 \times [0.0, 1.0]$$

### 5.2 Message Routing Engine & Aggregation
`CommunicationGraph.route_messages()` performs synchronous inter-agent message passing:

$$\tilde{\mathbf{m}}_i = \bigoplus_{j \in \mathcal{N}_{\text{in}}(i)} \mathbf{m}_j$$

| Aggregation Operator | Mathematical Formulation | Use Case |
| :---: | :---: | :--- |
| **`"mean"`** (Default) | $\tilde{\mathbf{m}}_i = \frac{1}{\|\mathcal{N}(i)\|} \sum_{j \in \mathcal{N}(i)} \mathbf{m}_j$ | Normalized continuous latent feature averaging (CommNet) |
| **`"sum"`** | $\tilde{\mathbf{m}}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{m}_j$ | Cumulative counting and target observation accumulation |
| **`"max"`** | $\tilde{\mathbf{m}}_i = \max_{j \in \mathcal{N}(i)} \mathbf{m}_j$ | Max-pooling saliency filtering |
| **`"concat"`** | $\tilde{\mathbf{m}}_i = \big[ \mathbf{m}_{j_1}^\top, \dots, \mathbf{m}_{j_k}^\top \big]^\top$ | Fixed-degree ordered concatenation |

---

## 6. Telemetry Info Dictionary

At each step, `env.step()` returns an `infos` dictionary containing operational telemetry for each active drone:

```python
infos[agent_id] = {
    "targets_photographed": int,   # Total cumulative targets photographed by this UAV
    "targets_collected": int,      # Backward compatibility alias
    "collision": Optional[str],    # "boundary", "obstacle", or None
    "steps_alive": int,            # Cumulative operational lifespan in steps
}
```

### Example Usage: Logging Episode Metrics
```python
observations, rewards, terminations, truncations, infos = env.step(actions)

for agent_id, info in infos.items():
    if info["collision"] is not None:
        print(f"UAV {agent_id} crashed into {info['collision']} after {info['steps_alive']} steps.")

team_photos = sum(info["targets_photographed"] for info in infos.values())
```

---

## 7. Diagnostic System Logging

`scrollswarm` uses Python's standard `logging` module under the `"scrollswarm"` logger hierarchy with a `logging.NullHandler()` attached by default, so it produces no output unless configured and adds no overhead inside `step()`.

To inspect backend resolution (`cpp` vs. `numba` vs. `numpy`), C++ extension load warnings, and environment lifecycle diagnostics in your scripts:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("scrollswarm").setLevel(logging.DEBUG)  # Optional: include reset() & probe details
```
