# Configuration & Telemetry

This page provides a reference for environment initialization parameters, `MapConfig` corridor settings, and runtime telemetry provided in the `infos` dictionary. For theoretical formulations of the reward function and episode truncation, see [Rewards & Episode Horizon](rewards_and_state.md). For physical kinematics and the corridor passage invariant, see [Environment Dynamics](corridor_and_dynamics.md).

---

## 1. Environment Constructor Arguments (`ScrollSwarmEnv`)

The primary simulation environment is initialized via `ScrollSwarmEnv`:

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

The physical airspace corridor, obstacle generation rates, and passage invariants are managed via `MapConfig`:

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

## 3. Telemetry Info Dictionary

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
