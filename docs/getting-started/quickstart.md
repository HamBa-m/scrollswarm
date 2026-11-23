# Quickstart & Visualizer

Get up and running with ScrollSwarm in under two minutes: fly drones interactively using the Pygame visualizer, or run a minimal multi-agent stepping script using Farama's PettingZoo `ParallelEnv` interface.

---

## 1. Interactive Flight Simulator

ScrollSwarm comes with a rich 2D Pygame flight visualizer. It lets you pilot individual drones, observe the 7-ray forward LiDAR rangefinder in real time, and inspect inter-drone communication graphs.

### Launching the Visualizer

```bash
# Default: 5 UAVs with chain communication topology
uv run python main.py

# Custom swarm size (1 to 10 UAVs) and communication network
uv run python main.py --num-agents 6 --comm-topology ring
```

### Interactive Controls

| Key | Action |
| :--- | :--- |
| **`W` / `UP`** | Climb focused UAV ($k = 5\text{ px}$) |
| **`S` / `DOWN`** | Dive focused UAV ($k = 5\text{ px}$) |
| **`TAB`** | Cycle to next active UAV |
| **`1`–`9`** | Directly select UAV #1 through #9 |
| **`A`** | Move all alive UAVs synchronously with current thrust |
| **`V`** | Toggle 7-ray forward LiDAR beam overlay |
| **`C`** | Toggle active inter-drone communication data links |
| **`R`** | Reset simulation and generate fresh corridor |
| **`SPACE`** | Pause / Resume flight |
| **`ESC`** | Quit visualizer |

### On-Screen HUD Telemetry
The top-left HUD displays:

- **UAVs Active:** Number of living drones vs. initial swarm size.
- **Targets Photographed:** Total survey targets successfully captured by the team.
- **Focused UAV:** ID, flight status (`AIRBORNE` vs. `CRASHED`), and personal photo score.

---

## 2. Minimal Multi-Agent Stepping Script

The following self-contained script shows how to initialize the environment, execute discrete actions in parallel, and retrieve observations, rewards, and PyTorch Geometric communication graph tensors:

```python
import numpy as np
from scrollswarm import ScrollSwarmEnv

# 1. Initialize environment with 5 drones
env = ScrollSwarmEnv(num_agents=5, max_cycles=500, render_mode=None)
observations, infos = env.reset(seed=42)

print(f"Active agents: {env.agents}")
print(f"Observation shape: {observations['agent_0'].shape}")  # (16,)

for step in range(100):
    if not env.agents:
        print("All agents have crashed. Episode ended early.")
        break

    # 2. Sample discrete actions for all active drones
    # Actions: 0 = UP (climb), 1 = STAY (level), 2 = DOWN (dive)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    # 3. Step environment synchronously (Dec-POMDP)
    observations, rewards, terminations, truncations, infos = env.step(actions)

    # 4. Extract PyG-compatible communication graph tensors
    # edge_index: (2, E) directed connectivity between living drones
    # edge_attr:  (E, 3) relative [dx, dy, distance] features
    edge_index, edge_attr = env.get_edge_index_and_attributes()

    if step % 20 == 0:
        team_score = sum(infos[a]["targets_photographed"] for a in env.agents)
        print(f"Step {step:03d} | Active: {len(env.agents)}/5 | Targets Photographed: {team_score}")

env.close()
```

---

## 3. Rendering to Image Arrays (Headless Training)

When training RL policies on headless compute clusters, you can render frames as NumPy uint8 image arrays without opening a display window:

```python
env = ScrollSwarmEnv(num_agents=4, render_mode="rgb_array")
env.reset(seed=42)

# Step environment
actions = {a: 1 for a in env.agents}
env.step(actions)

# Capture frame as RGB array of shape (400, 800, 3)
frame = env.render()
print(f"Captured frame shape: {frame.shape}, dtype: {frame.dtype}")

env.close()
```

---

## Next Steps

- Review the **[Configuration & Telemetry Reference](config.md)** for environment constructor arguments and runtime telemetry dictionaries.
- Explore the **[Formal Environment Model](../environment/overview.md)** for mathematical Dec-POMDP definitions.
- Inspect the **[Observation & LiDAR Specification](../environment/observation.md)** to understand the 16-dimensional observation vector.
- Check out **[Tutorial 01](../tutorials/01_parallel_env.md)** for training baseline policies.
