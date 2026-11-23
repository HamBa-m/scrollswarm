# Tutorial 01: PettingZoo ParallelEnv Multi-Agent Loop

This tutorial walks through building a complete, production-grade Multi-Agent Reinforcement Learning (MARL) control loop in ScrollSwarm using Farama Foundation's **PettingZoo `ParallelEnv`** standard.

You will learn how to:

1. Understand the synchronous Dec-POMDP multi-agent execution model.
2. Configure environment parameters, corridor geometries, and communication networks.
3. Manage the dynamic agent lifecycle: active vs. terminated vs. truncated agents.
4. Extract centralized state tensors for Centralized Training with Decentralized Execution (CTDE).
5. Capture rendered RGB frames for headless logging and video generation.

---

## 1. Multi-Agent Execution Model: The ParallelEnv Paradigm

In single-agent reinforcement learning (Gymnasium), the interaction follows a sequential loop: the agent receives observation $s_t$, takes action $a_t$, and receives $s_{t+1}, r_t, d_t$.

In cooperative Multi-Agent Reinforcement Learning modeled as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**, all living agents observe their individual perspectives and submit actions **simultaneously** at each time step $t$.

PettingZoo provides two distinct paradigms for multi-agent interaction:

| Paradigm | Interface | Execution Order | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **Agent-Environment Cycle (AEC)** | `env.observe()`, `env.step()` | Turn-based, sequential | Turn-based board games (e.g., Chess, Go) |
| **Parallel Environment (`ParallelEnv`)** | `env.step(actions_dict)` | Synchronous, parallel | Real-time swarms, robotics, simultaneous games |

ScrollSwarm inherits from `pettingzoo.ParallelEnv`. At every discrete time step $t$:

$$\mathbf{a}_t = \big( a_t^1, a_t^2, \dots, a_t^N \big) \in \pmb{\mathcal{A}} = \prod_{i=1}^N \mathcal{A}_i$$

All actions are dispatched in parallel via Python dictionaries keyed by agent identifier strings (`"agent_0"`, `"agent_1"`, etc.).

### Agent Lifecycle: `possible_agents` vs. `agents`

ScrollSwarm maintains two distinct agent lists conforming to PettingZoo specifications:

- **`env.possible_agents`**: The immutable, static list of all potential agents that could exist in the environment (e.g., `["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"]`). This set never shrinks or changes during an episode.
- **`env.agents`**: The dynamic list of agents currently active (alive and un-truncated). When a drone collides with an obstacle or boundary, it suffers structural termination (`alive = False`), is penalized, and is removed from `env.agents` on the subsequent step.

!!! warning "Important: Active Agent Set"
    The action dictionary passed to `env.step(actions)` must contain keys for **all** currently active agents in `env.agents`. Attempting to submit actions for dead agents is ignored, while omitting active agents will skip their physical updates.

---

## 2. Environment Initialization & Configuration

Instantiating `ScrollSwarmEnv` allows fine-grained control over swarm cardinality, reward balance, communication topologies, and rendering:

```python
from scrollswarm import ScrollSwarmEnv
from scrollswarm.map import MapConfig
from scrollswarm.comm import TopologyType

# Optional: Customize corridor dimensions and safety passage
custom_map = MapConfig(
    width=800,
    height=400,
    scroll_speed=1,
    min_corridor_gap=110,   # Guaranteed clearance passage (pixels)
    ceiling_y=15,           # Upper boundary buffer
    floor_margin=15,        # Lower boundary buffer
    seed=42,
)

# Initialize the multi-agent environment
env = ScrollSwarmEnv(
    num_agents=5,                           # Swarm cardinality N (1 to 10)
    max_cycles=1000,                        # Maximum episode horizon (T_max)
    map_config=custom_map,                  # Corridor geometry
    comm_topology=TopologyType.CHAIN,       # Communication graph network
    target_reward=1.0,                      # Photo acquisition reward
    crash_penalty=10.0,                     # Structural destruction penalty
    step_reward=0.01,                       # Per-step survival incentive
    team_reward=False,                      # Individual vs. team-shared rewards
    render_mode="human",                    # "human", "rgb_array", or None
)
```

### Action and Observation Spaces

For any agent `agent_id` in `env.possible_agents`:

```python
obs_space = env.observation_space("agent_0")
act_space = env.action_space("agent_0")

print(f"Observation space: {obs_space}")  # Box(-1.0, 1.0, (16,), float32)
print(f"Action space:      {act_space}")  # Discrete(3)
```

The discrete action indices map directly to vertical thrust kinematics:

- **`0` (`Action.UP`)**: Ascends (climbs) $\Delta y = -k$ pixels ($k = 5\text{ px}$).
- **`1` (`Action.STAY`)**: Maintains current flight altitude $\Delta y = 0\text{ px}$.
- **`2` (`Action.DOWN`)**: Descends (dives) $\Delta y = +k$ pixels ($k = 5\text{ px}$).

---

## 3. The Synchronous Stepping Loop

The standard execution loop consists of resetting the environment, querying policies, executing synchronous steps, and handling terminations:

```python
observations, infos = env.reset(seed=42)

while env.agents:
    # 1. Query policies or sample random actions for all living drones
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    # 2. Advance the environment by one synchronized step
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

### Unpacking the Step Return

`env.step()` returns five synchronized dictionaries, keyed by active agent identifiers:

1. **`observations`** (`Dict[str, np.ndarray]`):
   Each value is a normalized 16-dimensional float vector:
   $$\mathbf{o}_i = [y_{\text{norm}}, \; d_{\text{LiDAR}, 0\dots 6}, \; \mathbf{v}_{\text{tgt}}, \; \mathbf{v}_{\text{nbr}}] \in [-1.0, 1.0]^{16}$$
2. **`rewards`** (`Dict[str, float]`):
   Scalar feedback received by each drone during this transition:
   $$r_i(t) = r_{\text{step}} + r_{\text{target}} \cdot \mathbf{1}_{\text{photo}} - r_{\text{crash}} \cdot \mathbf{1}_{\text{collision}}$$
3. **`terminations`** (`Dict[str, bool]`):
   `True` if the agent suffered a fatal collision (boundary or obstacle contact) at this step.
4. **`truncations`** (`Dict[str, bool]`):
   `True` for all agents when `step_count >= max_cycles`.
5. **`infos`** (`Dict[str, Dict[str, Any]]`):
   Diagnostic telemetry containing:
   - `"targets_photographed"`: Cumulative count of targets successfully photographed by this agent.
   - `"collision"`: Contact cause (`"boundary"`, `"obstacle"`, or `None`).
   - `"steps_alive"`: Cumulative time steps this UAV has survived.

!!! danger "Caution: Avoiding KeyError on Agent Death"
    Once an agent terminates (`terminations[a] = True`), it is removed from `env.agents` on the next step. As a result, subsequent `observations`, `rewards`, and `infos` will **no longer contain keys** for that agent. Always check `if agent in observations` or track cumulative metrics in an external dictionary across `env.possible_agents`.

---

## 4. Centralized State for CTDE Algorithms

State-of-the-art Multi-Agent Actor-Critic frameworks (e.g., MAPPO, MADDPG, QMIX) require access to a global centralized state $\mathbf{s} \in \mathcal{S}$ during training, while decentralized actors select actions conditioned only on local observations $\mathbf{o}_i$.

ScrollSwarm exposes this centralized representation through `env.state()` and `env.state_space`:

```python
# Access the centralized global state vector
global_state = env.state()
print(f"Global state shape: {global_state.shape}")  # (16 * N,) -> (80,) for N=5
print(f"State space:        {env.state_space}")      # Box(-1.0, 1.0, (80,), float32)
```

### Static Dimension & Dead-Agent Zero-Masking

A critical requirement for neural networks is fixed input dimensionality. In ScrollSwarm:

$$\mathbf{s}(t) = \big[ \mathbf{o}_1(t)^\top, \; \mathbf{o}_2(t)^\top, \; \dots, \; \mathbf{o}_N(t)^\top \big]^\top \in \mathbb{R}^{16N}$$

When drone $j$ crashes and terminates:

$$\mathbf{o}_j(t) = \mathbf{0}_{16}$$

The environment automatically zero-masks the feature slices of dead agents, preserving the static shape $(16N,)$ throughout the entire episode.

---

## 5. Headless Rendering & Image Array Export

When training on headless servers or computing clusters without an X11/Wayland display server, set `render_mode="rgb_array"`:

```python
env = ScrollSwarmEnv(num_agents=5, max_cycles=200, render_mode="rgb_array")
env.reset(seed=42)

# Advance simulation
actions = {agent: 1 for agent in env.agents}
env.step(actions)

# Capture frame as uint8 NumPy image array
frame = env.render()
print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
# Output: Frame shape: (400, 800, 3), dtype: uint8
```

You can feed these frame arrays directly into `imageio` or `matplotlib` to render episode replays or visual telemetry logs.

---

## 6. Complete End-to-End Example

The script below demonstrates a full, robust rollout loop: initializing 5 UAVs, managing the agent lifecycle, recording episodic telemetry without `KeyError`, and safely terminating resources.

```python
"""
Tutorial 01: Complete PettingZoo ParallelEnv Rollout Script.
"""

from typing import Dict, Any
import numpy as np
from scrollswarm import ScrollSwarmEnv
from scrollswarm.comm import TopologyType


def run_episode(seed: int = 42) -> Dict[str, Any]:
    # 1. Initialize environment
    env = ScrollSwarmEnv(
        num_agents=5,
        max_cycles=300,
        comm_topology=TopologyType.CHAIN,
        target_reward=1.0,
        crash_penalty=10.0,
        step_reward=0.01,
        render_mode=None,  # Set to "human" for live Pygame window
    )

    observations, infos = env.reset(seed=seed)

    # 2. Initialize persistent metric tracker for all possible agents
    telemetry = {
        agent_id: {
            "cumulative_reward": 0.0,
            "steps_survived": 0,
            "targets_photographed": 0,
            "collision_cause": None,
        }
        for agent_id in env.possible_agents
    }

    step_idx = 0
    while env.agents:
        step_idx += 1

        # 3. Formulate joint action dictionary
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        # 4. Synchronous environment transition
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # 5. Safely record telemetry for living and newly-terminated agents
        for agent_id, reward in rewards.items():
            telemetry[agent_id]["cumulative_reward"] += reward

        for agent_id, info in infos.items():
            telemetry[agent_id]["steps_survived"] = info["steps_alive"]
            telemetry[agent_id]["targets_photographed"] = info["targets_photographed"]
            if info["collision"] is not None:
                telemetry[agent_id]["collision_cause"] = info["collision"]

        # Optional: Inspect centralized CTDE state
        if step_idx % 50 == 0:
            ctde_state = env.state()
            active_count = len(env.agents)
            print(f"Step {step_idx:03d} | Active UAVs: {active_count}/5 | CTDE State Norm: {np.linalg.norm(ctde_state):.2f}")

    env.close()

    # 6. Aggregate episode metrics
    total_team_reward = sum(d["cumulative_reward"] for d in telemetry.values())
    total_targets = sum(d["targets_photographed"] for d in telemetry.values())
    mean_survival = np.mean([d["steps_survived"] for d in telemetry.values()])

    return {
        "total_team_reward": total_team_reward,
        "total_targets": total_targets,
        "mean_survival_steps": mean_survival,
        "details": telemetry,
    }


if __name__ == "__main__":
    print("Executing ScrollSwarm PettingZoo ParallelEnv Rollout...")
    results = run_episode(seed=123)

    print("\n--- Episode Summary ---")
    print(f"Total Team Return:       {results['total_team_reward']:.2f}")
    print(f"Total Targets Surveyed:  {results['total_targets']}")
    print(f"Mean Survival Steps:     {results['mean_survival_steps']:.1f}")

    print("\n--- Per-Agent Diagnostics ---")
    for agent_id, stats in results["details"].items():
        status = "SURVIVED" if stats["collision_cause"] is None else f"CRASHED ({stats['collision_cause']})"
        print(f"  {agent_id:8s} | Return: {stats['cumulative_reward']:6.2f} | Steps: {stats['steps_survived']:3d} | Status: {status}")
```

---

## Next Steps

Now that you have mastered the basic multi-agent simulation loop:

- Proceed to [Tutorial 02: Heuristic Reactive Policy](02_heuristic_baseline.md) to build an explainable, rule-based controller using the 7-ray LiDAR and target tracker.
- Explore [Tutorial 03: PyG & GNN Integration](03_gnn_pyg.md) to leverage relational graph neural networks for inter-drone message passing.
