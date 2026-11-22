# Quickstart Guide

This guide walks through creating an environment instance, stepping random actions, and retrieving rewards and telemetry.

---

## Minimal Stepping Example

```python
import numpy as np
from scrollswarm import ScrollSwarmEnv

# 1. Initialize environment with 5 quadcopter agents
env = ScrollSwarmEnv(
    num_agents=5,
    max_cycles=1000,
    target_reward=1.0,
    crash_penalty=10.0,
    step_reward=0.01,
    team_reward=False,
    render_mode=None,  # Headless mode for maximum speed
)

# 2. Reset the environment (seed for reproducible terrain)
observations, infos = env.reset(seed=42)

print(f"Possible Agents: {env.possible_agents}")
print(f"Active Agents:   {env.agents}")
print(f"Single Observation shape: {observations['agent_0'].shape}")

# 3. Step through episode rollout
total_reward = 0.0
for step in range(200):
    if not env.agents:
        print(f"Swarm eliminated at step {step}!")
        break

    # Sample random actions: 0 = UP, 1 = STAY, 2 = DOWN
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    # Step the simulation
    observations, rewards, terminations, truncations, infos = env.step(actions)
    total_reward += sum(rewards.values())

    # Check CTDE centralized global state
    global_state = env.state()

print(f"Completed rollouts. Swarm Total Reward: {total_reward:.2f}")
env.close()
```

---

## Interactive Visualization

To launch the real-time Pygame desktop viewer with manual keyboard flight controls:

```bash
uv run python main.py -n 5 -c chain
```

| Key | Action |
| :--- | :--- |
| **`W` / `UP`** | Climb focused agent up ($k = 5\text{ px}$) |
| **`S` / `DOWN`** | Dive focused agent down ($k = 5\text{ px}$) |
| **`TAB`** / **`1`–`9`** | Switch active focused quadcopter |
| **`A`** | Move all alive agents synchronously |
| **`V`** | Toggle 7-ray LiDAR beam visualization |
| **`C`** | Toggle graph communication data links |
| **`R`** | Restart simulation & reseed cavern |
| **`SPACE`** | Pause / Resume |
