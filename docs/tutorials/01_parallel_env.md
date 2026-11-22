# Tutorial 01: The PettingZoo ParallelEnv Loop

This tutorial walks through initializing `ScrollSwarmEnv`, interacting with the standard PettingZoo `ParallelEnv` API, handling agent terminations, and extracting telemetry.

---

## Complete Executable Script

```python
"""
Tutorial 01: PettingZoo ParallelEnv Loop.
Demonstrates environment reset, random action stepping, and rollout termination.
"""

from scrollswarm import ScrollSwarmEnv

# 1. Instantiate the environment
env = ScrollSwarmEnv(
    num_agents=5,
    max_cycles=500,
    target_reward=1.0,
    crash_penalty=10.0,
    step_reward=0.01,
    team_reward=False,
    render_mode=None,  # Set to "human" for desktop GUI window
)

# 2. Reset the environment
observations, infos = env.reset(seed=42)

print(f"Total possible agents: {env.possible_agents}")
print(f"Active agents at step 0: {env.agents}")

episode_reward = 0.0
step_count = 0

# 3. Main rollout loop
while env.agents:
    step_count += 1

    # In PettingZoo, actions are dictionaries mapped by active agent IDs
    actions = {
        agent_id: env.action_space(agent_id).sample()
        for agent_id in env.agents
    }

    # Step transition
    observations, rewards, terminations, truncations, infos = env.step(actions)

    step_team_reward = sum(rewards.values())
    episode_reward += step_team_reward

    # Log any agent collisions this step
    for agent_id, info in infos.items():
        if info["collision"] is not None:
            print(f"Step {step_count}: {agent_id} crashed into {info['collision']}! Penalty applied.")

print(f"Episode finished at step {step_count}. Cumulative Swarm Return: {episode_reward:.2f}")
env.close()
```

---

## Explanation of Key Concepts

### Dynamic Agent List (`env.agents`)
In PettingZoo, `env.agents` contains only the agents that are currently active. When an agent crashes into a ceiling, floor, or tree, `terminations[agent] = True`. At the end of the `step()` call, terminated agents are pruned from `env.agents`. The episode ends when `env.agents` is empty or when `truncations` triggers at `max_cycles`.

### Space Dictionaries
- `env.observation_space(agent_id)` returns a `gym.spaces.Box(0.0, 1.0, shape=(15,))`.
- `env.action_space(agent_id)` returns a `gym.spaces.Discrete(3)` with actions `{0: UP, 1: STAY, 2: DOWN}`.
