# `ScrollSwarmEnv` API

The `ScrollSwarmEnv` class is the central simulation harness conforming to Farama Foundation's **PettingZoo `ParallelEnv`** standard for Multi-Agent Reinforcement Learning (MARL).

---

## Overview

- **Interface:** PettingZoo `ParallelEnv` (`env.reset()`, `env.step()`, `env.render()`, `env.close()`).
- **Agents:** String identifiers `"drone_0"`, `"drone_1"`, ..., `"drone_{N-1}"`.
- **Observation Space:** `spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)`.
- **Action Space:** `spaces.Discrete(3)` $\to$ `{0: Action.UP, 1: Action.STAY, 2: Action.DOWN}`.
- **CTDE Centralized State:** Available via `env.state()` returning `np.ndarray` of shape `(16 * num_agents,)`.
- **Relational GNN Tensors:** Available via `env.get_edge_index_and_attributes()`.

---

## Quick Usage

```python
from scrollswarm import ScrollSwarmEnv

env = ScrollSwarmEnv(num_agents=5, max_cycles=1000, render_mode="human")
observations, infos = env.reset(seed=42)

while env.agents:
    # Sample random actions or query policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

---

## Class Reference

::: scrollswarm.env.ScrollSwarmEnv
