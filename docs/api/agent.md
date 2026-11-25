# Agent & Dynamics API

The `agent` module defines the physical autonomous quadcopter UAV entity, movement kinematics, discrete actions, and agent group generation.

---

## Overview

- **Entity Model:** 2D quadcopter with fixed longitudinal station-keeping column, dynamic vertical position $y(t)$, and bounding hitbox ($20 \times 16\text{ px}$).
- **Kinematic Update:** $\Delta y = -k$ (UP), $\Delta y = 0$ (STAY), $\Delta y = +k$ (DOWN), clamped to corridor boundary margins.
- **Collision & Lifecycle:** Immediate crash (`alive = False`) upon boundary or obstacle contact.

---

## Quick Usage

```python
from scrollswarm.agent import Agent, Action, create_agents

# Create a coordinated swarm of 5 quadcopters
agents = create_agents(num_agents=5, map_width=800, map_height=400)

for agent in agents:
    # Execute climb action
    agent.step(Action.UP, map_height=400)
    print(f"UAV {agent.agent_id} at ({agent.x}, {agent.y}), Alive: {agent.alive}")
```

---

## Class & Function Reference

::: scrollswarm.agent.Action

::: scrollswarm.agent.Agent

::: scrollswarm.agent.create_agents
