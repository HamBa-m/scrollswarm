# Tutorial 02: Heuristic Reactive Policy

Before training deep neural networks, it is essential to establish a deterministic rule-based benchmark to verify task solvability. This tutorial builds a reactive obstacle-avoidance and target-tracking policy directly using the 15-dimensional LiDAR vector.

---

## The Heuristic Decision Logic

The 15-dimensional vector $\mathbf{o}_i$ provides:
- Normalized vertical coordinate: $y_i / H$ at index 0.
- Obstacle distances along rays: indices 1 to 7 (angles $-60^\circ$ to $+60^\circ$).
- Target distances along rays: indices 8 to 14 (angles $-60^\circ$ to $+60^\circ$).

### Rule Strategy:
1. **Target Seeking (Opportunistic):** If an upper target ray detects a target ($d_{\text{tgt}} < 1.0$), prioritize climbing (`UP`). If a lower target ray detects a target, prioritize diving (`DOWN`).
2. **Obstacle Avoidance (Defensive):** If upper rays ($d_{\text{obs}, 0..2}$) detect an oncoming ceiling or rock crag below a safety threshold ($\tau_{\text{safe}} = 0.45$), force diving (`DOWN`).
3. **Ground Avoidance (Defensive):** If lower rays ($d_{\text{obs}, 4..6}$) detect an oncoming floor or crag below $\tau_{\text{safe}}$, force climbing (`UP`).
4. **Safety Override:** Obstacle avoidance strictly supersedes target tracking.

---

## Implementation

```python
"""
Tutorial 02: Heuristic Reactive Policy.
Navigates the cavern and collects targets using deterministic LiDAR thresholding.
"""

from typing import Dict
import numpy as np
from scrollswarm import ScrollSwarmEnv, Action


def heuristic_policy(obs: np.ndarray, safe_threshold: float = 0.40) -> int:
    """
    Computes a discrete action (0: UP, 1: STAY, 2: DOWN) from a 15-D LiDAR vector.
    """
    # Unpack observation
    y_norm = obs[0]
    obs_rays = obs[1:8]     # Rays 0..6 (-60° to +60°)
    tgt_rays = obs[8:15]    # Rays 0..6 (-60° to +60°)

    # 1. Evaluate obstacle proximity
    upper_min_dist = np.min(obs_rays[0:3])   # Rays -60°, -40°, -20°
    center_dist    = obs_rays[3]             # Ray 0°
    lower_min_dist = np.min(obs_rays[4:7])   # Rays +20°, +40°, +60°

    # Immediate defensive avoidance overrides target tracking
    if upper_min_dist < safe_threshold:
        return Action.DOWN
    if lower_min_dist < safe_threshold:
        return Action.UP
    if center_dist < safe_threshold:
        # Obstacle straight ahead: steer toward the side with greater clearance
        return Action.UP if upper_min_dist > lower_min_dist else Action.DOWN

    # 2. Opportunistic target tracking
    upper_tgt = np.min(tgt_rays[0:3])
    lower_tgt = np.min(tgt_rays[4:7])

    if upper_tgt < 1.0 and upper_tgt <= lower_tgt:
        return Action.UP
    if lower_tgt < 1.0 and lower_tgt < upper_tgt:
        return Action.DOWN

    # Maintain current altitude if clear
    return Action.STAY


# Evaluate the heuristic policy over a full episode
env = ScrollSwarmEnv(num_agents=5, max_cycles=1000, seed=42)
obs, infos = env.reset(seed=42)

step = 0
while env.agents:
    step += 1
    actions = {a: heuristic_policy(obs[a]) for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)

total_score = sum(infos[a]["targets_collected"] for a in infos)
print(f"Heuristic swarm survived {step} steps with total score: {total_score}")
env.close()
```
