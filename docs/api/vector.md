# Vectorized Environments API

The `scrollswarm.vector` module provides synchronous (`SyncVectorScrollSwarm`) and multiprocess parallel (`AsyncVectorScrollSwarm`) runners designed explicitly for **CTDE** and **Graph-Based MARL** training.

Unlike generic PettingZoo-to-Gymnasium vectorizers that flatten `(E, N)` into a single batch axis and discard `.state()` and graph topologies, `scrollswarm.vector` preserves swarm structure across all $\mathcal{E}$ corridors.

---

## Overview

- **Preserved Swarm Axes:** Returns rectangular NumPy batches with explicit corridor ($\mathcal{E}$) and agent ($N$) dimensions (`obs: (E, N, 16)`, `rewards: (E, N)`, `alive_mask: (E, N)`).
- **CTDE Global State:** Exposes centralized critic inputs (`infos["state"]` of shape `(E, 16 * N)`).
- **Relational GNN Support:** Exports dense per-corridor adjacency (`(E, N, N)`) and spatial edge features (`(E, N, N, 3)`), plus a zero-dependency `batch_graphs()` converter to block-diagonal PyG COO tensors (`edge_index`, `edge_attr`, `batch`).
- **Auto-Reset Semantics:** Finished corridors immediately reset inside the same `step()` call while preserving `final_observation`, `final_state`, and `final_info` for exact GAE bootstrapping.
- **Cross-Platform Multiprocessing:** `AsyncVectorScrollSwarm` uses the `spawn` context and shards corridors across `num_workers` CPU processes, pairing seamlessly with `backend="cpp"` (zero JIT warmup).

---

## Quick Usage

```python
import numpy as np
from scrollswarm import AsyncVectorScrollSwarm, BaseVectorScrollSwarm

# Launch 8 corridors distributed across 4 CPU workers using the C++20 kernel
with AsyncVectorScrollSwarm(
    num_envs=8,
    num_workers=4,
    num_agents=4,
    max_cycles=500,
    comm_topology="radius",
    backend="auto",
) as vec_env:
    obs, infos = vec_env.reset(seed=42)
    # obs: (8, 4, 16), infos["state"]: (8, 64), infos["alive_mask"]: (8, 4)

    for _ in range(200):
        actions = vec_env.action_space.sample()  # shape (8, 4)
        obs, rewards, terminations, truncations, infos = vec_env.step(actions)

        # Convert dense graphs to PyG-compatible block-diagonal COO batch
        edge_index, edge_attr, batch = BaseVectorScrollSwarm.batch_graphs(
            infos["adj"], infos["edge_attr"]
        )
```

---

## Class Reference

::: scrollswarm.vector.BaseVectorScrollSwarm

::: scrollswarm.vector.SyncVectorScrollSwarm

::: scrollswarm.vector.AsyncVectorScrollSwarm
