# Rewards & Centralized Critic State

ScrollSwarm provides flexible reward formulations and full support for Centralized Training with Decentralized Execution (CTDE) algorithms (e.g., MAPPO, QMIX, MADDPG).

---

## Reward Formulation

Rewards are parameterized in `ScrollSwarmEnv`:

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `target_reward` | `float` | `1.0` | Positive reward upon collecting a floating target |
| `crash_penalty` | `float` | `10.0` | Negative penalty on collision with mountain boundary or tree |
| `step_reward` | `float` | `0.01` | Survival incentive granted for every active step alive |
| `team_reward` | `bool` | `False` | When `True`, agents receive the swarm-averaged cooperative reward |

### Individual Agent Reward ($r_i^{\text{indiv}}$)

At step $t$, the individual reward for agent $i$ is:

$$r_i^{\text{indiv}} = \begin{cases}
-R_{\text{crash}}, & \text{if agent collided this step} \\
R_{\text{step}} + R_{\text{target}} \cdot C_i, & \text{if agent survived}
\end{cases}$$

where $C_i \in \{0, 1\}$ indicates target collection.

### Cooperative Team Reward ($\bar{r}$)

When `team_reward=True`, all active agents receive the swarm-averaged reward scaled by total swarm size $N = \text{num\_agents}$:

$$\bar{r} = \frac{1}{N} \sum_{j=0}^{N-1} r_j^{\text{indiv}}$$

Averaging over total swarm size $N$ keeps the return scale invariant to team size expansions.

---

## Centralized Critic State (`env.state()`)

In CTDE architectures, the actor operates solely on the local observation $\mathbf{o}_i \in [0, 1]^{15}$, while the centralized critic evaluates the global state $\mathbf{s} \in [0, 1]^{15 \cdot N}$:

$$\mathbf{s} = \big[ \mathbf{o}_0^\top, \mathbf{o}_1^\top, \dots, \mathbf{o}_{N-1}^\top \big]^\top$$

### Dead Agent Invariant

When an agent $j$ is eliminated, its 15-dimensional slice in $\mathbf{s}$ is filled with zeros:

$$\mathbf{o}_j = \mathbf{0}_{15}, \quad \text{if } \text{alive}_j = \text{False}$$

This preserves invariant tensor dimensions throughout the episode, eliminating dynamic padding issues in centralized value networks.
