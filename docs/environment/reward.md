# Rewards & Episode Horizon

This section formalizes the multi-agent reward structure, optimization objectives, discount factor planning horizon, and finite horizon truncation vs. termination dynamics in ScrollSwarm.

---

## 1. Reward Function Formulation

At each simulation step $t$, every living UAV $i \in \mathcal{I}$ receives a scalar feedback reward $R_i(t) \in \mathbb{R}$. The reward signal balances survival, collision avoidance, and cooperative target reconnaissance:

$$R_i(t) = R_{\text{step}} + R_{\text{target}, i}(t) - R_{\text{crash}, i}(t)$$

### 1.1 Component Breakdown

1. **Step Survival Reward ($R_{\text{step}} = +0.01$):**
   Awarded at every time step that the UAV remains airborne and operational. Provides a dense baseline incentive preventing passive suicide policies.

2. **Target Photography Reward ($R_{\text{target}} = +1.0$):**
   Triggered when an uncollected target is successfully photographed within inspection radius $D_{\text{photo}} = 45\text{ px}$ under stochastic camera uncertainty ($X \sim \text{Bernoulli}(P_{\text{team}})$):
       - **Individual Reward Mode (`team_reward = False`, Default):**
         $$R_{\text{target}, i}(t) = \begin{cases} +1.0 & \text{if UAV } i \text{ was within } D_{\text{photo}} \text{ when photo succeeded} \\ 0.0 & \text{otherwise} \end{cases}$$
       - **Shared Team Reward Mode (`team_reward = True`):**
         The reward is split equally among all currently living swarm members:

         $$R_{\text{target}, i}(t) = \frac{+1.0}{|\mathcal{V}_{\text{alive}}(t)|}$$

4. **Crash Collision Penalty ($R_{\text{crash}} = -10.0$):**
   Assessed instantaneously when any vertex of UAV $i$ penetrates an obstacle or boundary plane:
   $$R_{\text{crash}, i}(t) = \begin{cases} -10.0 & \text{if UAV } i \text{ collides with boundary or obstacle} \\ 0.0 & \text{otherwise} \end{cases}$$


   Collision causes immediate structural termination ($\text{alive}_i \leftarrow \text{False}$).

---

## 2. Cumulative Return & Discount Factor

The reinforcement learning objective for each agent is to maximize its expected discounted return:

$$J_i(\pi) = \mathbb{E}\left[ \sum_{t=0}^T \gamma^t R_i(t) \right]$$

### 2.1 Discount Factor Selection
- **Recommended Range:** $\gamma \in [0.95, 0.99]$ (default baseline: $\gamma = 0.99$).
- **Effective Planning Horizon:** The effective horizon is given by $H_{\text{eff}} = \frac{1}{1 - \gamma}$:
      - At $\gamma = 0.95 \implies H_{\text{eff}} \approx 20\text{ steps}$ ($\approx 0.67\text{ seconds}$ at $30\text{ FPS}$). Focuses narrowly on imminent obstacle avoidance.
      - At $\gamma = 0.99 \implies H_{\text{eff}} \approx 100\text{ steps}$ ($\approx 3.3\text{ seconds}$). Enables drones to proactively position themselves vertically for upcoming targets while anticipating obstacles.

---

## 3. Episode Horizon: Termination vs. Truncation

Following the standard Gymnasium / PettingZoo MDP specification, ScrollSwarm strictly distinguishes between **terminations** (natural environmental ends) and **truncations** (artificial time-limit bounds):

### 3.1 Terminations 
- Evaluated on a **per-agent basis**.
- An agent terminates (`True`) if and only if it collides with a boundary or an obstacle.
- Once terminated, an agent is removed from `env.agents` for the remainder of the episode.
- If all agents terminate, the multi-agent episode ends immediately.

### 3.2 Truncations 
- Occurs when the global simulation step reaches the predefined horizon limit:
  $$\text{step_count} \ge T_{\max} = 1000\text{ steps}$$
- When truncated, all remaining active agents receive `truncations[agent_id] = True` simultaneously.
- Standard RL algorithms (e.g., PPO) use truncation to bootstrap the value function estimate $V(s_{T_{\max}})$ without misinterpreting the cutoff as a failure.

---

## 4. References

1. **Puterman, M. L. (1994).** *Markov Decision Processes: Discrete Stochastic Dynamic Programming.* John Wiley & Sons. [DOI: 10.1002/9780470316887](https://doi.org/10.1002/9780470316887).
2. **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction (2nd Edition).* MIT Press. [Book Link](http://incompleteideas.net/book/the-book-2nd.html).
3. **Oliehoek, F. A., & Amato, C. (2016).** *A Concise Introduction to Decentralized POMDPs.* SpringerBriefs in Intelligent Systems, Springer. [DOI: 10.1007/978-3-319-28929-8](https://doi.org/10.1007/978-3-319-28929-8).
