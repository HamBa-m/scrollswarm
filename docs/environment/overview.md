# Decision-Theoretic Formulations

ScrollSwarm is designed as a foundational benchmark for research at the intersection of Multi-Agent Reinforcement Learning (MARL), networked robotics, and sequential decision-making under uncertainty. Depending on the agent configuration, observability assumptions, and communication constraints, ScrollSwarm can be instantiated across multiple formal mathematical frameworks.

---

## 1. Decision-Theoretic Framework Taxonomy

The following matrix summarizes the fundamental decision-theoretic models formalized in the reinforcement learning literature ([Puterman, 1994](https://doi.org/10.1002/9780470316887); [Kaelbling et al., 1998](https://doi.org/10.1016/S0004-3702(98)00023-X); [Oliehoek & Amato, 2016](https://doi.org/10.1007/978-3-319-28929-8)). All listed models are natively supported by ScrollSwarm:

| Model | Scale | Observability | Communication |
| :--- | :---: | :---: | :---: |
| **MDP** | Single | Full | None |
| **POMDP** | Single | Partial | None |
| **MMDP** | Multi | Full | None |
| **Dec-MDP** | Multi | Jointly Full | None |
| **Dec-POMDP** | Multi | Partial | None |
| **Dec-POMDP-Comm** | Multi | Partial | Integrated (Graph $\mathcal{G}$) |

!!! info "ScrollSwarm Support"
    By configuring swarm size (`num_agents=1` vs. $N \ge 2$), enabling or bypassing graph communication (`comm_topology`), and selecting decentralized observations vs. centralized state vectors (`env.state()`), ScrollSwarm seamlessly maps to any of these six formulations.

---

## 2. Formal Mathematical Definitions

### 2.1 Markov Decision Process (MDP)
When ScrollSwarm is configured with a single UAV ($N=1$) having access to the complete environment state, the problem reduces to a classical MDP ([Bellman, 1957](https://doi.org/10.1512/iumj.1957.6.56038); [Puterman, 1994](https://doi.org/10.1002/9780470316887)):

$$\mathcal{M}_{\text{MDP}} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle$$

- $\mathcal{S}$: The set of all global environmental states, including UAV coordinates $(x, y)$, active obstacle coordinates and geometry, and target states.
- $\mathcal{A}$: The discrete vertical action set $\mathcal{A} = \{0: \text{UP}, 1: \text{STAY}, 2: \text{DOWN}\}$.
- $\mathcal{T}(s' \mid s, a)$: State transition probability function $P(s_{t+1} = s' \mid s_t = s, a_t = a)$, deterministic for kinematics and stochastic for target spawning.
- $\mathcal{R}(s, a)$: Real-valued scalar reward function evaluating step survival, obstacle collision, and target photography.
- $\gamma \in [0, 1)$: The discount factor for future returns.

### 2.2 Partially Observable Markov Decision Process (POMDP)
In realistic single-UAV operations, the agent cannot perceive the global state $\mathcal{S}$ directly; it must act based on its local, bounded sensor readings ([Kaelbling, Littman, & Cassandra, 1998](https://doi.org/10.1016/S0004-3702(98)00023-X); [Kochenderfer et al., 2015](https://mitpress.mit.edu/9780262029254/decision-making-under-uncertainty/)):

$$\mathcal{M}_{\text{POMDP}} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$

- $\Omega$: The observation space, represented by the 16-dimensional continuous feature vector $\mathbf{o} \in [-1.0, 1.0]^{16}$.
- $\mathcal{O}(o \mid s', a)$: The observation emission probability distribution $P(o_{t+1} = o \mid s_{t+1} = s', a_t = a)$, capturing forward 7-ray LiDAR rangefinder intersections and bearing measurements.

### 2.3 Multi-Agent MDP (MMDP)
When $N$ cooperative UAVs share full state information and are controlled by a centralized decision-maker ([Boutilier, 1996](https://dl.acm.org/doi/10.5555/237170.237197)):

$$\mathcal{M}_{\text{MMDP}} = \langle \mathcal{I}, \mathcal{S}, \{\mathcal{A}_i\}, \mathcal{T}, \mathcal{R}, \gamma \rangle$$

- $\mathcal{I} = \{1, \dots, N\}$: The finite set of $N$ UAV agents.
- $\mathcal{A} = \prod_{i \in \mathcal{I}} \mathcal{A}_i$: The joint action space of size $|\mathcal{A}| = 3^N$.
- $\mathcal{R}(s, \mathbf{a})$: Shared global team reward evaluating the collective performance of all UAVs.

### 2.4 Decentralized POMDP (Dec-POMDP)
When $N$ agents must choose actions independently based solely on their individual, local action-observation histories without communication ([Bernstein et al., 2002](https://doi.org/10.1287/moor.27.4.819.297); [Oliehoek & Amato, 2016](https://doi.org/10.1007/978-3-319-28929-8)):

$$\mathcal{M}_{\text{Dec-POMDP}} = \langle \mathcal{I}, \mathcal{S}, \{\mathcal{A}_i\}, \mathcal{T}, \mathcal{R}, \{\Omega_i\}, \mathcal{O}, \gamma \rangle$$

- $\Omega_i$: The local observation space of agent $i$ ($\mathbb{R}^{16}$).
- $\mathcal{O}(o_1, \dots, o_N \mid s', \mathbf{a})$: The joint observation emission probability function $P(\mathbf{o} \mid s', \mathbf{a})$.
- Each agent $i$ executes a decentralized policy $\pi_i(a_i \mid \tau_i)$, where $\tau_i = (o_i^0, a_i^0, \dots, o_i^t) \in (\Omega_i \times \mathcal{A}_i)^* \times \Omega_i$ is the agent's private history.
- The collective objective is to find a joint policy $\pi^* = \langle \pi_1^*, \dots, \pi_N^* \rangle$ that maximizes expected discounted cumulative team return:
  $$J(\pi) = \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t \mathcal{R}(s_t, \mathbf{a}_t) \;\Bigg|\; \mathbf{a}_t \sim \pi(\cdot \mid \pmb{\tau}_t) \right]$$

### 2.5 Decentralized POMDP with Communication (Dec-POMDP-Comm)
The canonical formulation of ScrollSwarm incorporates explicit inter-agent communication channels over a dynamic network graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ ([Xuan, Lesser, & Zilberstein, 2001](https://doi.org/10.1145/375735.376435); [Goldman & Zilberstein, 2004](https://doi.org/10.5555/1018430.1018453); [Oliehoek & Amato, 2016](https://doi.org/10.1007/978-3-319-28929-8)):

$$\mathcal{M}_{\text{Dec-POMDP-Comm}} = \langle \mathcal{I}, \mathcal{S}, \{\mathcal{A}_i\}, \mathcal{T}, \mathcal{R}, \{\Omega_i\}, \mathcal{O}, \Sigma, \mathcal{C}, \mathcal{G}, \gamma \rangle$$

- $\Sigma$: The communication message alphabet (or continuous latent vector space $\mathbb{R}^M$).
- $\mathcal{C}$: Optional communication cost function (in ScrollSwarm, $\mathcal{C} = 0$, modeling free localized broadcast).
- $\mathcal{G} = (\mathcal{V}, \mathcal{E})$: Dynamic communication topology (Chain, Ring, Radius, Capped, Full) determining allowed communication links between living agents.

  In each decision epoch, agents execute a two-stage step:
  
  1. **Communication Phase:** Every agent $i$ broadcasts message $m_i \in \Sigma$ to its graph neighbors $\mathcal{N}(i)$.
  2. **Action Phase:** Agent $i$ receives aggregated neighbor messages $\tilde{m}_i = \bigoplus_{j \in \mathcal{N}(i)} m_j$ and selects physical action $a_i \sim \pi_i(a_i \mid \tau_i, \tilde{m}_i)$.

---

## 3. Joint Observability & The Dec-MDP Property

A critical question in multi-agent reinforcement learning is whether the collection of local observations across all agents is sufficient to reconstruct the full underlying state of the environment.

According to Oliehoek & Amato ([2016, Definition 2.4](https://doi.org/10.1007/978-3-319-28929-8)), a Dec-POMDP is **jointly fully observable** (and thus constitutes a **Dec-MDP**) if the joint observation vector $\mathbf{o} = \langle o_1, \dots, o_N \rangle$ deterministically disambiguates the global state:

$$\forall \mathbf{o} \in \Omega_1 \times \dots \times \Omega_N, \quad \exists s \in \mathcal{S} \quad \text{such that} \quad P(s \mid \mathbf{o}) = 1$$

In ScrollSwarm, the global state vector provided by `env.state()` concatenates individual observations:

$$\mathbf{s} = \big[ \mathbf{o}_1^\top, \;\mathbf{o}_2^\top, \;\dots, \;\mathbf{o}_N^\top \big]^\top \in \mathbb{R}^{16\times N}$$

This property directly enables **Centralized Training with Decentralized Execution (CTDE)** frameworks such as **MAPPO** ([Yu et al., 2022](https://doi.org/10.1609/aaai.v36i8.20857)), **QMIX** ([Rashid et al., 2018](https://proceedings.mlr.press/v80/rashid18a.html)), and **MADDPG** ([Lowe et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html)), where a centralized value function $V(s)$ or centralized action-value critic $Q_{\text{tot}}(s, \mathbf{a})$ conditions on the joint observation during offline training, while decentralized actors $\pi_i(a_i \mid o_i)$ operate strictly on local observations during execution.

---

## 4. References & Academic Citations

1. **Bellman, R. (1957).** *A Markovian Decision Process.* Journal of Mathematics and Mechanics, 6(5), 679–684. [DOI: 10.1512/iumj.1957.6.56038](https://doi.org/10.1512/iumj.1957.6.56038).
2. **Puterman, M. L. (1994).** *Markov Decision Processes: Discrete Stochastic Dynamic Programming.* John Wiley & Sons. [DOI: 10.1002/9780470316887](https://doi.org/10.1002/9780470316887).
3. **Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998).** *Planning and acting in partially observable stochastic domains.* Artificial Intelligence, 101(1-2), 99–134. [DOI: 10.1016/S0004-3702(98)00023-X](https://doi.org/10.1016/S0004-3702(98)00023-X).
4. **Boutilier, C. (1996).** *Planning, learning and coordination in multiagent decision processes.* In Proceedings of the 6th Conference on Theoretical Aspects of Rationality and Knowledge (TARK), pp. 195–210. [ACM DL: 237197](https://dl.acm.org/doi/10.5555/237170.237197).
5. **Bernstein, D. S., Givan, R., Immerman, N., & Zilberstein, S. (2002).** *The complexity of decentralized control of Markov decision processes.* Mathematics of Operations Research, 27(4), 819–840. [DOI: 10.1287/moor.27.4.819.297](https://doi.org/10.1287/moor.27.4.819.297).
6. **Xuan, P., Lesser, V., & Zilberstein, S. (2001).** *Communication decisions in multi-agent cooperation: model and error analysis.* In Proceedings of the 5th International Conference on Autonomous Agents (AGENTS), pp. 516–523. [DOI: 10.1145/375735.376435](https://doi.org/10.1145/375735.376435).
7. **Goldman, C. V., & Zilberstein, S. (2004).** *Optimizing information exchange in cooperative multi-agent systems.* In Proceedings of the 2nd International Joint Conference on Autonomous Agents and Multiagent Systems (AAMAS), pp. 137–144. [ACM DL: 1018453](https://doi.org/10.5555/1018430.1018453).
8. **Oliehoek, F. A., & Amato, C. (2016).** *A Concise Introduction to Decentralized POMDPs.* SpringerBriefs in Intelligent Systems, Springer. [DOI: 10.1007/978-3-319-28929-8](https://doi.org/10.1007/978-3-319-28929-8).
9. **Kochenderfer, M. J., Wheeler, T. A., & Wray, K. H. (2015).** *Decision Making Under Uncertainty: Theory and Application.* MIT Press. [Book Link](https://mitpress.mit.edu/9780262029254/decision-making-under-uncertainty/).
10. **Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017).** *Multi-agent actor-critic for mixed cooperative-competitive environments.* In Advances in Neural Information Processing Systems (NeurIPS). [NeurIPS Link](https://proceedings.neurips.cc/paper/2017/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html).
11. **Rashid, T., Samvelyan, M., Schroeder, C., Farquhar, G., Foerster, J., & Whiteson, S. (2018).** *QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning.* In International Conference on Machine Learning (ICML). [PMLR Link](https://proceedings.mlr.press/v80/rashid18a.html).
12. **Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., & Wu, Y. (2022).** *The surprising effectiveness of PPO in cooperative multi-agent games.* In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI). [DOI: 10.1609/aaai.v36i8.20857](https://doi.org/10.1609/aaai.v36i8.20857).
