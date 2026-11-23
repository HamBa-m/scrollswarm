# Action Space & Kinematics

This page details the discrete action space, kinematic transition dynamics, vertical thrust equations, and joint action space formulations for the multi-agent drone swarm in ScrollSwarm.

---

## 1. Discrete Action Space

Each autonomous UAV $i \in \{1, \dots, N\}$ operates over a discrete action space conforming to Gymnasium and PettingZoo standards:

$$\mathcal{A}_i = \{0, 1, 2\} \equiv \{\text{UP}, \;\text{STAY}, \;\text{DOWN}\} \quad \big(\mathcal{A}_i \cong \text{Discrete}(3)\big)$$

| Action Index | Identifier | Vertical Velocity / Thrust | Physical Displacement $\Delta y$ | Description |
| :---: | :---: | :---: | :---: | :--- |
| **`0`** | `Action.UP` | Negative vertical velocity | $\Delta y = -k\text{ px}$ | Ascends (climbs) toward higher altitude |
| **`1`** | `Action.STAY` | Zero vertical velocity | $\Delta y = 0\text{ px}$ | Maintains current flight altitude |
| **`2`** | `Action.DOWN` | Positive vertical velocity | $\Delta y = +k\text{ px}$ | Descends (dives) toward lower altitude |

By default, the vertical step size is $k = 5\text{ px/step}$.

---

## 2. Kinematic Transition Equations

### 2.1 State Evolution
At time step $t$, UAV $i$ is positioned at $(x_i, y_i(t))$. Because UAVs occupy fixed longitudinal station-keeping columns along the $x$-axis, horizontal coordinates remain fixed in the screen frame:

$$x_i(t+1) = x_i(t)$$

Vertical altitude evolves according to the selected discrete action $a_{i, t} \in \mathcal{A}_i$:

$$y_{i, t+1} = \text{clip}\left( y_{i, t} + \Delta y(a_{i, t}), \quad \frac{h_{\text{uav}}}{2}, \quad H - \frac{h_{\text{uav}}}{2} \right)$$

Where:

- $h_{\text{uav}} = 16.0\text{ px}$ is the physical bounding height of the UAV.
- $H = 400\text{ px}$ is the total corridor height.
- Clamping ensures the drone geometry physically remains within the screen bounds even before boundary collision checks occur.

### 2.2 Terminated / Crashed UAV Invariant
If a UAV collides with an obstacle or corridor boundary at time $t$, its operational state transitions to `alive = False`:

$$\text{If } \text{alive}_i(t) = \text{False} \implies y_{i, t+1} = y_{i, t}, \quad \text{no actions executed}$$

Terminated drones become inactive obstacles on the field; their actions are ignored, and they are pruned from `env.agents`.

---

## 3. Joint Action Space 

In multi-agent systems with $N$ drones, the joint action space $\pmb{\mathcal{A}}$ is the Cartesian product of the individual action spaces ([Boutilier, 1996](https://dl.acm.org/doi/10.5555/237170.237197); [Oliehoek & Amato, 2016](https://doi.org/10.1007/978-3-319-28929-8)):

$$\pmb{\mathcal{A}} = \mathcal{A}_1 \times \mathcal{A}_2 \times \dots \times \mathcal{A}_N = \{0, 1, 2\}^N$$

The cardinality of the joint action space grows exponentially with swarm size $N$:

$$|\pmb{\mathcal{A}}| = 3^N$$

| Swarm Size $N$ | Joint Action Combinations $\|\pmb{\mathcal{A}}\|$ | Complexity Class |
| :---: | :---: | :--- |
| **1** | $3^1 = 3$ | Standard single-agent discrete space |
| **3** | $3^3 = 27$ | Easily solvable via centralized value iteration |
| **5** (Default) | $3^5 = 243$ | Combinatorial multi-agent coordination |
| **8** | $3^8 = 6,561$ | Requires factorized value functions (e.g., QMIX, VDN) |
| **10** (Maximum) | $3^{10} = 59,049$ | High-dimensional joint action space |

This combinatorial growth illustrates why independent decentralized execution or factorized value decomposition is necessary for large swarms.

---

## 4. References

1. **Boutilier, C. (1996).** *Planning, learning and coordination in multiagent decision processes.* In Proceedings of the 6th Conference on Theoretical Aspects of Rationality and Knowledge (TARK), pp. 195–210. [ACM DL: 237197](https://dl.acm.org/doi/10.5555/237170.237197).
2. **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction (2nd Edition).* MIT Press. [Book Link](http://incompleteideas.net/book/the-book-2nd.html).
3. **Oliehoek, F. A., & Amato, C. (2016).** *A Concise Introduction to Decentralized POMDPs.* SpringerBriefs in Intelligent Systems, Springer. [DOI: 10.1007/978-3-319-28929-8](https://doi.org/10.1007/978-3-319-28929-8).
