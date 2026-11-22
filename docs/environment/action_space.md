# Action Space & Kinematic Dynamics

Each agent $i$ possesses a discrete 1-dimensional action space $\mathcal{A}_i = \text{Discrete}(3)$ governing vertical motion.

---

## Action Set

| Action ID | Name | Semantic | Displacement $\Delta y$ |
| :---: | :---: | :--- | :---: |
| `0` | **`UP`** | Climb toward the cavern ceiling | $-k = -5\text{ px}$ |
| `1` | **`STAY`** | Maintain current vertical elevation | $0\text{ px}$ |
| `2` | **`DOWN`** | Dive toward the cavern floor | $+k = +5\text{ px}$ |

---

## Kinematic Update & Boundary Invariant

At each timestep, the agent's vertical coordinate $y_i$ updates according to:

$$y_i^{(t+1)} = \text{clip}\left( y_i^{(t)} + v_y \cdot k, \quad \frac{h_{\text{agent}}}{2}, \quad H - \frac{h_{\text{agent}}}{2} \right)$$

where $h_{\text{agent}} = 14\text{ px}$ is the agent's physical collision height, and $H = 400\text{ px}$ is the total viewport height.

---

## Physical Collision Model

The physical quadcopter is defined by:
- **Collision Radius:** $r_{\text{col}} = 8.0\text{ px}$ (circumscribed circle radius).
- **Physical Extents:** Width $w = 16.0\text{ px}$, height $h = 14.0\text{ px}$.

Collision checks are evaluated at every simulation step:
1. **Ceiling & Floor Boundaries:** Evaluated by testing the agent's bounding vertical span $[y_i - 7, y_i + 7]$ against $h_{\text{top}}[x_i]$ and $H - h_{\text{bot}}[x_i]$.
2. **Circular Tree Canopies:** Evaluated using Euclidean center-to-center distance:
   $$\|\mathbf{p}_i - \mathbf{p}_{\text{tree}}\| \le r_{\text{col}} + R_{\text{outer}}$$

Any collision results in **instant termination** (`alive = False`), removing the agent from future action calls and pruning its graph communication links.
