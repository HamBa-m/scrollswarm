# Action Space & Kinematic Dynamics

Each rocket probe $i$ possesses a discrete 1-dimensional action space $\mathcal{A}_i = \text{Discrete}(3)$ governing vertical motion.

---

## Action Set

| Action ID | Name | Semantic | Displacement $\Delta y$ |
| :---: | :---: | :--- | :---: |
| `0` | **`UP`** | Vertical thrust toward upper radiation sheath | $-k = -5\text{ px}$ |
| `1` | **`STAY`** | Maintain current vertical elevation | $0\text{ px}$ |
| `2` | **`DOWN`** | Vertical dive toward lower radiation sheath | $+k = +5\text{ px}$ |

---

## Kinematic Update & Boundary Invariant

At each timestep, the probe's vertical coordinate $y_i$ updates according to:

$$y_i^{(t+1)} = \text{clip}\left( y_i^{(t)} + v_y \cdot k, \quad \frac{h_{\text{probe}}}{2}, \quad H - \frac{h_{\text{probe}}}{2} \right)$$

where $h_{\text{probe}} = 14\text{ px}$ is the probe's collision height, and $H = 400\text{ px}$ is the total viewport height.

---

## Physical Collision Model

The physical rocket probe is defined by:
- **Collision Radius:** $r_{\text{col}} = 8.0\text{ px}$ (circumscribed circle radius).
- **Physical Extents:** Width $w = 16.0\text{ px}$, height $h = 14.0\text{ px}$.

Collision checks are evaluated at every simulation step:
1. **Radiation Sheath Boundaries:** Evaluated by testing the probe's bounding vertical span $[y_i - 7, y_i + 7]$ against the upper sheath ($y_{\text{ceil}} = 15\text{ px}$) and lower sheath ($y_{\text{floor}} = 385\text{ px}$).
2. **Dynamic Polygon Asteroids:** Evaluated using point-in-polygon containment and minimum Euclidean distance from probe center to polygon edges:
   $$\text{dist}(\mathbf{p}_i, \text{Asteroid}) \le r_{\text{col}}$$

Any collision results in **instant structural termination** (`alive = False`), removing the probe from future action calls and pruning its graph communication links.
