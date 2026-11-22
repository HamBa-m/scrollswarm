# Observation Space & Sensory Model

Each agent $i$ is equipped with an ego-centric forward LiDAR rangefinder that scans the oncoming half of the screen ($R_{\max} = 400\text{ px}$) across a field of view of $120^\circ$.

---

## 7-Ray LiDAR Geometry

The sensor projects $K = 7$ forward rays at equi-angular increments spanning from $-60^\circ$ (up-forward) to $+60^\circ$ (down-forward):

$$\theta_k = -\frac{\pi}{3} + k \cdot \frac{\pi}{9}, \quad k \in \{0, 1, \dots, 6\}$$

```text
               \   Ray 0: -60° (Ceiling / Upper Crags)
                \  Ray 1: -40°
                 \ Ray 2: -20°
Agent [>] ======= Ray 3:   0° (Corridor Centerline / Oncoming Obstacles)
                 / Ray 4: +20°
                /  Ray 5: +40°
               /   Ray 6: +60° (Floor / Lower Crags)
```

For each ray $k$, a parametric line is cast from the agent's center $(x_i, y_i)$:

$$\mathbf{p}_k(t) = (x_i + t \cos \theta_k, \; y_i + t \sin \theta_k), \quad t \in [0, R_{\max}]$$

---

## Ray-Obstacle Intersection Math

1. **Mountain Boundary Intersections:**
   For each integer sample $x = \lfloor x_i + t \cos \theta_k \rfloor$, the ray's vertical coordinate $y_k(t)$ is evaluated against:
   $$y_k(t) \le h_{\text{top}}[x] \quad \text{or} \quad y_k(t) \ge H - h_{\text{bot}}[x]$$
   The smallest $t \ge 0$ satisfying the inequality defines the boundary distance.

2. **Circular Tree Obstacle Intersections:**
   For a tree canopy at $(x_c, y_c)$ with radius $R_c$, the closest approach distance along the ray vector is computed:
   $$t_{\text{proj}} = (x_c - x_i)\cos \theta_k + (y_c - y_i)\sin \theta_k$$
   If the perpendicular distance squared $d_{\perp}^2 = \|\mathbf{c} - \mathbf{p}_i\|^2 - t_{\text{proj}}^2 < R_c^2$, the intersection distance is:
   $$t_{\text{entry}} = t_{\text{proj}} - \sqrt{R_c^2 - d_{\perp}^2}$$

3. **Unified Obstacle Distance Channel:**
   Boundary collisions and tree obstacles are unified into a single continuous reading:
   $$d_{\text{obs}, k} = \frac{\min(t_{\text{mountain}}, t_{\text{tree}}, R_{\max})}{R_{\max}} \in [0.0, 1.0]$$
   A value of $1.0$ indicates an entirely clear path ahead; values close to $0.0$ indicate imminent collision hazard.

4. **Target Detection Channel:**
   Floating collection targets are circular colliders of radius $r_{\text{tgt}} = 10\text{ px}$. If a ray intersects a target, its normalized distance is recorded:
   $$d_{\text{tgt}, k} = \frac{\min(t_{\text{target}}, R_{\max})}{R_{\max}} \in [0.0, 1.0]$$
   If no target lies along ray $k$, $d_{\text{tgt}, k} = 1.0$.

---

## 15-Dimensional Observation Vector

At each timestep, agent $i$ receives the vector $\mathbf{o}_i \in [0, 1]^{15}$:

$$\mathbf{o}_i = \left[ \frac{y_i}{H}, \; d_{\text{obs}, 0}, \dots, d_{\text{obs}, 6}, \; d_{\text{tgt}, 0}, \dots, d_{\text{tgt}, 6} \right]$$

| Index | Name | Range | Description |
| :---: | :--- | :---: | :--- |
| `0` | Normalized Elevation | $[0.0, 1.0]$ | Vertical position $y_i / H$ ($0.0 = \text{top}, 1.0 = \text{bottom}$) |
| `1`–`7` | Obstacle Distances ($d_{\text{obs}, 0..6}$) | $[0.0, 1.0]$ | Normalized distance to nearest ceiling, floor, or tree along rays $-60^\circ$ to $+60^\circ$ |
| `8`–`14` | Target Distances ($d_{\text{tgt}, 0..6}$) | $[0.0, 1.0]$ | Normalized distance to nearest target along rays $-60^\circ$ to $+60^\circ$ |
