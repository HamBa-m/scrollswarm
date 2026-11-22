# Observation Space & Sensory Model

Each rocket probe $i$ is equipped with an ego-centric forward LiDAR rangefinder that scans the oncoming half of the screen ($R_{\max} = 400\text{ px}$) across a field of view of $120^\circ$.

---

## 7-Ray LiDAR Geometry

The sensor projects $K = 7$ forward rays at equi-angular increments spanning from $-60^\circ$ (up-forward) to $+60^\circ$ (down-forward):

$$\theta_k = -\frac{\pi}{3} + k \cdot \frac{\pi}{9}, \quad k \in \{0, 1, \dots, 6\}$$

```text
               \   Ray 0: -60° (Ceiling Sheath / Upper Asteroids)
                \  Ray 1: -40°
                 \ Ray 2: -20°
Probe [>] ======== Ray 3:   0° (Corridor Centerline / Oncoming Shards)
                 / Ray 4: +20°
                /  Ray 5: +40°
               /   Ray 6: +60° (Floor Sheath / Lower Asteroids)
```

For each ray $k$, a parametric ray is projected from the probe center $(x_i, y_i)$ with unit direction $\mathbf{d}_k = (\cos \theta_k, \sin \theta_k)$:

$$\mathbf{p}_k(t) = (x_i + t \cos \theta_k, \; y_i + t \sin \theta_k), \quad t \in [0, R_{\max}]$$

---

## Ray-Obstacle Intersection Math

### 1. Radiation Sheath Boundary Intersections

Because the upper and lower sheath boundaries are horizontal planes ($y = y_{\text{ceil}}$ and $y = y_{\text{floor}}$), intersection distances are computed analytically in $\mathcal{O}(1)$:

$$t_{\text{ceil}} = \frac{y_{\text{ceil}} - y_i}{\sin \theta_k} \quad (\text{for } \sin \theta_k < 0)$$

$$t_{\text{floor}} = \frac{y_{\text{floor}} - y_i}{\sin \theta_k} \quad (\text{for } \sin \theta_k > 0)$$

### 2. Exact 2D Ray-Segment Determinant Intersection for Polygon Asteroids

Each asteroid is bounded by edge segments $\mathbf{e} = [\mathbf{a}, \mathbf{b}]$. The parametric intersection between ray $\mathbf{p}(t) = \mathbf{p}_0 + t \mathbf{d}$ and segment $\mathbf{s}(u) = \mathbf{a} + u (\mathbf{b} - \mathbf{a})$ ($u \in [0, 1]$) satisfies:

$$\mathbf{p}_0 + t \mathbf{d} = \mathbf{a} + u \mathbf{v}, \quad \text{where } \mathbf{v} = \mathbf{b} - \mathbf{a}$$

Rearranging into 2D linear form and solving via Cramer's rule / 2D cross-product determinant yields:

$$D = \mathbf{d} \times \mathbf{v} = d_x v_y - d_y v_x$$

If $|D| \ge 10^{-6}$, the ray and edge are non-parallel. Letting $\Delta = \mathbf{a} - \mathbf{p}_0$:

$$t = \frac{\Delta \times \mathbf{v}}{D} = \frac{\Delta_x v_y - \Delta_y v_x}{D}$$

$$u = \frac{\Delta \times \mathbf{d}}{D} = \frac{\Delta_x d_y - \Delta_y d_x}{D}$$

A valid intersection occurs if and only if:

$$0 \le t \le R_{\max} \quad \text{and} \quad 0 \le u \le 1$$

To maximize evaluation throughput, candidate asteroids first undergo bounding circle rejection tests ($d_{\perp} \le R$) before testing individual edge determinants.

### 3. Unified Obstacle Distance Channel

Sheath boundary hits and asteroid polygon edge hits are unified into a single continuous reading:

$$d_{\text{obs}, k} = \frac{\min(t_{\text{boundary}}, t_{\text{asteroid}}, R_{\max})}{R_{\max}} \in [0.0, 1.0]$$

A value of $1.0$ indicates an entirely clear path ahead; values close to $0.0$ indicate imminent collision hazard.

### 4. Telemetry Beacon Detection Channel

Cassini telemetry beacons are circular targets of radius $r_{\text{tgt}} = 8\text{ px}$. If ray $k$ intersects an active target canister, its normalized distance is recorded:

$$d_{\text{tgt}, k} = \frac{\min(t_{\text{target}}, R_{\max})}{R_{\max}} \in [0.0, 1.0]$$

If no target lies along ray $k$, $d_{\text{tgt}, k} = 1.0$.

---

## 15-Dimensional Observation Vector

At each timestep, probe $i$ receives the continuous vector $\mathbf{o}_i \in [0, 1]^{15}$:

$$\mathbf{o}_i = \left[ \frac{y_i}{H}, \; d_{\text{obs}, 0}, \dots, d_{\text{obs}, 6}, \; d_{\text{tgt}, 0}, \dots, d_{\text{tgt}, 6} \right]$$

| Index | Name | Range | Description |
| :---: | :--- | :---: | :--- |
| `0` | Normalized Elevation | $[0.0, 1.0]$ | Vertical position $y_i / H$ ($0.0 = \text{top}, 1.0 = \text{bottom}$) |
| `1`–`7` | Obstacle Distances ($d_{\text{obs}, 0..6}$) | $[0.0, 1.0]$ | Normalized distance to nearest ceiling, floor, or asteroid along rays $-60^\circ$ to $+60^\circ$ |
| `8`–`14` | Target Distances ($d_{\text{tgt}, 0..6}$) | $[0.0, 1.0]$ | Normalized distance to nearest telemetry beacon along rays $-60^\circ$ to $+60^\circ$ |
