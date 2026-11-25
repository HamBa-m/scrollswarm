# Environment Dynamics

This page defines the 2D corridor geometry, scrolling kinematics, procedural obstacles, guaranteed navigable gap proof, and collision checks.

---

## 1. Coordinate Frame & World Geometry

ScrollSwarm is modeled in a 2D Cartesian coordinate system:

- **Horizontal Axis ($x$):** Represents forward distance along the flight path (in pixels/meters). Space flows from right to left as the environment scrolls.
- **Vertical Axis ($y$):** Represents flight altitude, measured downwards from the screen ceiling ($y = 0$) to the ground floor ($y = H$).

$$\mathcal{W} = \big\{ (x, y) \in \mathbb{R}^2 \;\big|\; 0 \le x \le W, \quad y_{\text{ceil}} \le y \le y_{\text{floor}} \big\}$$

Where:

- $W$: Corridor width (default: $800\text{ px}$).
- $H$: Corridor total height (default: $400\text{ px}$).
- $y_{\text{ceil}}$: High-altitude boundary ceiling ($y_{\text{ceil}} = 15\text{ px}$).
- $y_{\text{floor}}$: Ground surface boundary ($y_{\text{floor}} = H - 15 = 385\text{ px}$).

---

## 2. World Kinematics & Scrolling Evolution

The simulation advances in discrete time steps $\Delta t$ at a nominal rate of $30\text{ FPS}$.

### 2.1 Forward Scrolling Motion
The corridor scrolls horizontally leftward at constant velocity $v_{\text{scroll}}$:

$$x_{t+1} = x_t - v_{\text{scroll}} \cdot \Delta t$$

By default, $v_{\text{scroll}} = 1.0\text{ px/step}$. Because all drones occupy stationary horizontal observation columns on the left portion of the screen ($x \in [80, 320]$), the relative approach velocity of obstacles and targets toward the swarm is precisely $v_{\text{scroll}}$.

### 2.2 Procedural Obstacle Models
Obstacles represent physical hazards of varying geometries and scales:

1. **Ground-Standing Structures:**

    - Modeled as axis-aligned rectangles rooted at the ground plane $y_{\text{floor}}$: $\text{Rect}_{\text{ground}} = \big[ x - w/2, \;x + w/2 \big] \times \big[ y_{\text{floor}} - h, \;y_{\text{floor}} \big]$
   - Width $w \sim \mathcal{U}(30.0, 70.0)\text{ px}$.
   - Height $h \sim \mathcal{U}(40.0, h_{\max})\text{ px}$.

2. **Overhead Structures:**
    - Modeled as rectangular barriers hanging down from the ceiling $y_{\text{ceil}}$:
         $\text{Rect}_{\text{overhead}} = \big[ x - w/2, \;x + w/2 \big] \times \big[ y_{\text{ceil}}, \;y_{\text{ceil}} + d \big]$
     - Width $w \sim \mathcal{U}(30.0, 70.0)\text{ px}$.
     - Depth $d \sim \mathcal{U}(30.0, d_{\max})\text{ px}$.

3. **Floating Debris & Airborne Hazards:**
Modeled as convex polygons with radius $r \sim \mathcal{U}(14.0, 32.0)\text{ px}$ undergoing epicyclic vertical oscillation:  $y(t) = y_{\text{base}} + A \sin(\omega t + \phi)$, where amplitude $A \sim \mathcal{U}(4.0, 16.0)\text{ px}$, frequency $\omega \sim \mathcal{U}(0.02, 0.06)\text{ rad/step}$, and initial phase $\phi \sim \mathcal{U}(0, 2\pi)$.

---

## 3. Guaranteed Navigable Passage Invariant

A critical benchmark design requirement in autonomous navigation is that an environment must never generate provably impassable bottlenecks ([Siegwart et al., 2011](https://mitpress.mit.edu/9780262015356/introduction-to-autonomous-mobile-robots/); [LaValle, 2006](http://planning.cs.uiuc.edu/)).

### 3.1 Theorem: Passage Solvability Invariant
Let $G(x, t)$ denote the vertical gap between the lowest point of any overhead obstacle and the highest point of any ground obstacle at horizontal coordinate $x$:

$$G(x, t) = y_{\text{ground_top}}(x, t) - y_{\text{overhead_bottom}}(x, t)$$

**Invariant:** For all horizontal positions $x \in [0, W]$ and all time steps $t \ge 0$:

$$G(x, t) \ge G_{\min} = 110\text{ px}$$

_Proof_: When an overhead structure with depth $d$ and a ground structure with height $h$ coexist at coordinate $x$:

1. The available vertical height between baseline boundaries is $\Delta Y = y_{\text{floor}} - y_{\text{ceil}} = 385 - 15 = 370\text{ px}$.
2. In `_create_procedural_obstacle()`, the maximum allowable structure dimension is bounded by:
   $h_{\max} = d_{\max} = \Delta Y - G_{\min} - 10 = 370 - 110 - 10 = 250\text{ px}$
3. Spawning intervals strictly enforce alternating placement or vertical margin checks. Therefore:
   $G(x, t) = \Delta Y - (h + d) \ge 370 - 250 = 120\text{ px} > 110\text{ px}$

Hence, the corridor unconditionally satisfies $G(x, t) \ge 110\text{ px}$, guaranteeing sufficient clearance for UAV quadcopters (collision height $16\text{ px}$).

---

## 4. Analytical Collision Detection

Physical collision between a drone and an obstacle is computed using exact point-to-polygon Euclidean distance and ray-casting containment tests:

1. **Bounding Circle Rejection:**
   If $\sqrt{(x_{\text{uav}} - x_{\text{obs}})^2 + (y_{\text{uav}} - y_{\text{obs}})^2} > R_{\text{uav}} + R_{\text{obs}}$, no collision is possible.
2. **Ray-Casting Inside-Polygon Test:**
   A horizontal ray is cast from $(x_{\text{uav}}, y_{\text{uav}})$ to $+\infty$. If the ray intersects an odd number of polygon boundary edges, the drone is inside the obstacle (Jordan Curve Theorem).
3. **Point-to-Segment Edge Distance:**
   For each polygon edge segment $\mathbf{e} = (\mathbf{v}_a, \mathbf{v}_b)$, the minimum distance to $(x_{\text{uav}}, y_{\text{uav}})$ is computed analytically via vector projection:

$$\mathbf{p}_{\text{proj}} = \mathbf{v}_a + \text{clip}\left(\frac{(\mathbf{p} - \mathbf{v}_a) \cdot (\mathbf{v}_b - \mathbf{v}_a)}{\|\mathbf{v}_b - \mathbf{v}_a\|^2}, \;0, \;1\right) (\mathbf{v}_b - \mathbf{v}_a)$$

  If $\min_e \|\mathbf{p} - \mathbf{p}_{\text{proj}}\| \le R_{\text{uav}}$, a collision occurs.

---

## 5. References

1. **LaValle, S. M. (2006).** *Planning Algorithms.* Cambridge University Press. [Cambridge / Author Link](http://planning.cs.uiuc.edu/) / [DOI: 10.1017/CBO9780511546877](https://doi.org/10.1017/CBO9780511546877).
2. **Siegwart, R., Nourbakhsh, I. R., & Scaramuzza, D. (2011).** *Introduction to Autonomous Mobile Robots (2nd Edition).* MIT Press. [MIT Press Link](https://mitpress.mit.edu/9780262015356/introduction-to-autonomous-mobile-robots/).
