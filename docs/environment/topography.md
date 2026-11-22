# Saturn Ring Topography & Dynamic Asteroid Kinematics

Rather than static or sinusoidal terrain, ScrollSwarm models an active exploration corridor within a planetary ring system (such as Saturn's main rings). The corridor is bounded by upper and lower radiation sheath planes and populated by dynamic, multi-scale polygon asteroids exhibiting realistic orbital-plane perturbations.

---

## 1. Radiation Sheath Boundaries

The vertical exploration corridor is delimited by upper and lower radiation sheath boundaries:

- **Ceiling Boundary:** $y_{\text{ceil}} = 15\text{ px}$
- **Floor Boundary:** $y_{\text{floor}} = H - 15\text{ px} = 385\text{ px}$

Any probe whose collision polygon penetrates $y \le y_{\text{ceil}}$ or $y \ge y_{\text{floor}}$ undergoes instantaneous structural termination.

---

## 2. Dynamic Polygon Asteroid Modeling

Asteroids and icy moonlets are modeled as irregular convex polygons generated via radial perturbation:

$$\mathbf{v}_k = \left( r_k \cos \theta_k, \; r_k \sin \theta_k \right), \quad k \in \{0, 1, \dots, V-1\}$$

where $V \in \{5, 6, 7\}$ vertices are sorted by angle $\theta_k \in [0, 2\pi)$ and jittered radially by $r_k \in [0.75 R, 1.20 R]$. Asteroid characteristic radii span a broad power-law spectrum:

$$R \in [R_{\min}, R_{\max}] = [14.0\text{ px}, 75.0\text{ px}]$$

---

## 3. Physical Foundations: Keplerian Drift & Hill's Epicyclic Motion

In a local rotating coordinate frame centered on a reference Keplerian orbit at semi-major axis $a_0$ with orbital mean motion $n = \sqrt{G M_{\text{central}} / a_0^3}$, the linearized equations of relative motion are given by **Hill's Equations** (also known as the Clohessy-Wiltshire equations):

$$\ddot{x} - 2n \dot{y} - 3n^2 x = 0$$

$$\ddot{y} + 2n \dot{x} = 0$$

$$\ddot{z} + n^2 z = 0$$

### Vertical Epicyclic Harmonic Oscillation

Notice that the vertical out-of-plane equation completely decouples from the in-plane radial ($x$) and azimuthal ($y$) dynamics:

$$\ddot{z} + n^2 z = 0$$

The exact analytical solution is simple harmonic oscillation:

$$z(t) = z_0 + A \sin(n t + \phi)$$

In ScrollSwarm's 2D coordinate frame, the vertical position of an asteroid at simulation step $t$ corresponds directly to this solution:

$$y(t) = y_{\text{base}} + A(R) \sin\big( \omega(R) \cdot t + \phi \big)$$

where:
- $\phi \sim \mathcal{U}(0, 2\pi)$ is the initial orbital inclination phase.
- $\omega(R) = \omega_0 + \Delta \omega \cdot \frac{R_{\max} - R}{R_{\max} - R_{\min}}$ is the size-dependent oscillation frequency.
- $A(R) \le 30\text{ px}$ is the vertical epicyclic amplitude.

### Size-Velocity Scaling & Thermal Equipartition

In dense planetary rings subject to frequent mutual collisions, velocity dispersion is governed by energy exchange toward thermal equipartition:

$$\frac{1}{2} m \sigma_v^2 \approx \text{const} \implies \sigma_v \propto m^{-1/2}$$

For quasi-two-dimensional ring shards where mass scales with area $m \propto R^2$, this yields an inverse scaling $\sigma_v \propto R^{-1}$. Small icy shards possess higher relative velocity dispersions than massive moonlets, drifting and oscillating substantially faster:

$$v_x(R) = v_0 + \Delta v \cdot \left(\frac{R_{\max} - R}{R_{\max} - R_{\min}}\right)$$

with $v_0 = 1.0\text{ px/step}$ and $\Delta v = 1.8\text{ px/step}$ ($v_x \in [1.0, 2.8]\text{ px/step}$).

---

## 4. Guaranteed Minimal Passage Invariant

To ensure that the environment remains fair and learnable, asteroid generation guarantees a minimum navigable opening across the vertical corridor at all times:

$$\text{Clearance}_{\min} \ge 110\text{ px}$$

Because rocket probes have a total height of $h_{\text{probe}} = 14\text{ px}$, this invariant guarantees that even when an asteroid is at the peak of its epicyclic oscillation amplitude $y_{\text{base}} \pm A$, at least one unobstructed corridor lane (either above or below the obstacle) provides a minimum clearance of $110\text{ px}$.

---

## Academic References & Citations

1. **Hill, G. W. (1878).** *Researches in the Lunar Theory*. American Journal of Mathematics, 1(1), 5–26.
2. **Murray, C. D., & Dermott, S. F. (1999).** *Solar System Dynamics*. Cambridge University Press (Chapters 3 & 10: "The Three-Body Problem" and "Planetary Ring Dynamics").
3. **Goldreich, P., & Tremaine, S. (1982).** *The dynamics of planetary rings*. Annual Review of Astronomy and Astrophysics, 20(1), 249–283.
4. **Stewart, G. R., Lin, D. N., & Bodenheimer, P. (1984).** *Collision-induced transport processes in planetary rings*. In R. Greenberg & A. Brahic (Eds.), *Planetary Rings* (pp. 447–497). University of Arizona Press.
5. **Salo, H. (1992).** *Numerical simulations of dense collisional systems: An application to Saturn's rings*. The Astronomical Journal, 104, 1964–1972.
