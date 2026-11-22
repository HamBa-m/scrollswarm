# Cavern Topography & Procedural Terrain

To stimulate meaningful multi-agent maneuvering, ScrollSwarm synthesizes procedural, undulating cavern corridors rather than static rectangular corridors.

---

## Multi-Scale Wave Formulation

The terrain height profiles are generated continuously via multi-frequency sinusoidal synthesis evaluated at global scrolling distance $t$:

### 1. Centerline Meander (Climbs & Dives)
The navigable corridor's center elevation $C(t)$ shifts vertically to create macroscopic peaks and valleys:

$$C(t) = \frac{H}{2} + A_1 \sin\left(\frac{2\pi t}{\lambda_1} + \phi_1\right) + A_2 \sin\left(\frac{2\pi t}{\lambda_2} + \phi_2\right)$$

- **Macro Wave:** $\lambda_1 = 540\text{ px}, \; A_1 = 65\text{ px}$ (large-scale gradual climbs and descents).
- **Meso Wave:** $\lambda_2 = 190\text{ px}, \; A_2 = 28\text{ px}$ (medium S-chicane undulations).

### 2. Dynamic Chokepoints (Width Modulation)
The navigable corridor opening $G(t)$ expands into wide open caverns and contracts into tight bottleneck gorges:

$$G(t) = \bar{G} + A_G \sin\left(\frac{2\pi t}{\lambda_G} + \phi_G\right)$$

- **Base Width:** $\bar{G} = 175\text{ px}$
- **Modulation Amplitude:** $A_G = 35\text{ px}$ (corridor pulses between $140\text{ px}$ and $210\text{ px}$)
- **Wavelength:** $\lambda_G = 380\text{ px}$

---

## Stalactites & Stalagmites (Cosine-Bell Crags)

Occasional protruding rock crags emerge from either the ceiling (stalactites) or floor (stalagmites). Each crag $c$ has a peak horizontal coordinate $x_c$, total base width $W_c$, and inward protrusion height $H_c \in [25, 45]\text{ px}$.

The protrusion height at coordinate $x$ is given by a smooth cosine-bell window:

$$\delta_c(x) = \begin{cases}
H_c \cdot \frac{1}{2} \left[ 1 + \cos\left( \frac{\pi |x - x_c|}{W_c / 2} \right) \right], & \text{if } |x - x_c| \le \frac{W_c}{2} \\
0, & \text{otherwise}
\end{cases}$$

---

## Guaranteed Passage Invariant

At every column $x \in [0, W)$, the simulation strictly verifies the minimal passage invariant:

$$\text{Clearance}(x) = H - \big( h_{\text{top}}[x] + h_{\text{bot}}[x] \big) \ge G_{\min} = 110\text{ px}$$

Because quadcopter agents have a height of $h_{\text{agent}} = 14\text{ px}$, a clearance of at least $110\text{ px}$ mathematically guarantees that a safe navigable passage exists across every single column of the viewport.
