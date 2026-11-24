# Tutorial 04: Computational Geometry Acceleration

In Multi-Agent Reinforcement Learning (MARL), sample throughput is often the primary bottleneck determining training duration. Complex policy gradient algorithms (MAPPO, QMIX, HAPPO) require millions of environment transitions to converge, meaning environment stepping latency directly dictates research velocity.

In 2D aerial simulations involving LiDAR sensor emulation and complex procedural terrain, **analytical geometric calculations** (raycasting, line-segment intersections, and point-in-polygon queries) represent the vast majority of CPU execution time.

This tutorial guides you through:

1. **Profiling the Computational Bottleneck:** Analyzing why analytical 2D raycasting dominates simulation cycles.
2. **The Hardware Abstraction Layer:** Understanding the multi-backend dispatch architecture (`numpy`, `numba`, `cpp`, `auto`).
3. **Compilation & Installation:** Building the high-performance C++20 native kernel and enabling Numba LLVM JIT.
4. **Environment Integration:** Seamlessly selecting and hot-swapping acceleration backends in `ScrollSwarmEnv`.
5. **Direct Micro-Kernel Invocation:** Using the geometry subsystem directly for custom raycasting and trajectory planning.
6. **Empirical Benchmarks:** Evaluating micro-latency ($\mu\text{s}$), stepping throughput ($\text{steps/s}$), and scaling properties.
7. **Numerical Equivalence & Float32 Parity:** Understanding cross-backend precision guarantees and fast-math tolerances.

---

## 1. The Computational Bottleneck: Why Geometry Dominates?

In ScrollSwarm, each active UAV continuously perceives its local airspace through forward-facing LiDAR sensor rays. 

At every discrete environment step $t$:

- $N$ active drones in the fleet.
- $K = 7$ forward ray beams per drone (spanning a $120^\circ$ field of view).
- $M$ procedural polygon obstacles, each with $V_m$ vertices and bounding edges.
- Upper ceiling and lower floor piecewise linear terrain segments.

$$\text{Ray Tests per Step} = N \times K \times \sum_{m=1}^M V_m + \text{Boundary Intersections}$$

For a standard swarm of $N=4$ drones operating near 5 polygonal obstacles with 4–6 vertices each, every step performs **hundreds of parametric line-segment intersection tests**:

$$\mathbf{p} + t \mathbf{r} = \mathbf{q} + u \mathbf{s} \implies t = \frac{(\mathbf{q} - \mathbf{p}) \times \mathbf{s}}{\mathbf{r} \times \mathbf{s}}$$

In addition, micro-step kinematic integration validates whether drone hitboxes penetrate any obstacle polygon (via ray-crossing parity tests) or violate the minimum clearance margin (via point-to-polygon distance queries).

```mermaid
pie title CPU Step Execution Time Breakdown (Pure Python Baseline)
    "Raycasting & LiDAR Geometry" : 79.3
    "Collision Hitbox & Boundaries" : 9.8
    "Relational Graph (PyG edge_index)" : 5.4
    "PettingZoo State & Dict Packaging" : 3.6
    "Kinematics & Drift Dynamics" : 1.9
```

When evaluated using pure Python and standard NumPy loops, analytical geometry accounts for **$79.3\%$ of total step latency**. Accelerating these geometric routines produces dramatic, order-of-magnitude improvements in overall environment throughput.

---

## 2. Multi-Backend Architecture: Pluggable Hardware Abstraction

ScrollSwarm decouples high-level PettingZoo environment logic from low-level geometric computations via a unified dispatch layer in `scrollswarm.geometry`.

```mermaid
flowchart TD
    ENV["<b>ScrollSwarmEnv / MultiRayLiDAR</b><br/>High-Level Environment & Sensors"]
    DISPATCH["<b>Geometry Dispatch Layer</b><br/>scrollswarm.geometry.resolve_backend()"]

    subgraph BACKENDS ["Pluggable Computational Engines"]
        direction TB
        CPP["<b>Native C++20 Kernel</b><br/>• pybind11 Zero-Copy Array Views<br/>• Static Runtime Linkage (-O3)"]
        NUMBA["<b>Numba LLVM JIT</b><br/>• Automatic CPU Vectorization<br/>• ~1.5s Initial Compilation"]
        NUMPY["<b>Pure NumPy Baseline</b><br/>• Vectorized Analytical Trigonometry<br/>• Zero Compilation / Dependency Free"]
    end

    ENV -->|Passes Flattened Geometry Buffers| DISPATCH
    DISPATCH -->|backend='cpp'| CPP
    DISPATCH -->|backend='numba'| NUMBA
    DISPATCH -->|backend='numpy'| NUMPY

    classDef envStyle fill:#1a2332,stroke:#38bdf8,stroke-width:1.5px,color:#f1f5f9;
    classDef dispatchStyle fill:#1a2332,stroke:#818cf8,stroke-width:1.5px,color:#f1f5f9;
    classDef backendStyle fill:#111827,stroke:#34d399,stroke-width:1.5px,color:#f1f5f9;

    class ENV envStyle;
    class DISPATCH dispatchStyle;
    class CPP,NUMBA,NUMPY backendStyle;
```

### Contiguous Buffer Flattening

To enable zero-copy passing into compiled native kernels (C++ and LLVM machine code), ScrollSwarm provides `extract_geometry_buffers()`:

```python
from scrollswarm.geometry import extract_geometry_buffers

# Transforms dynamic Python polygon objects into flat contiguous C-arrays
obs_meta, obs_edges, obs_ranges, target_data = extract_geometry_buffers(
    obstacles=scrolling_map.obstacles,
    targets=scrolling_map.targets,
)
```

- **`obs_meta`** (`np.ndarray[np.float32]`, shape `(M, 6)`): Polygon bounding boxes $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$, edge start offset, and edge count.
- **`obs_edges`** (`np.ndarray[np.float32]`, shape `(TotalEdges, 4)`): Contiguous segment coordinates $(x_1, y_1, x_2, y_2)$ for all polygon facets.
- **`obs_ranges`** (`np.ndarray[np.float32]`, shape `(M, 2)`): Horizontal span $[x_{\min}, x_{\max}]$ for early broad-phase rejection.
- **`target_data`** (`np.ndarray[np.float32]`, shape `(T, 4)`): Target center coordinates $(x, y)$, inspection radius, and active status.

---

## 3. Installation & Compilation of Acceleration Kernels

ScrollSwarm requires **zero compiler tools** to run out-of-the-box using the NumPy baseline. Depending on your computing cluster or workstation setup, you can enable either Numba or C++20 acceleration.

### Option A: Numba LLVM JIT Acceleration

*(Recommended for Quick Setup)* Numba compiles Python functions directly into optimized machine instructions via LLVM without requiring an external C++ toolchain:

```bash
# Add Numba to your active virtual environment
uv add numba
```

Once installed, ScrollSwarm automatically detects Numba and activates the `@njit(fastmath=True, nogil=True)` execution kernels.

### Option B: Native C++20 Kernel Compilation

For production training clusters where compilation warmup latency is undesirable, ScrollSwarm includes a header-only C++20 geometry kernel (`scrollswarm/geometry/cpp/geometry.hpp`) bound via `pybind11`.

To build the native binary on your machine:

```bash
uv run python scripts/build_geometry_cpp.py
```

The build script automatically detects the host C++ toolchain (GCC, Clang, or MSVC) and compiles `_geometry_cpp` with high-performance optimization flags:

```bash
# Example MinGW GCC compile command executed by the builder:
g++ -O3 -DNDEBUG -std=c++20 -shared -fPIC \
    -march=native -ffast-math \
    -static-libgcc -static-libstdc++ -static \
    -I<python-include> -I<pybind11-include> \
    scrollswarm/geometry/cpp/bindings.cpp \
    -o scrollswarm/geometry/_geometry_cpp.pyd
```

!!! tip "Static Runtime Linkage"
    The build script uses `-static-libgcc -static-libstdc++ -static` on Windows/MinGW. This embeds all C++ runtime symbols directly into the shared library, ensuring portability without external runtime `.dll` or `.so` dependencies.

---

## 4. Configuring Backends in `ScrollSwarmEnv`

You can specify the desired backend when creating the environment using the `backend` keyword argument:

```python
from scrollswarm import ScrollSwarmEnv
from scrollswarm.geometry import get_available_backends

# Check which backends are operational on the host system
print("Available backends:", get_available_backends())
# Output: ['numpy', 'numba', 'cpp']

# 1. Automatic Selection (Default: selects fastest available: cpp -> numba -> numpy)
env = ScrollSwarmEnv(num_agents=4, backend="auto")

# 2. Explicit C++20 Native Kernel
env_cpp = ScrollSwarmEnv(num_agents=4, backend="cpp")

# 3. Explicit Numba LLVM JIT
env_numba = ScrollSwarmEnv(num_agents=4, backend="numba")

# 4. Explicit Pure NumPy Portable Baseline
env_numpy = ScrollSwarmEnv(num_agents=4, backend="numpy")
```

### Complete End-to-End Stepping & Profiling Script

The following script initializes environments across all available backends, steps them through identical random policies, and logs the execution duration:

```python
import time
from scrollswarm import ScrollSwarmEnv
from scrollswarm.geometry import get_available_backends

def benchmark_env(backend: str, num_steps: int = 500, num_agents: int = 4):
    env = ScrollSwarmEnv(
        num_agents=num_agents,
        max_cycles=num_steps + 100,
        backend=backend,
    )
    env.reset(seed=42)

    # Warmup step
    actions = {a: 1 for a in env.agents}
    env.step(actions)

    # Measured run
    start_time = time.perf_counter()
    steps_completed = 0

    while steps_completed < num_steps and env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        steps_completed += 1

    elapsed = time.perf_counter() - start_time
    steps_per_sec = steps_completed / elapsed
    ms_per_step = (elapsed / steps_completed) * 1000.0

    print(f"Backend [{backend:5s}]: {steps_completed:4d} steps | "
          f"{steps_per_sec:7.1f} steps/s | {ms_per_step:6.3f} ms/step")
    env.close()

if __name__ == "__main__":
    for b in get_available_backends():
        benchmark_env(backend=b, num_steps=500)
```

---

## 5. Direct Micro-Kernel Invocation 

Researchers building custom trajectory optimizers, model-predictive control (MPC) loops, or non-PettingZoo simulators can invoke the optimized geometry kernels directly.

### LiDAR Raycasting Kernel

```python
import numpy as np
from scrollswarm.geometry import cast_rays_cpp, extract_geometry_buffers
from scrollswarm.map import ScrollingMap, MapConfig

# 1. Initialize map and extract flattened geometry buffers
map_cfg = MapConfig(seed=42)
scrolling_map = ScrollingMap(map_cfg)
obs_meta, obs_edges, obs_ranges, target_data = extract_geometry_buffers(
    scrolling_map.obstacles, scrolling_map.targets
)

# 2. Define LiDAR sensor beam configuration
origin_x = 100.0   # Drone x position
origin_y = 200.0   # Drone y position
num_rays = 7
fov = np.radians(120.0)
ray_angles = np.linspace(-fov / 2.0, fov / 2.0, num_rays, dtype=np.float32)
max_range = 400.0  # Max sensor range in pixels

# 3. Call native C++ kernel directly
distances, hit_types = cast_rays_cpp(
    origin_x=origin_x,
    origin_y=origin_y,
    ray_angles=ray_angles,
    max_range=max_range,
    ceiling_y=float(scrolling_map.ceiling_y),
    floor_y=float(scrolling_map.floor_y),
    obs_meta=obs_meta,
    obs_edges=obs_edges,
    obs_ranges=obs_ranges,
    target_data=target_data,
)

print(f"Normalized Distances [0, 1]:\n{distances}")
print(f"Surface Hit Types (0=None, 1=Obstacle, 2=Boundary, 3=Target):\n{hit_types}")
```

### Point-in-Polygon & Distance Testing

```python
from scrollswarm.geometry import point_in_polygon_cpp, distance_to_polygon_cpp

# Test if drone coordinates penetrate an obstacle
test_x, test_y = 150.0, 220.0
is_inside = point_in_polygon_cpp(test_x, test_y, obs_edges, start_idx=0, edge_count=4)
print("Penetration violation:", is_inside)

# Compute minimum Euclidean distance to polygon boundary
min_dist = distance_to_polygon_cpp(test_x, test_y, obs_edges, start_idx=0, edge_count=4)
print(f"Proximity clearance: {min_dist:.2f} pixels")
```

---

## 6. Empirical Benchmarks: Micro-Latency & Step Throughput

The empirical benchmark suite ([`scripts/benchmark_backends.py`](https://github.com/HamBa-m/scrollswarm/blob/main/scripts/benchmark_backends.py)) tests both isolated kernel micro-latency and macro environment stepping throughput under standardized conditions.

To run the benchmark on your local machine:

```bash
uv run python scripts/benchmark_backends.py --steps 500 --num-agents 4 --seed 42
```

### Benchmark Results (Intel Core i7-11850H @ 2.50 GHz, 4 UAVs, Seed 42)

| Backend Engine | Isolated Raycast Latency | Raycast Speedup | Stepping Throughput | Stepping Latency | End-to-End Speedup | JIT Warmup Overhead |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NumPy Baseline** | $214.24\,\mu\text{s}$ | $1.00\times$ (ref) | $691.4\text{ steps/s}$ | $1.446\text{ ms/step}$ | $1.00\times$ (ref) | **$0.0\,\text{s}$ (None)** |
| **C++20 Kernel** | $\mathbf{6.41\,\mu\text{s}}$ | $\mathbf{33.44\times}$ | $\mathbf{1,652.3\text{ steps/s}}$ | $\mathbf{0.605\text{ ms/step}}$ | $\mathbf{2.39\times}$ | **$0.0\,\text{s}$ (None)** |
| **Numba LLVM JIT** | $\mathbf{4.28\,\mu\text{s}}$ | $\mathbf{50.08\times}$ | $\mathbf{1,740.2\text{ steps/s}}$ | $\mathbf{0.575\text{ ms/step}}$ | $\mathbf{2.52\times}$ | $\approx 1.5\,\text{s}$ |

```mermaid
xychart-beta
    title "Isolated Raycasting Latency (Microseconds, Lower is Better)"
    x-axis ["NumPy Baseline", "C++20 Native", "Numba LLVM JIT"]
    y-axis "Latency (μs)" 0 --> 250
    bar [214.24, 6.41, 4.28]
```

### Key Analytical Takeaways

1. **Micro-Kernel Acceleration:** Both C++20 and Numba eliminate the Python interpreter loop overhead during line-segment intersection checks, achieving **$33\times$ to $50\times$ faster** isolated raycasting.
2. **End-to-End Stepping Speedup:** In full PettingZoo simulation, the environment step rate increases from $\sim 690\text{ steps/s}$ to over **$1,700\text{ steps/s}$** on a single CPU thread ($2.5\times$ overall speedup). The remaining time is consumed by Python dictionary creation, PyG graph assembly, and PyTorch tensor allocations.
3. **Warmup vs. Instant Start:**
   - **Numba** delivers slightly higher peak micro-throughput thanks to LLVM loop optimization, but incurs an initial $\sim 1.5\,\text{s}$ compilation lag on first invocation.
   - **C++20** achieves comparable high throughput with **zero warmup latency**, making it ideal for distributed Ray/SubprocVec workers where processes restart frequently.

### Swarm Cardinality Scaling ($N \in [1, 10]$)

To characterize how environment throughput and computational speedup scale with fleet size, ScrollSwarm provides an automated swarm cardinality sweep:

```bash
uv run python scripts/benchmark_backends.py --sweep-agents --steps 500 --seed 42
```

![Swarm Cardinality Scaling](../assets/benchmark_agent_scaling.png)

#### Empirical Scaling Results (Intel Core i7-11850H @ 2.50 GHz, 500 Steps, Seed 42)

| Swarm Cardinality ($N$) | NumPy Baseline | Native C++20 | Numba LLVM JIT | C++20 Speedup | Numba Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 1$** | $2,186.6\text{ steps/s}$ | $5,441.5\text{ steps/s}$ | $5,790.5\text{ steps/s}$ | **$2.49\times$** | **$2.65\times$** |
| **$N = 2$** | $1,258.7\text{ steps/s}$ | $2,764.3\text{ steps/s}$ | $2,756.7\text{ steps/s}$ | **$2.20\times$** | **$2.19\times$** |
| **$N = 3$** | $829.0\text{ steps/s}$ | $2,153.8\text{ steps/s}$ | $2,115.5\text{ steps/s}$ | **$2.60\times$** | **$2.55\times$** |
| **$N = 4$** | $553.1\text{ steps/s}$ | $1,641.2\text{ steps/s}$ | $1,645.3\text{ steps/s}$ | **$2.97\times$** | **$2.97\times$** |
| **$N = 5$** | $558.1\text{ steps/s}$ | $1,314.4\text{ steps/s}$ | $1,381.4\text{ steps/s}$ | **$2.35\times$** | **$2.47\times$** |
| **$N = 6$** | $392.4\text{ steps/s}$ | $1,087.1\text{ steps/s}$ | $1,137.3\text{ steps/s}$ | **$2.77\times$** | **$2.90\times$** |
| **$N = 7$** | $387.8\text{ steps/s}$ | $763.8\text{ steps/s}$ | $624.6\text{ steps/s}$ | **$1.97\times$** | **$1.61\times$** |
| **$N = 8$** | $332.6\text{ steps/s}$ | $588.1\text{ steps/s}$ | $689.1\text{ steps/s}$ | **$1.77\times$** | **$2.07\times$** |
| **$N = 9$** | $271.4\text{ steps/s}$ | $559.7\text{ steps/s}$ | $585.1\text{ steps/s}$ | **$2.06\times$** | **$2.16\times$** |
| **$N = 10$** | $266.6\text{ steps/s}$ | $581.1\text{ steps/s}$ | $605.3\text{ steps/s}$ | **$2.18\times$** | **$2.27\times$** |

#### Analytical Observations

1. **Sustained $2.0\times\text{--}3.0\times$ Real-World Speedup:** Across all swarm cardinalities $N \in [1, 10]$, the compiled C++20 and Numba kernels consistently deliver **$2.0\times$ to $3.0\times$ faster environment execution** than pure NumPy.
2. **Single-Agent Peak Rate:** For $N=1$, environment execution exceeds **$5,400\text{--}5,700\text{ steps/second}$** ($<0.19\text{ ms/step}$), making single-drone policy debugging near instantaneous.
3. **Decoupled Scaling with Vectorized Workers:** Because a single corridor realistically bounds swarm density to $N \le 10$, large-scale distributed MARL architectures scale linearly across CPU cores by instantiating $\mathcal{E}$ vectorized environment workers via PettingZoo / SubprocVec, maintaining high per-worker throughput without artificial crowding.

---

## 7. Numerical Parity & Float32 Precision Analysis

A critical requirement in scientific RL research is **numerical reproducibility**: swapping acceleration backends must not alter the mathematical semantics of the environment or skew observation distributions.

ScrollSwarm includes an automated cross-backend verification suite in [`tests/test_geometry.py`](https://github.com/HamBa-m/scrollswarm/blob/main/tests/test_geometry.py) that asserts mathematical parity across all backends:

```bash
uv run pytest tests/test_geometry.py -v
```

### FastMath Reassociation & Machine Epsilon

Due to compiler fast-math optimizations (`-ffast-math` in C++ and `fastmath=True` in Numba), floating-point arithmetic reassociates operations:

$$(a + b) + c \neq a + (b + c) \quad \text{(in IEEE 754 float32)}$$

When calculating ray intersections near acute polygon vertices at grazing angles ($400\text{ px}$ ray distance), trigonometric reassociation induces a negligible variance:

$$\max |\mathbf{d}_{\text{cpp}} - \mathbf{d}_{\text{numpy}}| \le 1.37 \times 10^{-4}\text{ pixels}$$

Because $10^{-4}\text{ px}$ is three orders of magnitude smaller than individual pixel coordinates, the test suite asserts strict parity at $\text{atol} = 10^{-3}$:

```python
np.testing.assert_allclose(distances_cpp, distances_numpy, atol=1e-3, rtol=1e-5)
np.testing.assert_array_equal(hits_cpp, hits_numpy)
```

All backends produce identical discrete hit classifications (0=None, 1=Obstacle, 2=Boundary, 3=Target) and indistinguishable continuous normalized distances.

---

## 8. Summary & Best Practices

| Deployment Scenario | Recommended Backend | Rationale |
| :--- | :--- | :--- |
| **Local Prototyping & CI Pipelines** | `backend="auto"` (or `"numpy"`) | Runs immediately out of the box on any machine with zero build setup. |
| **High-Throughput MARL Training** | `backend="cpp"` | Maximum sustained throughput with zero JIT compilation pause when launching multi-process workers. |
| **Environments without C++ Toolchains** | `backend="numba"` | Highest per-core execution rate via pure `uv add numba`. |

### Next Steps

- Explore [Tutorial 01: PettingZoo ParallelEnv Loop](01_parallel_env.md) to integrate accelerated environments with custom multi-agent loops.
- Review [Tutorial 03: PyG & GNN Integration](03_gnn_pyg.md) to process dynamic relational swarm graphs alongside accelerated observations.
- Consult the [Geometry API Reference](../api/geometry.md) for full function signatures and buffer structures.
