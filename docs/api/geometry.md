# Geometry & Computational Acceleration

ScrollSwarm decouples its 2D analytical geometry calculations from high-level PettingZoo environment logic via a modular hardware abstraction layer.

## Architectural Overview

Raycasting and obstacle collision checks account for approximately **$79.3\%$** of execution time in raw Python. The `scrollswarm.geometry` subsystem provides pluggable computational backends:

- **`numpy` (Baseline):** Pure Python / NumPy analytical geometry with zero build dependencies. Always available across all platforms.
- **`numba` (LLVM JIT):** JIT-compiled native CPU machine code (`@njit(fastmath=True, nogil=True)`), offering $50\times$ speedup on isolated raycasting without requiring a C++ compiler.
- **`cpp` (Native C++20 Extension):** Pre-compiled native C++20 geometry kernel built via pybind11 with zero JIT warmup latency ($33\times$ isolated raycasting speedup).
- **`auto` (Default):** Automatically probes the host environment and selects the fastest available backend (priority: `cpp` $\to$ `numba` $\to$ `numpy`).

### Building the C++ Kernel Locally
To compile the high-performance C++20 geometry kernel on your machine:
```bash
uv run python scripts/build_geometry_cpp.py
```
This generates `scrollswarm/geometry/_geometry_cpp.pyd` (or `.so` on Linux/macOS) with static runtime linkage.

```python
from scrollswarm import ScrollSwarmEnv

# Explicitly choose acceleration backend
env = ScrollSwarmEnv(backend="auto")  # or "cpp", "numba", "numpy"
```

---

## API Reference

::: scrollswarm.geometry
    options:
      members:
        - get_available_backends
        - resolve_backend
        - is_numba_available
        - is_cpp_available
        - extract_geometry_buffers
