# Geometry & Computational Acceleration

ScrollSwarm decouples its 2D analytical geometry calculations from high-level PettingZoo environment logic via a modular hardware abstraction layer.

## Architectural Overview

Raycasting and obstacle collision checks account for approximately **$79.3\%$** of execution time in raw Python. The `scrollswarm.geometry` subsystem provides pluggable computational backends:

- **`numpy` (Baseline):** Pure Python / NumPy analytical geometry with zero build dependencies. Always available across all platforms.
- **`numba` (LLVM JIT):** JIT-compiled native CPU machine code (`@njit(fastmath=True, nogil=True)`), offering $5\times\text{--}15\times$ speedups on geometry routines without requiring a C++ compiler.
- **`cpp` (Native Extension, Future):** High-throughput C++20 kernel built with `nanobind` and OpenMP.
- **`auto` (Default):** Automatically probes the host environment and selects the fastest available backend.

```python
from scrollswarm import ScrollSwarmEnv

# Explicitly choose acceleration backend
env = ScrollSwarmEnv(backend="auto")  # or "numpy", "numba", "cpp"
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
