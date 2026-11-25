# Repository Agent Guidelines (`AGENTS.md`)

This file provides repository-specific development context, environment setup, invariants, and verification rules for AI coding agents working on **ScrollSwarm**.

---

## 1. Project Overview & Architecture

ScrollSwarm is a 2D cooperative Multi-Agent Reinforcement Learning (MARL) testbed for UAV swarm reconnaissance adhering to Farama Foundation's **PettingZoo `ParallelEnv`** standard, **CTDE** (`state()`), and **PyTorch Geometric** (`edge_index`, `edge_attr`).

### Core Package Layout (`scrollswarm/`)
- [`scrollswarm/env.py`](scrollswarm/env.py): `ScrollSwarmEnv` (`ParallelEnv`), synchronized stepping, boundary/obstacle collision culling, Bernoulli target photography, `team_reward` sharing, and Pygame rendering.
- [`scrollswarm/agent.py`](scrollswarm/agent.py): `Agent`, `Action` (`0: UP`, `1: STAY`, `2: DOWN`), and `create_agents()` (`1 <= num_agents <= 10`).
- [`scrollswarm/map.py`](scrollswarm/map.py): `ScrollingMap` & `MapConfig`, procedural terrain corridor generation enforcing `min_corridor_gap`.
- [`scrollswarm/sensors.py`](scrollswarm/sensors.py): `RaySensor` constructing the 16D observation vector (`self_y`, 7 LiDAR rays, 4 target features, 4 nearest living neighbor features).
- [`scrollswarm/comm.py`](scrollswarm/comm.py): `CommunicationGraph` & `TopologyType` (`chain`, `ring`, `full`, `radius`, `capped`), dead-agent pruning, PyG COO export, and `route_messages()`.
- [`scrollswarm/vector.py`](scrollswarm/vector.py): `BaseVectorScrollSwarm`, `SyncVectorScrollSwarm`, and multiprocess `AsyncVectorScrollSwarm` (`spawn` context) preserving `(E, N, ...)` CTDE + graph structure and block-diagonal `batch_graphs()`.
- [`scrollswarm/geometry/`](scrollswarm/geometry/): Pluggable computational geometry engines (`numpy_backend.py`, `numba_backend.py`, `cpp/geometry.hpp`, `cpp/bindings.cpp`) selected via `backend="auto" | "cpp" | "numba" | "numpy"`.
- [`scrollswarm/sprites.py`](scrollswarm/sprites.py): Procedural Pygame quadcopter rendering (`_create_procedural_quadcopter_surface`, cached). Do **not** reintroduce unused `.svg` assets.

---

## 2. Development Setup (`uv`)

ScrollSwarm uses `uv` with Python 3.12+.

```bash
# Install all runtime, docs, acceleration, benchmark, and dev dependencies
uv sync --extra all

# Optional: rebuild the native C++20 geometry extension locally
uv run python scripts/build_geometry_cpp.py
```

---

## 3. Coding Standards & Invariants

1. **Formatting (`black`):** Strictly enforced in CI at `line-length = 88`, `target-version = ["py312"]` (configured in `pyproject.toml`). Run `uv run black .` after Python edits.
2. **Docstrings & Typing:** Use type annotations and Google-style docstrings across public classes/functions (`mkdocstrings` renders API reference directly from docstrings).
3. **Cross-Backend Numerical Parity:** Any change to raycasting, polygon containment, or distance math in `numpy_backend.py`, `numba_backend.py`, or `cpp/geometry.hpp` **must** maintain parity across all three backends (`atol=1e-3` for `float32` distances, exact equality for hit types and trajectories).
4. **Zero Hot-Path Logging Overhead:** `scrollswarm` uses `logging.getLogger("scrollswarm")` with `NullHandler()`. Never emit `INFO`/`DEBUG` logs or `print()` statements inside `step()`, `observe()`, or `cast_rays_*()`.
5. **Windows Multiprocessing Safety:** Workers in `scrollswarm/vector.py` must remain top-level pickleable functions compatible with `mp.get_context("spawn")`.
6. **C++ Build Portability:** Keep static linkage flags (`-static-libgcc -static-libstdc++ -static`) on Windows/MinGW in `scripts/build_geometry_cpp.py` and `os.add_dll_directory` support in `scrollswarm/geometry/__init__.py`.
7. **Swarm Cardinality Bound:** Single-corridor `num_agents` is intentionally bounded to `[1, 10]`; scale total agent count across parallel corridors using `SyncVectorScrollSwarm` / `AsyncVectorScrollSwarm`.
8. **White-Box Documentation Sync:** Keep `README.md`, `docs/index.md`, and `docs/getting-started/installation.md` test counts, benchmark specs, and code snippets synchronized with actual passing implementations.

---

## 4. Mandatory Verification Before Committing

Run all three checks locally (matching `.github/workflows/ci.yml` and `.github/workflows/docs.yml`) before committing or opening a PR:

```bash
# 1. Verify Black formatting
uv run black --check .

# 2. Run full unit, PettingZoo compliance, backend parity, and vector tests
uv run pytest

# 3. Verify documentation builds strictly with zero warnings/errors
uv run mkdocs build --strict
```
