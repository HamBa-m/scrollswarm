# Changelog

All notable changes to **ScrollSwarm** are documented in this file.

## [0.2.0] - 2026-11-25

### Added
- **Multi-Backend Computational Geometry Acceleration (`scrollswarm.geometry`):**
  - Native C++20 geometry kernel (`scrollswarm/geometry/cpp/geometry.hpp`, `bindings.cpp`) built via `pybind11` (`scripts/build_geometry_cpp.py`) achieving **$33.44\times$** isolated raycasting speedup ($6.41\,\mu\text{s}$ vs. $214.24\,\mu\text{s}$) with **$0.0\,\text{s}$** JIT warmup.
  - Numba LLVM JIT backend (`scrollswarm/geometry/numba_backend.py`) with `@njit(fastmath=True, nogil=True, cache=True)` achieving **$50.08\times$** isolated raycasting speedup ($4.28\,\mu\text{s}$).
  - Pluggable `backend` dispatch parameter (`"auto"`, `"cpp"`, `"numba"`, `"numpy"`) on `ScrollSwarmEnv`, `Agent`, and `RaySensor`.
- **CTDE & Graph-Aware Vectorized Runners (`scrollswarm.vector`):**
  - Added `BaseVectorScrollSwarm`, `SyncVectorScrollSwarm`, and multiprocess `AsyncVectorScrollSwarm` (`spawn` context) preserving `(E, N, ...)` swarm structure, CTDE global state `(E, 16 * N)`, `alive_mask`, auto-reset `final_observation` / `final_state`, and `batch_graphs()` block-diagonal PyG COO conversion.
- **Empirical Benchmarking Suite:**
  - `scripts/benchmark_backends.py` with `--sweep-agents` scaling across $N \in [1, 10]$ UAVs (`docs/assets/benchmark_agent_scaling.png`).
  - `scripts/benchmark_vector.py` pure-environment vector scaling sweep across $\mathcal{E} \in \{1, 2, 4, 8, 16, 32, 64\}$ corridors (`docs/assets/benchmark_vector_scaling.png`, reaching **4,301.3 env-steps/s** / **17,205.1 agent-steps/s** at $\mathcal{E} = 64$).
- **Diagnostic Stdlib Logging:**
  - Added `logging.getLogger("scrollswarm")` with `NullHandler`, C++ extension load-failure `WARNING` diagnostics, and `INFO`/`DEBUG` lifecycle logs with zero `step()` hot-path overhead.
- **Documentation & Tutorials:**
  - Added **Tutorial 04: Computational Geometry Acceleration** (`docs/tutorials/04_acceleration_backends.md`).
  - Added **Tutorial 05: Vectorized MARL & Multi-Core Scaling** (`docs/tutorials/05_vectorized_marl.md`).
  - Added API reference pages for **Geometry Backends** (`docs/api/geometry.md`) and **Vectorized Runners** (`docs/api/vector.md`).
  - Redesigned the `Core Architecture` Mermaid diagram in `docs/index.md` into a compact vertical pipeline.
- **Tooling, Formatting & CI:**
  - Adopted `black` formatting (`line-length = 88`, Python 3.12+) in `pyproject.toml` and `CONTRIBUTING.md`.
  - Added GitHub Actions CI workflow (`.github/workflows/ci.yml`) running `black --check .` and `pytest`.
  - Added `dev` and `benchmark` optional extras in `pyproject.toml`.

### Changed & Fixed
- Expanded unit test suite from 52 to **69 tests** (`tests/test_geometry.py`, `tests/test_env.py`, `tests/test_sensors.py`, `tests/test_comm.py`, `tests/test_logging.py`, `tests/test_vector.py`), covering polygon parity, 60-step multi-backend rollouts, `team_reward=True` splitting, and crash culling.
- Removed unused `.svg` sprite assets (`assets/*.svg`, `scrollswarm/assets/*.svg`) and empty `scrollswarm/assets/` directory (`sprites.py` renders quadcopters procedurally).

---

## [0.1.0] - 2026-11-23

### Added
- Initial release of **ScrollSwarm**: 2D multi-agent reinforcement learning (MARL) testbed for cooperative UAV swarm reconnaissance implementing Farama PettingZoo's `ParallelEnv`.
- Procedural scrolling aerial corridor (`ScrollingMap`, `MapConfig`) with guaranteed navigable gap clearance, static/dynamic hazards, and stochastic Bernoulli target inspection.
- 16-dimensional agent observation (`RaySensor` 7-ray forward LiDAR, target bearing, neighbor proximity), CTDE centralized `state()`, and PyTorch Geometric (`edge_index`, `edge_attr`) graph topologies (`chain`, `ring`, `full`, `radius`, `capped`).
- Interactive Pygame simulator (`main.py`), heuristic reactive baseline (`scripts/run_heuristic_policy.py`), MkDocs Material site (`docs/`), Tutorials 01–03, and PEP 621 packaging metadata.
