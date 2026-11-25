"""
Geometry Acceleration and Multi-Backend Dispatch Subsystem for ScrollSwarm.

Decouples 2D raycasting and collision detection from high-level environment
logic, allowing runtime selection between:
- ``numpy``: Pure Python / NumPy baseline (zero compiler/JIT dependencies).
- ``numba``: LLVM machine-code JIT acceleration (CPU @njit with fastmath/nogil).
- ``cpp``: High-performance pre-compiled C++20 kernel via nanobind (future).
- ``auto``: Automatically selects the fastest available backend on the host system.
"""

import importlib
import logging
import os
from pathlib import Path
import sys
from typing import Any, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from scrollswarm.geometry.numpy_backend import (
    cast_rays_numpy,
    point_in_polygon_numpy,
    distance_to_polygon_numpy,
)

logger = logging.getLogger(__name__)

# Optional Numba backend
try:
    import numba
    from scrollswarm.geometry.numba_backend import (
        cast_rays_numba,
        point_in_polygon_numba,
        distance_to_polygon_numba,
        HAS_NUMBA,
    )
except ImportError as exc:
    logger.debug("Numba geometry backend unavailable: %s", exc)
    HAS_NUMBA = False
    cast_rays_numba = None
    point_in_polygon_numba = None
    distance_to_polygon_numba = None


def _has_compiled_cpp_artifact(geometry_dir: Optional[Path] = None) -> bool:
    """Returns True if a compiled C++ geometry extension binary exists on disk."""
    target_dir = geometry_dir if geometry_dir is not None else Path(__file__).resolve().parent
    return any(target_dir.glob("_geometry_cpp*.pyd")) or any(target_dir.glob("_geometry_cpp*.so"))


def _load_cpp_extension(geometry_dir: Optional[Path] = None) -> Optional[Any]:
    """Attempts to load the compiled C++20 geometry extension with failure diagnostics."""
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        base_prefix = getattr(sys, "base_prefix", sys.prefix)
        if os.path.exists(base_prefix):
            try:
                os.add_dll_directory(base_prefix)
            except OSError:
                pass

    try:
        try:
            return importlib.import_module("scrollswarm.geometry._geometry_cpp")
        except ImportError:
            return importlib.import_module("_geometry_cpp")
    except (ImportError, OSError, AttributeError) as exc:
        if _has_compiled_cpp_artifact(geometry_dir):
            logger.warning(
                "Compiled C++20 geometry extension exists but failed to load (%s); falling back.",
                exc,
            )
        else:
            logger.debug("Compiled C++20 geometry extension not built (%s).", exc)
        return None


_geometry_cpp = _load_cpp_extension()
if _geometry_cpp is not None:
    cast_rays_cpp = _geometry_cpp.cast_rays_cpp
    point_in_polygon_cpp = _geometry_cpp.point_in_polygon_cpp
    distance_to_polygon_cpp = _geometry_cpp.distance_to_polygon_cpp
    HAS_CPP = True
else:
    HAS_CPP = False
    cast_rays_cpp = None
    point_in_polygon_cpp = None
    distance_to_polygon_cpp = None


def is_numba_available() -> bool:
    """Returns True if Numba JIT acceleration is installed and functional."""
    return bool(HAS_NUMBA and cast_rays_numba is not None)


def is_cpp_available() -> bool:
    """Returns True if the native C++20 geometry extension is compiled and loaded."""
    return bool(HAS_CPP and _geometry_cpp is not None)


def get_available_backends() -> List[str]:
    """Returns list of all geometry backends operational on the host system."""
    backends = ["numpy"]
    if is_numba_available():
        backends.append("numba")
    if is_cpp_available():
        backends.append("cpp")
    return backends


def resolve_backend(requested: str = "auto") -> str:
    """
    Resolves backend name against host system capabilities.

    Priority order for 'auto':
    1. 'cpp' (compiled native C++20 kernel)
    2. 'numba' (LLVM JIT CPU acceleration)
    3. 'numpy' (guaranteed portable fallback)
    """
    req = requested.lower().strip()
    if req == "auto":
        if is_cpp_available():
            return "cpp"
        if is_numba_available():
            return "numba"
        return "numpy"

    if req == "numba":
        if not is_numba_available():
            raise RuntimeError(
                "Numba backend requested, but 'numba' is not installed or import failed."
            )
        return "numba"

    if req == "cpp":
        if not is_cpp_available():
            raise RuntimeError(
                "C++ geometry backend requested, but 'scrollswarm._geometry_cpp' is not compiled."
            )
        return "cpp"

    if req == "numpy":
        return "numpy"

    raise ValueError(
        f"Unknown geometry backend: '{requested}'. Valid options: ['auto', 'numpy', 'numba', 'cpp']."
    )


def extract_geometry_buffers(
    obstacles: List[object],
    targets: List[object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts contiguous flat NumPy arrays from scrolling map entities
    for zero-overhead vectorized / JIT geometry kernels.

    Returns:
        obs_meta:        (N_obs, 4) float32 - [x, y, radius, max_dim]
        obs_edges:       (N_edges, 4) float32 - [ax, ay, bx, by]
        obs_edge_ranges: (N_obs, 2) int32 - [start_idx, end_idx]
        target_data:     (N_tgts, 3) float32 - [x, y, radius]
    """
    n_obs = len(obstacles)
    obs_meta = np.zeros((n_obs, 4), dtype=np.float32)
    obs_edge_ranges = np.zeros((n_obs, 2), dtype=np.int32)

    all_edges: List[Tuple[float, float, float, float]] = []

    for i, obs in enumerate(obstacles):
        ox = float(getattr(obs, "x", 0.0))
        oy = float(getattr(obs, "y", 0.0))
        radius = float(getattr(obs, "radius", 20.0))
        max_dim = float(max(
            radius,
            getattr(obs, "width", 40.0),
            getattr(obs, "height", 60.0),
        ))
        obs_meta[i] = [ox, oy, radius, max_dim]

        start_e = len(all_edges)
        edges = getattr(obs, "edges", [])
        for (ax, ay), (bx, by) in edges:
            all_edges.append((float(ax), float(ay), float(bx), float(by)))
        end_e = len(all_edges)
        obs_edge_ranges[i] = [start_e, end_e]

    if all_edges:
        obs_edges = np.array(all_edges, dtype=np.float32)
    else:
        obs_edges = np.zeros((0, 4), dtype=np.float32)

    # Active targets
    active_tgts = [t for t in targets if not getattr(t, "collected", False)]
    n_tgts = len(active_tgts)
    target_data = np.zeros((n_tgts, 3), dtype=np.float32)
    for j, tgt in enumerate(active_tgts):
        tx = float(getattr(tgt, "x", 0.0))
        ty = float(getattr(tgt, "y", 0.0))
        tr = float(getattr(tgt, "size", 16.0))
        target_data[j] = [tx, ty, tr]

    return obs_meta, obs_edges, obs_edge_ranges, target_data
