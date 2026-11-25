"""
Unit tests for ScrollSwarm geometry acceleration backends.

Verifies:
1. Backend discovery and resolution mechanics (auto, numpy, numba, cpp).
2. Buffer extraction from dynamic scrolling map entities.
3. Mathematical precision and numerical parity across backends.
4. Raycasting against horizontal boundaries, arbitrary polygons, and targets.
"""

import numpy as np
import pytest

from scrollswarm.geometry import (
    get_available_backends,
    resolve_backend,
    is_numba_available,
    is_cpp_available,
    extract_geometry_buffers,
)
from scrollswarm.geometry.numpy_backend import (
    cast_rays_numpy,
    point_in_polygon_numpy,
    distance_to_polygon_numpy,
)
from scrollswarm.items import Obstacle, ObstacleType, Target
from scrollswarm.map import ScrollingMap, MapConfig
from scrollswarm.env import ScrollSwarmEnv


def test_backend_discovery():
    """Verifies that numpy backend is always present and auto-resolution works."""
    backends = get_available_backends()
    assert "numpy" in backends
    assert resolve_backend("numpy") == "numpy"
    assert resolve_backend("  NUMPY ") == "numpy"

    resolved_auto = resolve_backend("auto")
    assert resolved_auto in ["numpy", "numba", "cpp"]
    assert resolve_backend(" AUTO ") == resolved_auto

    with pytest.raises(ValueError):
        resolve_backend("nonexistent_backend")


def _polygon_to_edges(verts: np.ndarray) -> np.ndarray:
    """Converts an ordered (V, 2) vertex array into a closed (V, 4) edge array."""
    n = len(verts)
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    nxt = np.roll(verts, -1, axis=0)
    return np.hstack([verts, nxt]).astype(np.float32)


def _assert_polygon_backend_parity(pinp_fn, dist_fn) -> None:
    """Asserts point-in-polygon and distance parity against NumPy on convex and concave shapes."""
    # Concave L-shaped polygon
    concave_verts = np.array(
        [
            [100.0, 100.0],
            [180.0, 100.0],
            [180.0, 130.0],
            [130.0, 130.0],
            [130.0, 180.0],
            [100.0, 180.0],
        ],
        dtype=np.float32,
    )
    concave_edges = _polygon_to_edges(concave_verts)

    probe_points = [
        (115.0, 115.0),  # Inside main body
        (115.0, 160.0),  # Inside vertical arm
        (160.0, 160.0),  # Outside inside the concave notch
        (90.0, 140.0),  # Outside near left edge
        (200.0, 115.0),  # Outside near right edge
    ]
    for px, py in probe_points:
        assert bool(pinp_fn(px, py, concave_verts)) is point_in_polygon_numpy(
            px, py, concave_verts
        )
        np_dist = distance_to_polygon_numpy(px, py, concave_verts, concave_edges)
        assert dist_fn(px, py, concave_verts, concave_edges) == pytest.approx(
            np_dist, abs=1e-3
        )

    # Empty polygon containment edge case
    empty_verts = np.zeros((0, 2), dtype=np.float32)
    assert bool(pinp_fn(50.0, 50.0, empty_verts)) is False


def test_buffer_extraction():
    """Verifies flat numeric array extraction from obstacles and targets."""
    obs1 = Obstacle(
        x=100.0, y=200.0, width=40.0, height=60.0, obstacle_type=ObstacleType.STRUCTURE
    )
    obs2 = Obstacle(x=250.0, y=150.0, radius=25.0, obstacle_type=ObstacleType.FLOATING)
    tgt1 = Target(x=300.0, y=180.0, size=16.0)
    tgt2 = Target(x=400.0, y=220.0, size=16.0)
    tgt2.collected = True  # should be excluded

    meta, edges, ranges, targets = extract_geometry_buffers([obs1, obs2], [tgt1, tgt2])

    assert meta.shape == (2, 4)
    assert meta[0, 0] == 100.0
    assert meta[0, 1] == 200.0
    assert ranges.shape == (2, 2)
    assert edges.shape[0] > 0
    assert edges.shape[1] == 4
    # Only 1 uncollected target
    assert targets.shape == (1, 3)
    assert targets[0, 0] == 300.0


def test_point_in_polygon_and_distance():
    """Tests polygon containment and Euclidean distance routines."""
    # A 40x40 square centered at (100, 100)
    hw = 20.0
    verts = np.array(
        [
            [100 - hw, 100 - hw],
            [100 + hw, 100 - hw],
            [100 + hw, 100 + hw],
            [100 - hw, 100 + hw],
        ],
        dtype=np.float32,
    )

    edges = np.array(
        [
            [verts[0, 0], verts[0, 1], verts[1, 0], verts[1, 1]],
            [verts[1, 0], verts[1, 1], verts[2, 0], verts[2, 1]],
            [verts[2, 0], verts[2, 1], verts[3, 0], verts[3, 1]],
            [verts[3, 0], verts[3, 1], verts[0, 0], verts[0, 1]],
        ],
        dtype=np.float32,
    )

    # Inside point
    assert point_in_polygon_numpy(100.0, 100.0, verts) is True
    assert distance_to_polygon_numpy(100.0, 100.0, verts, edges) == 0.0

    # Outside point: 10 pixels to the right
    assert point_in_polygon_numpy(130.0, 100.0, verts) is False
    dist = distance_to_polygon_numpy(130.0, 100.0, verts, edges)
    assert pytest.approx(dist, abs=1e-4) == 10.0


def test_cast_rays_boundary_and_obstacle():
    """Tests analytical raycasting against ceiling, floor, and obstacle polygon."""
    angles = np.array([-np.pi / 4, 0.0, np.pi / 4], dtype=np.float32)
    origin_x = 100.0
    origin_y = 200.0
    ceil_y = 50.0
    floor_y = 350.0
    max_range = 400.0

    # Put a target directly ahead at x=200, y=200 (distance = 100)
    target_data = np.array([[200.0, 200.0, 10.0]], dtype=np.float32)

    # Empty obstacles
    obs_meta = np.zeros((0, 4), dtype=np.float32)
    obs_edges = np.zeros((0, 4), dtype=np.float32)
    obs_ranges = np.zeros((0, 2), dtype=np.int32)

    obs_d, obs_t, obs_e, tgt_d, tgt_t, tgt_e = cast_rays_numpy(
        origin_x,
        origin_y,
        angles,
        max_range,
        ceil_y,
        floor_y,
        obs_meta,
        obs_edges,
        obs_ranges,
        target_data,
    )

    # Ray 0: angle -45 deg (points up-right). Should hit ceiling y=50
    # dy = 50 - 200 = -150. sin(-45) = -sqrt(2)/2. distance = 150 * sqrt(2) approx 212.13
    expected_ceil_dist = 150.0 * np.sqrt(2.0)
    assert pytest.approx(obs_d[0], abs=0.5) == expected_ceil_dist
    assert obs_t[0] == 1  # boundary

    # Ray 1: angle 0 deg (points straight right). No obstacle, should hit target at x=200 - 10 = 190
    assert obs_d[1] == max_range
    assert obs_t[1] == 0
    assert pytest.approx(tgt_d[1], abs=0.5) == 90.0  # 100 - radius 10 = 90
    assert tgt_t[1] == 1


def test_numba_parity_when_available():
    """
    If Numba is installed, verifies strict numerical parity between
    NumPy and Numba implementations down to 1e-4 tolerance.
    """
    if not is_numba_available():
        pytest.skip("Numba is not installed; skipping parity test.")

    from scrollswarm.geometry.numba_backend import (
        cast_rays_numba,
        point_in_polygon_numba,
        distance_to_polygon_numba,
    )

    rng = np.random.default_rng(42)
    angles = np.linspace(-np.pi / 3, np.pi / 3, 7, dtype=np.float32)
    max_range = 400.0
    ceil_y = 15.0
    floor_y = 385.0

    # Create random corridor scene
    cfg = MapConfig(seed=42)
    scrolling_map = ScrollingMap(cfg)
    obs_meta, obs_edges, obs_ranges, target_data = extract_geometry_buffers(
        scrolling_map.obstacles, scrolling_map.targets
    )

    origin_x = 120.0
    origin_y = 210.0

    # Run NumPy
    d_np, t_np, e_np, td_np, tt_np, te_np = cast_rays_numpy(
        origin_x,
        origin_y,
        angles,
        max_range,
        ceil_y,
        floor_y,
        obs_meta,
        obs_edges,
        obs_ranges,
        target_data,
    )

    # Run Numba
    d_nb, t_nb, e_nb, td_nb, tt_nb, te_nb = cast_rays_numba(
        origin_x,
        origin_y,
        angles,
        max_range,
        ceil_y,
        floor_y,
        obs_meta,
        obs_edges,
        obs_ranges,
        target_data,
    )

    np.testing.assert_allclose(
        d_np, d_nb, atol=1e-3, err_msg="Obstacle ray distance parity failed"
    )
    np.testing.assert_array_equal(
        t_np, t_nb, err_msg="Obstacle ray hit type parity failed"
    )
    np.testing.assert_allclose(
        e_np, e_nb, atol=1e-3, err_msg="Obstacle endpoint parity failed"
    )

    np.testing.assert_allclose(
        td_np, td_nb, atol=1e-3, err_msg="Target ray distance parity failed"
    )
    np.testing.assert_array_equal(
        tt_np, tt_nb, err_msg="Target ray hit type parity failed"
    )

    _assert_polygon_backend_parity(point_in_polygon_numba, distance_to_polygon_numba)


def test_cpp_parity_when_available():
    """
    If C++20 geometry kernel is compiled, verifies strict numerical parity
    between NumPy, Numba, and C++ implementations down to 1e-3 tolerance.
    """
    if not is_cpp_available():
        pytest.skip("C++ kernel is not compiled; skipping parity test.")

    from scrollswarm.geometry import (
        cast_rays_cpp,
        point_in_polygon_cpp,
        distance_to_polygon_cpp,
    )

    angles = np.linspace(-np.pi / 3, np.pi / 3, 7, dtype=np.float32)
    max_range = 400.0
    ceil_y = 15.0
    floor_y = 385.0

    cfg = MapConfig(seed=42)
    scrolling_map = ScrollingMap(cfg)
    obs_meta, obs_edges, obs_ranges, target_data = extract_geometry_buffers(
        scrolling_map.obstacles, scrolling_map.targets
    )

    origin_x = 120.0
    origin_y = 210.0

    # Run NumPy
    d_np, t_np, e_np, td_np, tt_np, te_np = cast_rays_numpy(
        origin_x,
        origin_y,
        angles,
        max_range,
        ceil_y,
        floor_y,
        obs_meta,
        obs_edges,
        obs_ranges,
        target_data,
    )

    # Run C++
    d_cpp, t_cpp, e_cpp, td_cpp, tt_cpp, te_cpp = cast_rays_cpp(
        origin_x,
        origin_y,
        angles,
        max_range,
        ceil_y,
        floor_y,
        obs_meta,
        obs_edges,
        obs_ranges,
        target_data,
    )

    np.testing.assert_allclose(
        d_np, d_cpp, atol=1e-3, err_msg="C++ obstacle ray distance parity failed"
    )
    np.testing.assert_array_equal(
        t_np, t_cpp, err_msg="C++ obstacle ray hit type parity failed"
    )
    np.testing.assert_allclose(
        e_np, e_cpp, atol=1e-3, err_msg="C++ obstacle endpoint parity failed"
    )

    np.testing.assert_allclose(
        td_np, td_cpp, atol=1e-3, err_msg="C++ target ray distance parity failed"
    )
    np.testing.assert_array_equal(
        tt_np, tt_cpp, err_msg="C++ target ray hit type parity failed"
    )

    _assert_polygon_backend_parity(point_in_polygon_cpp, distance_to_polygon_cpp)


def test_cast_rays_empty_scene_parity():
    """Verifies raycasting parity across all available backends when obstacles and targets are empty."""
    angles = np.linspace(-np.pi / 3, np.pi / 3, 7, dtype=np.float32)
    obs_meta = np.zeros((0, 4), dtype=np.float32)
    obs_edges = np.zeros((0, 4), dtype=np.float32)
    obs_ranges = np.zeros((0, 2), dtype=np.int32)
    target_data = np.zeros((0, 3), dtype=np.float32)

    ref = cast_rays_numpy(
        100.0,
        200.0,
        angles,
        400.0,
        15.0,
        385.0,
        obs_meta,
        obs_edges,
        obs_ranges,
        target_data,
    )

    backend_fns = []
    if is_numba_available():
        from scrollswarm.geometry.numba_backend import cast_rays_numba

        backend_fns.append(("numba", cast_rays_numba))
    if is_cpp_available():
        from scrollswarm.geometry import cast_rays_cpp

        backend_fns.append(("cpp", cast_rays_cpp))

    for name, fn in backend_fns:
        res = fn(
            100.0,
            200.0,
            angles,
            400.0,
            15.0,
            385.0,
            obs_meta,
            obs_edges,
            obs_ranges,
            target_data,
        )
        np.testing.assert_allclose(
            ref[0], res[0], atol=1e-3, err_msg=f"{name} empty obs dist mismatch"
        )
        np.testing.assert_array_equal(
            ref[1], res[1], err_msg=f"{name} empty obs type mismatch"
        )
        np.testing.assert_allclose(
            ref[3], res[3], atol=1e-3, err_msg=f"{name} empty tgt dist mismatch"
        )
        np.testing.assert_array_equal(
            ref[4], res[4], err_msg=f"{name} empty tgt type mismatch"
        )


def test_env_backend_configuration():
    """Verifies ScrollSwarmEnv accepts backend argument and steps correctly."""
    for b in get_available_backends() + ["auto"]:
        env = ScrollSwarmEnv(num_agents=3, max_cycles=10, backend=b)
        assert env.backend == b
        obs, info = env.reset(seed=123)
        assert len(obs) == 3
        for a in env.agents:
            assert obs[a].shape == (16,)

        actions = {a: 0 for a in env.agents}
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        assert len(next_obs) == 3
        env.close()


def test_env_multi_step_backend_parity():
    """Verifies 60-step seeded environment rollout parity across all operational backends."""
    backends = get_available_backends()
    if len(backends) < 2:
        pytest.skip(
            "Only one backend available; skipping cross-backend rollout parity."
        )

    def _collect_rollout(backend_name: str):
        env = ScrollSwarmEnv(num_agents=4, max_cycles=100, backend=backend_name)
        obs, _ = env.reset(seed=42)
        snapshots = []
        for step_idx in range(60):
            if not env.agents:
                break
            actions = {a: (step_idx + i) % 3 for i, a in enumerate(env.agents)}
            obs, rewards, terms, truncs, _ = env.step(actions)
            snapshots.append((list(env.agents), obs, rewards, terms, truncs))
        env.close()
        return snapshots

    baseline = _collect_rollout("numpy")
    for b in backends:
        if b == "numpy":
            continue
        candidate = _collect_rollout(b)
        assert len(candidate) == len(baseline), f"{b} rollout length diverged"
        for step_i, (ref, cand) in enumerate(zip(baseline, candidate)):
            assert cand[0] == ref[0], f"{b} active agents diverged at step {step_i}"
            for a in ref[1]:
                np.testing.assert_allclose(
                    cand[1][a],
                    ref[1][a],
                    atol=1e-3,
                    err_msg=f"{b} obs diverged at step {step_i}",
                )
                assert cand[2][a] == pytest.approx(ref[2][a], abs=1e-5)
                assert cand[3][a] is ref[3][a]
                assert cand[4][a] is ref[4][a]
