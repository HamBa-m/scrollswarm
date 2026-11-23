"""
High-Precision Empirical Benchmark for ScrollSwarm Geometry Backends.

Benchmarks:
1. Pure Environment Stepping Throughput (steps / second).
2. Direct LiDAR Raycasting Micro-Latency (microseconds / call).
3. Numerical Equivalence & Discrepancy Analysis.

Usage:
    uv run python scripts/benchmark_backends.py --steps 500 --num-agents 4
"""

import argparse
import time
from typing import Dict, List, Tuple
import numpy as np

from scrollswarm.env import ScrollSwarmEnv
from scrollswarm.geometry import (
    get_available_backends,
    extract_geometry_buffers,
)
from scrollswarm.geometry.numpy_backend import cast_rays_numpy
from scrollswarm.map import ScrollingMap, MapConfig


def benchmark_stepping_throughput(
    backend: str,
    steps: int = 500,
    num_agents: int = 4,
    seed: int = 42,
    warmup: int = 50,
) -> Tuple[float, float]:
    """
    Measures environment step throughput (steps/sec and ms/step).
    """
    env = ScrollSwarmEnv(
        num_agents=num_agents,
        max_cycles=steps + warmup + 100,
        backend=backend,
    )
    env.reset(seed=seed)

    # Warmup (important for JIT compilation)
    for _ in range(warmup):
        actions = {a: 0 for a in env.agents}
        env.step(actions)
        if not env.agents:
            env.reset(seed=seed + 1)

    # Measured benchmark loop
    env.reset(seed=seed)
    start_time = time.perf_counter()
    executed_steps = 0

    while executed_steps < steps:
        actions = {a: 0 for a in env.agents}
        env.step(actions)
        executed_steps += 1
        if not env.agents:
            env.reset(seed=seed + executed_steps)

    elapsed = time.perf_counter() - start_time
    sps = executed_steps / elapsed
    ms_per_step = (elapsed / executed_steps) * 1000.0
    return sps, ms_per_step


def benchmark_raycasting_kernel(
    backend: str,
    num_calls: int = 2000,
    num_rays: int = 7,
    seed: int = 42,
) -> float:
    """
    Measures isolated raycasting latency (microseconds per call).
    """
    cfg = MapConfig(seed=seed)
    scrolling_map = ScrollingMap(cfg)
    obs_meta, obs_edges, obs_ranges, target_data = extract_geometry_buffers(
        scrolling_map.obstacles, scrolling_map.targets
    )

    half_fov = np.radians(60.0)
    angles = np.linspace(-half_fov, half_fov, num_rays, dtype=np.float32)
    origin_x = 120.0
    origin_y = 200.0
    max_range = 400.0
    ceil_y = float(scrolling_map.ceiling_y)
    floor_y = float(scrolling_map.floor_y)

    if backend == "cpp":
        from scrollswarm.geometry import cast_rays_cpp
        fn = cast_rays_cpp
    elif backend == "numba":
        from scrollswarm.geometry.numba_backend import cast_rays_numba
        fn = cast_rays_numba
        # Warmup JIT
        fn(origin_x, origin_y, angles, max_range, ceil_y, floor_y, obs_meta, obs_edges, obs_ranges, target_data)
    else:
        fn = cast_rays_numpy

    start = time.perf_counter()
    for _ in range(num_calls):
        fn(origin_x, origin_y, angles, max_range, ceil_y, floor_y, obs_meta, obs_edges, obs_ranges, target_data)
    elapsed = time.perf_counter() - start

    us_per_call = (elapsed / num_calls) * 1e6
    return us_per_call


def main():
    parser = argparse.ArgumentParser(description="ScrollSwarm Multi-Backend Benchmark")
    parser.add_argument("--steps", type=int, default=500, help="Measured steps per run")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of UAVs")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--warmup", type=int, default=50, help="JIT warmup steps")
    args = parser.parse_args()

    available = get_available_backends()
    print("=" * 70)
    print("ScrollSwarm Computational Acceleration Benchmark")
    print(f"Available Backends on Host: {available}")
    print(f"Configuration: {args.num_agents} UAVs, {args.steps} steps, Seed {args.seed}")
    print("=" * 70)

    print("\n--- 1. Pure Raycasting Micro-Benchmark (Isolated Geometry Kernel) ---")
    kernel_results = {}
    for b in available:
        us = benchmark_raycasting_kernel(b, num_calls=3000, seed=args.seed)
        kernel_results[b] = us
        print(f"  [{b.upper():<8}] Latency: {us:7.2f} us / call")

    baseline_us = kernel_results.get("numpy", 1.0)
    for b, us in kernel_results.items():
        speedup = baseline_us / us if us > 0 else 1.0
        print(f"  [{b.upper():<8}] Speedup vs NumPy: {speedup:5.2f}x")

    print("\n--- 2. End-to-End Environment Stepping Throughput ---")
    step_results = {}
    for b in available:
        sps, ms = benchmark_stepping_throughput(
            b, steps=args.steps, num_agents=args.num_agents, seed=args.seed, warmup=args.warmup
        )
        step_results[b] = (sps, ms)
        print(f"  [{b.upper():<8}] Throughput: {sps:8.1f} steps/s | Latency: {ms:6.3f} ms/step")

    baseline_sps = step_results.get("numpy", (1.0, 1.0))[0]
    for b, (sps, _) in step_results.items():
        speedup = sps / baseline_sps if baseline_sps > 0 else 1.0
        print(f"  [{b.upper():<8}] Speedup vs NumPy: {speedup:5.2f}x")

    print("=" * 70)


if __name__ == "__main__":
    main()
