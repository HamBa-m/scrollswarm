"""
Empirical Vectorized Throughput Benchmark for ScrollSwarm.

Measures pure environment stepping throughput (zero neural network overhead)
for ``SyncVectorScrollSwarm`` and ``AsyncVectorScrollSwarm`` across corridor
batch sizes E in {1, 2, 4, 8, 16, 32, 64}, highlighting multi-core scaling
and the Sync vs. Async IPC crossover point.

Usage:
    uv run python scripts/benchmark_vector.py --steps 400 --num-agents 4
"""

import argparse
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from scrollswarm import SyncVectorScrollSwarm, AsyncVectorScrollSwarm
from scrollswarm.geometry import resolve_backend


def measure_vector_runner(
    mode: str,
    num_envs: int,
    num_agents: int,
    steps: int,
    seed: int,
    warmup: int,
    backend: str,
    max_workers: int,
) -> Tuple[float, float, float, float]:
    """
    Returns (batch_sps, env_sps, agent_sps, ms_per_batch).
    """
    common_kwargs = dict(
        num_envs=num_envs,
        num_agents=num_agents,
        max_cycles=1000,
        comm_topology="chain",
        backend=backend,
    )

    if mode == "async":
        runner = AsyncVectorScrollSwarm(
            **common_kwargs, num_workers=min(num_envs, max_workers)
        )
    else:
        runner = SyncVectorScrollSwarm(**common_kwargs)

    try:
        runner.reset(seed=seed)
        actions = np.full((num_envs, num_agents), 1, dtype=np.int64)

        for _ in range(warmup):
            runner.step(actions)

        runner.reset(seed=seed + 1000)
        start = time.perf_counter()
        for step_idx in range(steps):
            actions.fill(step_idx % 3)
            runner.step(actions)
        elapsed = time.perf_counter() - start
    finally:
        runner.close()

    batch_sps = steps / elapsed
    env_sps = batch_sps * num_envs
    agent_sps = env_sps * num_agents
    ms_per_batch = (elapsed / steps) * 1000.0
    return batch_sps, env_sps, agent_sps, ms_per_batch


def run_vector_sweep(
    env_counts: List[int],
    num_agents: int = 4,
    steps: int = 400,
    seed: int = 42,
    warmup: int = 20,
    backend: str = "auto",
    max_workers: int = 8,
    save_plot: Optional[str] = "docs/assets/benchmark_vector_scaling.png",
) -> Dict[str, Dict[int, Tuple[float, float, float, float]]]:
    """Runs Sync vs Async scaling sweep across ``env_counts`` and plots results."""
    resolved = resolve_backend(backend)
    results: Dict[str, Dict[int, Tuple[float, float, float, float]]] = {
        "sync": {},
        "async": {},
    }

    print("=" * 82)
    print("ScrollSwarm Vectorized Execution Benchmark (Pure Environment Stepping)")
    print(
        f"Backend: {backend} -> {resolved} | Agents/Env (N): {num_agents} | Steps/Run: {steps}"
    )
    print(f"Async Worker Cap: {max_workers} | Corridor Batch Sizes (E): {env_counts}")
    print("=" * 82)

    for e in env_counts:
        print(
            f"\n--> Evaluating Batch Size E = {e} corridors ({e * num_agents} total UAVs)..."
        )
        for mode in ("sync", "async"):
            metrics = measure_vector_runner(
                mode=mode,
                num_envs=e,
                num_agents=num_agents,
                steps=steps,
                seed=seed,
                warmup=warmup,
                backend=resolved,
                max_workers=max_workers,
            )
            results[mode][e] = metrics
            print(
                f"    [{mode.upper():<5}] Batch: {metrics[0]:7.1f} batches/s | "
                f"Env: {metrics[1]:8.1f} env-steps/s | "
                f"Agent: {metrics[2]:9.1f} agent-steps/s | "
                f"{metrics[3]:6.3f} ms/batch"
            )

    # Print Markdown-ready table
    print("\n" + "=" * 82)
    print("### Empirical Vector Scaling Summary")
    print(
        "| Batch Size (E) | Total UAVs | Sync (env-steps/s) | Async (env-steps/s) | Async (agent-steps/s) | Async vs Sync |"
    )
    print("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for e in env_counts:
        s_env = results["sync"][e][1]
        a_env = results["async"][e][1]
        a_agt = results["async"][e][2]
        ratio = a_env / s_env if s_env > 0 else 1.0
        print(
            f"| **E = {e}** | {e * num_agents} | {s_env:,.1f} | "
            f"**{a_env:,.1f}** | {a_agt:,.1f} | **{ratio:.2f}x** |"
        )
    print("=" * 82)

    if save_plot:
        import matplotlib.pyplot as plt

        x_pos = np.arange(len(env_counts))
        x_labels = [str(e) for e in env_counts]

        sync_env_sps = [results["sync"][e][1] for e in env_counts]
        async_env_sps = [results["async"][e][1] for e in env_counts]
        sync_agent_sps = [results["sync"][e][2] / 1000.0 for e in env_counts]
        async_agent_sps = [results["async"][e][2] / 1000.0 for e in env_counts]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

        # Panel 1: Total Environment Steps/s
        ax1.plot(
            x_pos,
            sync_env_sps,
            label="SyncVectorScrollSwarm (1 Core)",
            color="#d97706",
            marker="o",
            linestyle="--",
            linewidth=2.2,
            markersize=6,
        )
        ax1.plot(
            x_pos,
            async_env_sps,
            label=f"AsyncVectorScrollSwarm (<= {max_workers} Workers)",
            color="#0284c7",
            marker="s",
            linestyle="-",
            linewidth=2.4,
            markersize=6,
        )
        ax1.set_title(
            "Aggregate Environment Throughput vs. Batch Size",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax1.set_xlabel("Parallel Corridors (E)", fontsize=10, labelpad=8)
        ax1.set_ylabel("Total Environment Steps / Second", fontsize=10, labelpad=8)
        ax1.set_xticks(x_pos, x_labels)
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(frameon=True, fontsize=9, loc="upper left")

        # Panel 2: Aggregate Agent Transitions/s (thousands)
        ax2.plot(
            x_pos,
            sync_agent_sps,
            label="SyncVectorScrollSwarm (1 Core)",
            color="#d97706",
            marker="o",
            linestyle="--",
            linewidth=2.2,
            markersize=6,
        )
        ax2.plot(
            x_pos,
            async_agent_sps,
            label=f"AsyncVectorScrollSwarm (<= {max_workers} Workers)",
            color="#059669",
            marker="^",
            linestyle="-",
            linewidth=2.4,
            markersize=6,
        )
        ax2.set_title(
            f"Aggregate Agent Transitions / Second (N = {num_agents} UAVs)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax2.set_xlabel("Parallel Corridors (E)", fontsize=10, labelpad=8)
        ax2.set_ylabel(
            "Agent Transitions / Second (thousands)", fontsize=10, labelpad=8
        )
        ax2.set_xticks(x_pos, x_labels)
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(frameon=True, fontsize=9, loc="upper left")

        plt.tight_layout()
        plt.savefig(save_plot, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n[Artifact Generated] Vector scaling plot saved to: {save_plot}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="ScrollSwarm Vectorization Benchmark")
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--backend", type=str, default="auto")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--plot", type=str, default="docs/assets/benchmark_vector_scaling.png"
    )
    args = parser.parse_args()

    run_vector_sweep(
        env_counts=args.envs,
        num_agents=args.num_agents,
        steps=args.steps,
        seed=args.seed,
        warmup=args.warmup,
        backend=args.backend,
        max_workers=args.max_workers,
        save_plot=args.plot,
    )


if __name__ == "__main__":
    main()
