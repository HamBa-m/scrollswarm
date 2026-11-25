"""
ScrollSwarm: Reusable Random Action Policy Evaluation & GIF Recorder.

This script executes a random exploration baseline across the swarm, prints
comprehensive episodic telemetry, and optionally records animated GIFs of the
flight trajectory.

Usage:
    # Run 1 episode with default 5 UAVs and save GIF
    uv run python scripts/run_random_policy.py --record-gif

    # Run multi-seed benchmark without recording
    uv run python scripts/run_random_policy.py --episodes 5 --no-record-gif
"""

import argparse
import os
import shutil
from typing import Dict, Any, List
import numpy as np
from PIL import Image

from scrollswarm.env import ScrollSwarmEnv
from scrollswarm.comm import TopologyType


def evaluate_random_policy(
    num_agents: int = 5,
    max_cycles: int = 300,
    seed: int = 42,
    record_gif: bool = True,
    gif_path: str = "assets/random_policy.gif",
    step_skip: int = 2,
) -> Dict[str, Any]:
    """
    Executes an episode of the Random Action Policy and optionally saves a GIF.
    """
    if record_gif:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    render_mode = "rgb_array" if record_gif else None
    env = ScrollSwarmEnv(
        num_agents=num_agents,
        max_cycles=max_cycles,
        comm_topology=TopologyType.CHAIN,
        target_reward=1.0,
        crash_penalty=10.0,
        step_reward=0.01,
        render_mode=render_mode,
    )

    observations, infos = env.reset(seed=seed)

    # Telemetry tracker for all agents
    tracker = {
        a: {
            "cumulative_reward": 0.0,
            "steps_survived": 0,
            "targets_photographed": 0,
            "collision": None,
        }
        for a in env.possible_agents
    }

    frames: List[Image.Image] = []
    step_count = 0

    if record_gif:
        initial_frame = env.render()
        if initial_frame is not None:
            frames.append(Image.fromarray(initial_frame))

    while env.agents:
        step_count += 1

        # Sample random actions from Discrete(3) for all active drones
        actions = {a: env.action_space(a).sample() for a in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)

        for a, r in rewards.items():
            tracker[a]["cumulative_reward"] += r

        for a, info in infos.items():
            tracker[a]["steps_survived"] = info["steps_alive"]
            tracker[a]["targets_photographed"] = info["targets_photographed"]
            if info["collision"] is not None:
                tracker[a]["collision"] = info["collision"]

        if record_gif and (step_count % step_skip == 0):
            frame = env.render()
            if frame is not None:
                frames.append(Image.fromarray(frame))

    env.close()

    # Save animated GIF
    if record_gif and frames:
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        # 50ms per frame * step_skip 2 = 100ms per simulated step (~10 FPS animation)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=60,
            loop=0,
            optimize=True,
        )
        print(f"[GIF] Saved {len(frames)} frames to: {gif_path}")

        # Mirror copy to docs/assets if applicable
        docs_gif_path = os.path.join("docs", gif_path)
        os.makedirs(os.path.dirname(docs_gif_path), exist_ok=True)
        shutil.copyfile(gif_path, docs_gif_path)
        print(f"[GIF] Copied mirror to: {docs_gif_path}")

    # Compute summary metrics
    total_return = sum(d["cumulative_reward"] for d in tracker.values())
    total_targets = sum(d["targets_photographed"] for d in tracker.values())
    mean_survival = np.mean([d["steps_survived"] for d in tracker.values()])
    crashes = sum(1 for d in tracker.values() if d["collision"] is not None)

    return {
        "seed": seed,
        "total_return": total_return,
        "total_targets": total_targets,
        "mean_survival": mean_survival,
        "crashes": crashes,
        "total_drones": len(tracker),
        "details": tracker,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ScrollSwarm Random Policy Evaluator & GIF Recorder"
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of benchmark episodes to run"
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=300,
        help="Maximum simulation steps per episode",
    )
    parser.add_argument(
        "--num-agents", type=int, default=5, help="Number of UAVs in the swarm (1-10)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Initial random seed")
    parser.add_argument(
        "--record-gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record animated GIF",
    )
    parser.add_argument(
        "--gif-path",
        type=str,
        default="assets/random_policy.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--step-skip", type=int, default=2, help="Frame recording sub-sample rate"
    )
    args = parser.parse_args()

    print(f"=== Running Random Policy Evaluation ({args.episodes} Episode(s)) ===")
    all_returns = []
    all_survivals = []
    all_targets = []
    total_crashes = 0
    total_drones = 0

    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        should_record = args.record_gif and (ep == 0)  # Record first episode
        res = evaluate_random_policy(
            num_agents=args.num_agents,
            max_cycles=args.max_cycles,
            seed=ep_seed,
            record_gif=should_record,
            gif_path=args.gif_path,
            step_skip=args.step_skip,
        )

        all_returns.append(res["total_return"])
        all_survivals.append(res["mean_survival"])
        all_targets.append(res["total_targets"])
        total_crashes += res["crashes"]
        total_drones += res["total_drones"]

        print(f"\n[Episode {ep + 1}/{args.episodes} | Seed {ep_seed}]")
        print(f"  Team Return:        {res['total_return']:8.2f}")
        print(
            f"  Mean Survival:      {res['mean_survival']:8.1f} / {args.max_cycles} steps"
        )
        print(f"  Targets Captured:   {res['total_targets']:8d}")
        print(f"  Drone Crashes:      {res['crashes']}/{res['total_drones']}")
        for agent_id, d in res["details"].items():
            status = (
                "SURVIVED" if d["collision"] is None else f"CRASHED ({d['collision']})"
            )
            print(
                f"    {agent_id:8s} | Steps: {d['steps_survived']:3d} | Return: {d['cumulative_reward']:6.2f} | Status: {status}"
            )

    if args.episodes > 1:
        print("\n=== Aggregate Benchmark Summary ===")
        print(
            f"  Mean Return:      {np.mean(all_returns):8.2f} ± {np.std(all_returns):.2f}"
        )
        print(
            f"  Mean Survival:    {np.mean(all_survivals):8.1f} ± {np.std(all_survivals):.1f} / {args.max_cycles} steps"
        )
        print(
            f"  Mean Targets:     {np.mean(all_targets):8.2f} ± {np.std(all_targets):.2f}"
        )
        print(
            f"  Crash Rate:       {total_crashes / total_drones * 100:7.1f}% ({total_crashes}/{total_drones})"
        )


if __name__ == "__main__":
    main()
