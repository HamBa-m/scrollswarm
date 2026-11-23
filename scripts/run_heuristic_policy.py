"""
ScrollSwarm: Reusable Reactive Heuristic Policy Evaluation & GIF Recorder.

This script executes the hierarchical rule-based ReactiveSwarmPolicy controller
(boundary safety, LiDAR obstacle evasion, target tracking, corridor centering),
prints comprehensive episodic telemetry, and optionally records animated GIFs
of the flight trajectory.

Usage:
    # Run 1 episode with default 5 UAVs and save GIF
    uv run python scripts/run_heuristic_policy.py --record-gif

    # Run multi-seed benchmark without recording
    uv run python scripts/run_heuristic_policy.py --episodes 5 --no-record-gif
"""

import argparse
import os
import shutil
from typing import Dict, Any, List
import numpy as np
from PIL import Image

from scrollswarm.env import ScrollSwarmEnv
from scrollswarm.comm import TopologyType
from scrollswarm.agent import Action


class ReactiveSwarmPolicy:
    """
    Hierarchical rule-based reactive policy for autonomous UAV navigation.

    Tiers:
      1. Boundary Constraints: Prevents ceiling and ground penetration.
      2. Forward LiDAR Evasion: Evaluates obstacle distance and votes on
         upper vs. lower ray clearance to steer away from imminent hazards.
      3. Target Tracking: Aligns vertical altitude with nearest target waypoint.
      4. Corridor Centering: Gently drifts toward the vertical corridor median.

    Parameters:
        ceiling_margin: Normalized altitude threshold below which UP is forbidden.
        floor_margin: Normalized altitude threshold above which DOWN is forbidden.
        lidar_threshold: Normalized distance threshold triggering emergency evasion.
        target_deadband: Deadband for target vertical alignment.
    """

    def __init__(
        self,
        ceiling_margin: float = 0.15,
        floor_margin: float = 0.85,
        lidar_threshold: float = 0.35,
        target_deadband: float = 0.03,
    ):
        self.ceiling_margin = ceiling_margin
        self.floor_margin = floor_margin
        self.lidar_threshold = lidar_threshold
        self.target_deadband = target_deadband

    def act(self, observation: np.ndarray) -> int:
        """
        Computes discrete action from the 16-dimensional continuous observation.

        Args:
            observation: 16-dimensional float32 array.

        Returns:
            Integer action: 0 (UP), 1 (STAY), or 2 (DOWN).
        """
        y_norm = observation[0]
        lidar = observation[1:8]
        target_dy = observation[9]

        # Tier 1: Hard Boundary Constraints
        if y_norm <= self.ceiling_margin:
            return Action.DOWN
        if y_norm >= self.floor_margin:
            return Action.UP

        # Tier 2: Forward LiDAR Obstacle Evasion
        min_obstacle_dist = np.min(lidar)
        if min_obstacle_dist < self.lidar_threshold:
            upper_clearance = np.mean(lidar[0:3])  # Rays 0, 1, 2 (-60°, -40°, -20°)
            lower_clearance = np.mean(lidar[4:7])  # Rays 4, 5, 6 (+20°, +40°, +60°)

            if upper_clearance > lower_clearance:
                return Action.UP
            else:
                return Action.DOWN

        # Tier 3: Target Waypoint Tracking
        if abs(target_dy) > self.target_deadband:
            return Action.DOWN if target_dy > 0 else Action.UP

        # Tier 4: Corridor Equilibrium Centering
        if y_norm < 0.45:
            return Action.DOWN
        elif y_norm > 0.55:
            return Action.UP

        return Action.STAY


def evaluate_heuristic_policy(
    num_agents: int = 5,
    max_cycles: int = 300,
    seed: int = 42,
    record_gif: bool = True,
    gif_path: str = "assets/heuristic_policy.gif",
    step_skip: int = 2,
    policy: ReactiveSwarmPolicy = None,
) -> Dict[str, Any]:
    """
    Executes an episode of the ReactiveSwarmPolicy and optionally saves a GIF.
    """
    if record_gif:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    policy = policy or ReactiveSwarmPolicy()
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

        # Query reactive policy for all living drones
        actions = {a: policy.act(observations[a]) for a in env.agents}

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
    parser = argparse.ArgumentParser(description="ScrollSwarm Heuristic Policy Evaluator & GIF Recorder")
    parser.add_argument("--episodes", type=int, default=1, help="Number of benchmark episodes to run")
    parser.add_argument("--max-cycles", type=int, default=300, help="Maximum simulation steps per episode")
    parser.add_argument("--num-agents", type=int, default=5, help="Number of UAVs in the swarm (1-10)")
    parser.add_argument("--seed", type=int, default=42, help="Initial random seed")
    parser.add_argument("--record-gif", action=argparse.BooleanOptionalAction, default=True, help="Record animated GIF")
    parser.add_argument("--gif-path", type=str, default="assets/heuristic_policy.gif", help="Output GIF path")
    parser.add_argument("--step-skip", type=int, default=2, help="Frame recording sub-sample rate")
    args = parser.parse_args()

    policy = ReactiveSwarmPolicy()

    print(f"=== Running Reactive Heuristic Policy Evaluation ({args.episodes} Episode(s)) ===")
    all_returns = []
    all_survivals = []
    all_targets = []
    total_crashes = 0
    total_drones = 0

    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        should_record = args.record_gif and (ep == 0)
        res = evaluate_heuristic_policy(
            num_agents=args.num_agents,
            max_cycles=args.max_cycles,
            seed=ep_seed,
            record_gif=should_record,
            gif_path=args.gif_path,
            step_skip=args.step_skip,
            policy=policy,
        )

        all_returns.append(res["total_return"])
        all_survivals.append(res["mean_survival"])
        all_targets.append(res["total_targets"])
        total_crashes += res["crashes"]
        total_drones += res["total_drones"]

        print(f"\n[Episode {ep + 1}/{args.episodes} | Seed {ep_seed}]")
        print(f"  Team Return:        {res['total_return']:8.2f}")
        print(f"  Mean Survival:      {res['mean_survival']:8.1f} / {args.max_cycles} steps")
        print(f"  Targets Captured:   {res['total_targets']:8d}")
        print(f"  Drone Crashes:      {res['crashes']}/{res['total_drones']}")
        for agent_id, d in res["details"].items():
            status = "SURVIVED" if d["collision"] is None else f"CRASHED ({d['collision']})"
            print(f"    {agent_id:8s} | Steps: {d['steps_survived']:3d} | Return: {d['cumulative_reward']:6.2f} | Status: {status}")

    if args.episodes > 1:
        print("\n=== Aggregate Benchmark Summary ===")
        print(f"  Mean Return:      {np.mean(all_returns):8.2f} ± {np.std(all_returns):.2f}")
        print(f"  Mean Survival:    {np.mean(all_survivals):8.1f} ± {np.std(all_survivals):.1f} / {args.max_cycles} steps")
        print(f"  Mean Targets:     {np.mean(all_targets):8.2f} ± {np.std(all_targets):.2f}")
        print(f"  Crash Rate:       {total_crashes / total_drones * 100:7.1f}% ({total_crashes}/{total_drones})")


if __name__ == "__main__":
    main()
