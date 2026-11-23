"""
PettingZoo ParallelEnv implementation for ScrollSwarm (UAV Swarm Reconnaissance).

Provides a standard Multi-Agent Reinforcement Learning (MARL) interface:
- Observations: Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
    [self_y/H, 7 LiDAR ray obstacle distances, 4 nearest target features, 4 nearest neighbor features]
- Actions: Discrete(3) -> {0: UP, 1: STAY, 2: DOWN}
- Rewards:
    - Target inspection under action uncertainty: +1.0
    - Survival step: +0.01
    - Obstacle / boundary collision: -10.0 (causes individual termination)
- Terminations: True on collision with boundary or obstacle.
- Truncations: True when max_cycles (default: 1000) is reached.
- Rendering: Supports both "human" (Pygame window) and "rgb_array" (NumPy image tensor).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import functools
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

from scrollswarm.map import (
    ScrollingMap,
    MapConfig,
    COLOR_AIRSPACE,
    COLOR_GRID_LINE,
    COLOR_CEILING,
    COLOR_GROUND,
    COLOR_BOUNDARY_RIM,
    COLOR_BUILDING_FILL,
    COLOR_BUILDING_OUTLINE,
    COLOR_BUILDING_SPECULAR,
    COLOR_HAZARD_FILL,
    COLOR_HAZARD_OUTLINE,
    COLOR_TARGET,
    COLOR_TARGET_INSPECTED,
)
from scrollswarm.items import ObstacleType
from scrollswarm.agent import Agent, Action, create_agents, COLOR_AGENT_DEAD
from scrollswarm.sensors import RaySensor
from scrollswarm.comm import CommunicationGraph, TopologyType
from scrollswarm.sprites import get_quadcopter_sprite


class ScrollSwarmEnv(ParallelEnv):
    """
    ScrollSwarm standard PettingZoo Parallel Environment for UAV Swarm Reconnaissance.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "scrollswarm_v0",
        "render_fps": 30,
    }

    def __init__(
        self,
        num_agents: int = 5,
        max_cycles: int = 1000,
        map_config: Optional[MapConfig] = None,
        comm_topology: Union[TopologyType, str] = TopologyType.CHAIN,
        target_reward: float = 1.0,
        crash_penalty: float = 10.0,
        step_reward: float = 0.01,
        team_reward: bool = False,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        if not (1 <= num_agents <= 10):
            raise ValueError(f"num_agents must be between 1 and 10, got {num_agents}")

        self._num_agents = num_agents
        self.max_cycles = max_cycles
        self.map_config = map_config or MapConfig(width=800, height=400, fps=30)
        self.comm_topology = comm_topology
        self.comm_graph = CommunicationGraph(topology=comm_topology)
        self.target_reward = float(target_reward)
        self.crash_penalty = float(crash_penalty)
        self.step_reward = float(step_reward)
        self.team_reward = bool(team_reward)
        self.render_mode = render_mode

        # Agent IDs conforming to PettingZoo standards
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agents = self.possible_agents[:]

        # Pre-populate space dictionaries
        self.observation_spaces = {
            a: spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: spaces.Discrete(3)
            for a in self.possible_agents
        }

        # Step counter for truncation
        self.step_count = 0

        # Physical map engine & agent entities
        self.scrolling_map = ScrollingMap(self.map_config)
        self._agent_map: Dict[str, Agent] = {}

        # Pygame rendering attributes (lazy initialized)
        self._pygame_initialized = False
        self._screen = None
        self._clock = None
        self._font = None
        self._id_font = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gym.Space:
        """
        16-dimensional observation vector:
        [self_y / H, 7 obstacle LiDAR distances, 4 target features, 4 neighbor features]
        """
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.Space:
        """Discrete(3): {0: UP (-5px), 1: STAY (0px), 2: DOWN (+5px)}."""
        return spaces.Discrete(3)

    @property
    def state_space(self) -> gym.Space:
        """
        Centralized state space for CTDE algorithms (MAPPO, QMIX, MADDPG).
        Concatenation of 16-dimensional observation vectors across all possible agents.
        Shape: (16 * num_agents,)
        """
        total_dim = 16 * self._num_agents
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(total_dim,),
            dtype=np.float32,
        )

    def state(self) -> np.ndarray:
        """
        Returns the centralized global state vector for CTDE.
        Concatenates observation vectors across all possible agents.
        Dead agents contribute a zero vector of shape (16,).
        """
        parts = []
        all_agents = list(self._agent_map.values())
        for agent_id in self.possible_agents:
            agent = self._agent_map.get(agent_id)
            if agent is not None and agent.alive:
                obs = agent.get_observation(self.scrolling_map, other_agents=all_agents)
            else:
                obs = np.zeros(16, dtype=np.float32)
            parts.append(obs)
        return np.concatenate(parts, dtype=np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """
        Resets the environment, generating a fresh corridor and placing UAVs.
        """
        self.step_count = 0
        self.agents = self.possible_agents[:]

        # Reset scrolling map with provided seed
        self.scrolling_map.reset(seed=seed)

        # Re-create internal agent entities centered in corridor
        agent_list = create_agents(
            num_agents=self._num_agents,
            map_width=self.scrolling_map.width,
            scrolling_map=self.scrolling_map,
            step_size=5,
        )
        self._agent_map = {
            f"agent_{i}": agent_list[i] for i in range(self._num_agents)
        }

        # Build initial observations and telemetry infos
        all_agents = list(self._agent_map.values())
        observations = {
            a: self._agent_map[a].get_observation(self.scrolling_map, other_agents=all_agents)
            for a in self.agents
        }
        infos = {
            a: {
                "targets_photographed": 0,
                "targets_collected": 0,
                "collision": None,
                "steps_alive": 0,
            }
            for a in self.agents
        }

        return observations, infos

    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """
        Executes one synchronized multi-agent step:
        1. Executes discrete vertical actions for all living UAVs.
        2. Advances scrolling map (obstacles and targets shift left).
        3. Checks boundary and obstacle collisions (instant individual termination).
        4. Calculates soft inter-agent proximity penalties.
        5. Evaluates stochastic target inspection under action uncertainty.
        6. Constructs PettingZoo standard dictionaries.
        """
        self.step_count += 1
        rewards: Dict[str, float] = {a: 0.0 for a in self.agents}
        terminations: Dict[str, bool] = {a: False for a in self.agents}
        truncations: Dict[str, bool] = {a: False for a in self.agents}
        infos: Dict[str, Dict[str, Any]] = {a: {} for a in self.agents}

        # 1. Execute vertical movement actions
        for agent_id in self.agents:
            if agent_id in actions:
                action = actions[agent_id]
                self._agent_map[agent_id].step(action, self.scrolling_map.height)

        # 2. Advance scrolling map
        self.scrolling_map.step()

        # 3. Collision checks (instant structural termination on obstacle/boundary contact)
        for agent_id in self.agents:
            agent = self._agent_map[agent_id]
            collision_cause = None

            if agent.check_collision_boundary(self.scrolling_map):
                collision_cause = "boundary"
            elif agent.check_collision_obstacles(self.scrolling_map.obstacles):
                collision_cause = "obstacle"

            if collision_cause is not None:
                terminations[agent_id] = True
                rewards[agent_id] -= self.crash_penalty
                infos[agent_id]["collision"] = collision_cause
            else:
                infos[agent_id]["collision"] = None
                rewards[agent_id] += self.step_reward

        # 4. Stochastic target inspection under action uncertainty
        # Evaluate all targets against living drones
        living_agents = [self._agent_map[a] for a in self.agents if self._agent_map[a].alive]
        targets_photographed_this_step = 0
        for target in self.scrolling_map.targets:
            if target.collected:
                continue
            if target.sample_inspection(living_agents, self.scrolling_map.rng):
                targets_photographed_this_step += 1
                # Find which drones contributed to the photo
                contributing_drones = [
                    d for d in living_agents
                    if np.sqrt((d.x - target.x) ** 2 + (d.y - target.y) ** 2) <= target.inspection_radius
                ]
                for d in contributing_drones:
                    d.score += 1

                if self.team_reward and living_agents:
                    team_share = self.target_reward / len(living_agents)
                    for d in living_agents:
                        rewards[f"agent_{d.agent_id}"] += team_share
                else:
                    for d in contributing_drones:
                        rewards[f"agent_{d.agent_id}"] += self.target_reward

        # 6. Check episode truncation
        is_truncated = self.step_count >= self.max_cycles
        for agent_id in self.agents:
            truncations[agent_id] = is_truncated
            agent = self._agent_map[agent_id]
            infos[agent_id]["targets_photographed"] = agent.score
            infos[agent_id]["targets_collected"] = agent.score  # Backward compatibility
            infos[agent_id]["steps_alive"] = agent.steps_alive

        # 7. Build observation vectors
        all_agents = list(self._agent_map.values())
        observations: Dict[str, np.ndarray] = {
            a: self._agent_map[a].get_observation(self.scrolling_map, other_agents=all_agents)
            for a in self.agents
        }

        # 8. Cull terminated or truncated agents from active list
        self.agents = [
            a for a in self.agents
            if not terminations[a] and not truncations[a]
        ]

        return observations, rewards, terminations, truncations, infos

    def render(self) -> Optional[np.ndarray]:
        """
        Renders the aerial survey corridor.
        In 'human' mode: updates live Pygame display window.
        In 'rgb_array' mode: returns an (H, W, 3) uint8 NumPy image array.
        """
        if self.render_mode is None:
            return None

        if not self._pygame_initialized:
            import pygame
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((self.map_config.width, self.map_config.height))
                pygame.display.set_caption("ScrollSwarm - UAV Swarm Reconnaissance")
            else:
                self._screen = pygame.Surface((self.map_config.width, self.map_config.height))
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("Consolas", 13)
            self._id_font = pygame.font.SysFont("Consolas", 10, bold=True)
            self._pygame_initialized = True

        import pygame

        w = self.scrolling_map.width
        h = self.scrolling_map.height

        # A. Airspace background & survey grid
        self._screen.fill(COLOR_AIRSPACE)
        for offset_lane in (60, 120, 180, 240, 300, 360):
            pygame.draw.line(self._screen, COLOR_GRID_LINE, (0, offset_lane), (w, offset_lane), width=1)

        # B. Ceiling and Ground boundaries
        ceil_y = int(self.scrolling_map.ceiling_y)
        floor_y = int(self.scrolling_map.floor_y)
        pygame.draw.rect(self._screen, COLOR_CEILING, (0, 0, w, ceil_y))
        pygame.draw.rect(self._screen, COLOR_GROUND, (0, floor_y, w, h - floor_y))
        pygame.draw.line(self._screen, COLOR_BOUNDARY_RIM, (0, ceil_y), (w, ceil_y), 2)
        pygame.draw.line(self._screen, COLOR_BOUNDARY_RIM, (0, floor_y), (w, floor_y), 2)

        # C. Multi-scale obstacles (buildings, overhead towers, floating hazards)
        for obs in self.scrolling_map.obstacles:
            verts = [(int(vx), int(vy)) for vx, vy in obs.vertices]
            if len(verts) >= 3:
                fill_color = COLOR_HAZARD_FILL if obs.obstacle_type == ObstacleType.FLOATING else COLOR_BUILDING_FILL
                edge_color = COLOR_HAZARD_OUTLINE if obs.obstacle_type == ObstacleType.FLOATING else COLOR_BUILDING_OUTLINE
                pygame.draw.polygon(self._screen, fill_color, verts)
                pygame.draw.polygon(self._screen, edge_color, verts, width=2)
                if obs.obstacle_type != ObstacleType.FLOATING:
                    pygame.draw.line(self._screen, COLOR_BUILDING_SPECULAR, verts[0], verts[2], 1)

        # D. Inspection targets (with photo radius halos)
        for tgt in self.scrolling_map.targets:
            cx, cy, s = int(tgt.x), int(tgt.y), tgt.size
            if tgt.collected:
                pygame.draw.circle(self._screen, COLOR_TARGET_INSPECTED, (cx, cy), s)
                pygame.draw.circle(self._screen, (255, 255, 255), (cx, cy), 2)
            else:
                # Translucent photo inspection radius ring
                pygame.draw.circle(self._screen, (255, 215, 0), (cx, cy), int(tgt.inspection_radius), width=1)
                pygame.draw.circle(self._screen, COLOR_TARGET, (cx, cy), s)
                pygame.draw.circle(self._screen, (255, 240, 150), (cx, cy), s + 3, width=1)
                pygame.draw.circle(self._screen, (255, 255, 255), (cx, cy), 2)

        # E. Communication graph links
        living_agents = [self._agent_map[a] for a in self.possible_agents if self._agent_map[a].alive]
        comm_edges = self.comm_graph.get_communication_edges(living_agents)
        for p1, p2 in comm_edges:
            pygame.draw.line(self._screen, (70, 140, 230), (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 1)

        # F. UAV Quadcopters
        for agent_id in self.possible_agents:
            agent = self._agent_map[agent_id]
            color = agent.color if agent.alive else COLOR_AGENT_DEAD
            sprite = get_quadcopter_sprite(color, size=(24, 24))
            sw, sh = sprite.get_size()
            self._screen.blit(sprite, (int(agent.x - sw // 2), int(agent.y - sh // 2)))

            # Agent ID label
            tag_surf = self._id_font.render(f"#{agent.agent_id + 1}", True, (255, 255, 255))
            self._screen.blit(tag_surf, (int(agent.x - tag_surf.get_width() // 2), int(agent.y - 18)))

        # Update display or export rgb array
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            # Pygame surface to NumPy uint8 array of shape (H, W, 3)
            img_array = pygame.surfarray.array3d(self._screen)
            return np.transpose(img_array, (1, 0, 2))

    def close(self) -> None:
        """Closes any open Pygame window and cleans up resources."""
        if self._pygame_initialized:
            import pygame
            pygame.quit()
            self._pygame_initialized = False

    def get_adjacency_matrix(self) -> np.ndarray:
        """Convenience method: returns the (N, N) binary adjacency matrix."""
        all_agents = [self._agent_map[a] for a in self.possible_agents]
        return self.comm_graph.get_adjacency_matrix(all_agents)

    def get_edge_index_and_attributes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience method: returns PyG-compatible edge_index (2, E) and edge_attr (E, 3)."""
        all_agents = [self._agent_map[a] for a in self.possible_agents]
        return self.comm_graph.get_edge_index_and_attributes(
            all_agents,
            map_width=float(self.scrolling_map.width),
            map_height=float(self.scrolling_map.height),
        )
