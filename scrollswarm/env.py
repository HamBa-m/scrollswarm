"""
PettingZoo ParallelEnv implementation for ScrollSwarm.

Provides a standard Multi-Agent Reinforcement Learning (MARL) interface:
- Observations: Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32) from 7-ray LiDAR.
- Actions: Discrete(3) -> {0: UP, 1: STAY, 2: DOWN}.
- Rewards: Raw ground-truth signal (+1.0 on target collection, 0.0 otherwise).
- Terminations: True on collision with mountain boundary or tree.
- Truncations: True when max_cycles (default: 1000) is reached.
- Telemetry: Rich collision and score metadata in infos dictionary.
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
    COLOR_SPACE_VOID,
    COLOR_RING_HAZE,
    COLOR_RING_SHEATH,
    COLOR_RING_RIM,
    COLOR_ASTEROID_FILL,
    COLOR_ASTEROID_OUTLINE,
    COLOR_ASTEROID_SPECULAR,
    COLOR_TARGET,
)
from scrollswarm.agent import Agent, Action, create_agents, COLOR_AGENT_DEAD
from scrollswarm.sensors import RaySensor
from scrollswarm.comm import CommunicationGraph, TopologyType
from scrollswarm.sprites import get_rocket_sprite


class ScrollSwarmEnv(ParallelEnv):
    """
    ScrollSwarm standard PettingZoo Parallel Environment.
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
            a: spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: spaces.Discrete(3)
            for a in self.possible_agents
        }

        # Step counter for truncation
        self.step_count = 0

        # Physical map engine & agent instances
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
        15-dimensional observation vector:
        [self_y / H, 7 normalized obstacle ray distances, 7 normalized target ray distances]
        """
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(15,),
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
        Concatenation of 15-dimensional observation vectors across all possible agents.
        Shape: (15 * num_agents,)
        """
        total_dim = 15 * self._num_agents
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_dim,),
            dtype=np.float32,
        )

    def state(self) -> np.ndarray:
        """
        Returns the centralized global state vector for CTDE (e.g. MAPPO, QMIX).
        Formed by concatenating the 15-dimensional observation vectors across all possible agents.
        Dead agents contribute a zero vector of shape (15,).
        """
        parts = []
        for agent_id in self.possible_agents:
            agent = self._agent_map.get(agent_id)
            if agent is not None and agent.alive:
                obs = agent.get_observation(self.scrolling_map)
            else:
                obs = np.zeros(15, dtype=np.float32)
            parts.append(obs)
        return np.concatenate(parts, dtype=np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """
        Resets the environment, generating a fresh multi-scale corridor and placing agents.
        Returns:
            observations: Dict[agent_id, np.ndarray of shape (15,)]
            infos: Dict[agent_id, dict]
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

        observations = {
            a: self._agent_map[a].get_observation(self.scrolling_map)
            for a in self.agents
        }
        infos = {
            a: {
                "targets_collected": 0,
                "collision": None,
                "steps_survived": 0,
            }
            for a in self.agents
        }

        if self.render_mode == "human":
            self.render()

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
        Executes a step for all active agents.
        Returns:
            (observations, rewards, terminations, truncations, infos)
        """
        self.step_count += 1
        is_truncated = self.step_count >= self.max_cycles

        rewards: Dict[str, float] = {}
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        infos: Dict[str, Dict[str, Any]] = {}

        # 1. Apply vertical movements for all currently active agents
        for agent_id in self.agents:
            act = actions.get(agent_id, Action.STAY)
            agent = self._agent_map[agent_id]
            if agent.alive:
                agent.step(act, self.scrolling_map.height)

        # 2. Advance scrolling world by 1 pixel
        self.scrolling_map.step()

        # 3. Check collisions, target collections, and calculate individual transitions
        indiv_rewards: Dict[str, float] = {}
        for agent_id in self.agents:
            agent = self._agent_map[agent_id]
            collision_type = None

            if agent.alive:
                # Check mountain collision
                if agent.check_collision_mountain(self.scrolling_map):
                    collision_type = "mountain"
                # Check asteroid obstacle collision
                elif agent.check_collision_obstacles(self.scrolling_map.asteroids):
                    collision_type = "asteroid"

                # Check target collection
                collected = agent.check_target_collection(self.scrolling_map.targets)

                # Reward formulation: penalty on crash (-10), survival bonus (+0.01), target collection (+1.0)
                if collision_type is not None:
                    r = -self.crash_penalty
                else:
                    r = self.step_reward + (self.target_reward * float(collected))
            else:
                r = 0.0

            indiv_rewards[agent_id] = r
            is_terminated = not agent.alive

            terminations[agent_id] = is_terminated
            truncations[agent_id] = is_truncated
            infos[agent_id] = {
                "targets_collected": agent.score,
                "collision": collision_type,
                "steps_survived": agent.steps_alive,
                "individual_reward": r,
            }

        # Scaled cooperative team reward (mean across total swarm size)
        team_reward_val = sum(indiv_rewards.values()) / float(self._num_agents)

        for agent_id in self.agents:
            infos[agent_id]["team_reward"] = team_reward_val
            if self.team_reward:
                rewards[agent_id] = team_reward_val
            else:
                rewards[agent_id] = indiv_rewards[agent_id]

        # 4. Generate observations for agents that are still active
        observations = {
            agent_id: self._agent_map[agent_id].get_observation(self.scrolling_map)
            for agent_id in self.agents
        }

        # 5. Filter dead or truncated agents per PettingZoo parallel specification
        self.agents = [
            a for a in self.agents if not (terminations[a] or truncations[a])
        ]

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def get_adjacency_matrix(self) -> np.ndarray:
        """Returns the current (N, N) binary adjacency matrix for alive agents."""
        return self.comm_graph.get_adjacency_matrix(list(self._agent_map.values()))

    def get_edge_index_and_attributes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns PyG-compatible graph representations for active agents:
            edge_index: (2, E)
            edge_attr:  (E, 3) relative [dx, dy, normalized_dist]
        """
        return self.comm_graph.get_edge_index_and_attributes(
            list(self._agent_map.values()),
            map_width=float(self.scrolling_map.width),
            map_height=float(self.scrolling_map.height),
        )

    def render(self) -> Optional[np.ndarray]:
        """Renders the environment in 'human' (window) or 'rgb_array' mode."""
        if self.render_mode is None:
            return None

        import pygame

        w = self.scrolling_map.width
        h = self.scrolling_map.height

        if not self._pygame_initialized:
            pygame.init()
            pygame.font.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((w, h))
                pygame.display.set_caption(f"ScrollSwarm ({self._num_agents} Agents)")
                self._clock = pygame.time.Clock()
            else:
                # Headless offscreen surface for rgb_array
                self._screen = pygame.Surface((w, h))

            self._font = pygame.font.SysFont("Consolas", 13)
            self._id_font = pygame.font.SysFont("Consolas", 10, bold=True)
            self._pygame_initialized = True

        surface = self._screen

        # 0. Cosmic void background & subtle Saturnian ring dust streams
        surface.fill(COLOR_SPACE_VOID)
        for offset_lane in (80, 150, 220, 290):
            dust_y = offset_lane
            pygame.draw.line(surface, (20, 26, 40), (0, dust_y), (w, dust_y), 1)

        # 1. Ring sheath boundary planes (flat boundaries with luminous rims)
        ceil_y = int(self.scrolling_map.ceiling_y)
        floor_y = int(self.scrolling_map.floor_y)
        pygame.draw.rect(surface, COLOR_RING_SHEATH, (0, 0, w, ceil_y))
        pygame.draw.rect(surface, COLOR_RING_SHEATH, (0, floor_y, w, h - floor_y))
        pygame.draw.line(surface, COLOR_RING_RIM, (0, ceil_y), (w, ceil_y), 2)
        pygame.draw.line(surface, COLOR_RING_RIM, (0, floor_y), (w, floor_y), 2)

        # 2. Dynamic polygon asteroids & icy moonlets
        for ast in getattr(self.scrolling_map, "asteroids", []):
            verts = [(int(vx), int(vy)) for vx, vy in ast.vertices]
            if len(verts) >= 3:
                pygame.draw.polygon(surface, COLOR_ASTEROID_FILL, verts)
                pygame.draw.polygon(surface, COLOR_ASTEROID_OUTLINE, verts, width=2)
                # Frosted ice specular facet line
                pygame.draw.line(surface, COLOR_ASTEROID_SPECULAR, verts[0], verts[2], 1)

        # 3. Targets (golden Cassini telemetry canisters)
        for tgt in self.scrolling_map.targets:
            if not tgt.collected:
                cx, cy, s = int(tgt.x), int(tgt.y), tgt.size
                pygame.draw.circle(surface, COLOR_TARGET, (cx, cy), s)
                pygame.draw.circle(surface, (255, 235, 120), (cx, cy), s + 3, width=1)
                pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 2)

        # 4. Agents (rocket probe sprites)
        all_agents = list(self._agent_map.values())
        for agent in all_agents:
            color = agent.color if agent.alive else COLOR_AGENT_DEAD
            sprite = get_rocket_sprite(color, size=(24, 24))
            sw, sh = sprite.get_size()
            surface.blit(sprite, (int(agent.x - sw // 2), int(agent.y - sh // 2)))

            # Draw indicator only when dead
            if not agent.alive:
                pygame.draw.circle(surface, (180, 50, 50), (int(agent.x), int(agent.y)), 13, width=1)

            # ID label above probe
            lbl = self._id_font.render(str(agent.agent_id), True, (220, 225, 235))
            surface.blit(lbl, (int(agent.x - 3), int(agent.y - sh // 2 - 12)))

        # 5. Telemetry HUD
        hud_bg = pygame.Surface((w - 24, 28), pygame.SRCALPHA)
        pygame.draw.rect(hud_bg, (16, 22, 32, 185), (0, 0, w - 24, 28), border_radius=4)
        surface.blit(hud_bg, (12, 18))

        alive_count = sum(1 for a in all_agents if a.alive)
        team_score = sum(a.score for a in all_agents)
        txt1 = self._font.render(
            f"Step: {self.step_count}/{self.max_cycles} | Probes Active: {alive_count}/{self._num_agents} | Team Telemetry: {team_score}",
            True,
            (240, 245, 250),
        )
        surface.blit(txt1, (22, 23))

        if self.render_mode == "human":
            pygame.display.flip()
            if self._clock:
                self._clock.tick(self.metadata["render_fps"])
            return None

        elif self.render_mode == "rgb_array":
            # Convert surface to (H, W, 3) NumPy array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(surface)),
                axes=(1, 0, 2),
            )

    def close(self) -> None:
        """Closes Pygame windows and cleans up display resources."""
        if self._pygame_initialized:
            import pygame
            pygame.quit()
            self._pygame_initialized = False
            self._screen = None
