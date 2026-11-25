"""
Vectorized MARL Environments for ScrollSwarm (CTDE & Graph-Aware).

Provides synchronous (``SyncVectorScrollSwarm``) and multiprocess asynchronous
(``AsyncVectorScrollSwarm``) wrappers that run ``E`` independent corridors in
parallel while preserving explicit ``(E, N, ...)`` swarm axes, CTDE centralized
states ``(E, 16 * N)``, ``alive_mask``, dense/block-diagonal communication
graphs, and automatic episode resets.
"""

from abc import ABC, abstractmethod
import logging
import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from gymnasium import spaces

from scrollswarm.comm import TopologyType
from scrollswarm.env import ScrollSwarmEnv
from scrollswarm.map import MapConfig

logger = logging.getLogger(__name__)


def _extract_dense_graph(env: ScrollSwarmEnv) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts ``(N, N)`` adjacency and ``(N, N, 3)`` dense edge features."""
    adj = env.get_adjacency_matrix().astype(np.float32, copy=False)
    n = adj.shape[0]
    edge_attr_dense = np.zeros((n, n, 3), dtype=np.float32)

    edge_index, edge_attr = env.get_edge_index_and_attributes()
    if edge_index.shape[1] > 0:
        src_idx = edge_index[0]
        tgt_idx = edge_index[1]
        edge_attr_dense[tgt_idx, src_idx] = edge_attr

    return adj, edge_attr_dense


def _snapshot_env_state(
    env: ScrollSwarmEnv,
    obs_dict: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Packs single-env observations, CTDE state, liveness mask, and graph arrays."""
    n = len(env.possible_agents)
    obs = np.zeros((n, 16), dtype=np.float32)
    alive_mask = np.zeros((n,), dtype=bool)

    for i, agent_id in enumerate(env.possible_agents):
        agent = env._agent_map[agent_id]
        alive_mask[i] = bool(agent.alive)
        if agent_id in obs_dict:
            obs[i] = obs_dict[agent_id]

    state = env.state().astype(np.float32, copy=False)
    adj, edge_attr = _extract_dense_graph(env)
    return obs, state, alive_mask, adj, edge_attr


def _reset_single_env(
    env: ScrollSwarmEnv,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Resets one ``ScrollSwarmEnv`` and extracts its initial vectorized state."""
    obs_dict, info_dict = env.reset(seed=seed)
    obs, state, alive_mask, adj, edge_attr = _snapshot_env_state(env, obs_dict)
    return obs, state, alive_mask, adj, edge_attr, {"agents": info_dict}


def _step_and_autoreset(
    env: ScrollSwarmEnv,
    action_row: np.ndarray,
    autoreset_seed: Optional[int],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    """
    Steps one ``ScrollSwarmEnv`` and auto-resets it immediately if all agents
    have terminated or truncated.
    """
    n = len(env.possible_agents)
    active_set = set(env.agents)
    actions_dict = {
        agent_id: int(action_row[i])
        for i, agent_id in enumerate(env.possible_agents)
        if agent_id in active_set
    }

    raw_obs, raw_rewards, raw_terms, raw_truncs, raw_infos = env.step(actions_dict)

    rewards = np.zeros((n,), dtype=np.float32)
    terminations = np.zeros((n,), dtype=bool)
    truncations = np.zeros((n,), dtype=bool)
    for i, agent_id in enumerate(env.possible_agents):
        if agent_id in raw_rewards:
            rewards[i] = float(raw_rewards[agent_id])
            terminations[i] = bool(raw_terms[agent_id])
            truncations[i] = bool(raw_truncs[agent_id])

    step_obs, step_state, step_alive, step_adj, step_edge_attr = _snapshot_env_state(
        env, raw_obs
    )
    episode_done = len(env.agents) == 0

    if episode_done:
        final_obs = step_obs
        final_state = step_state
        obs, state, alive_mask, adj, edge_attr, reset_info = _reset_single_env(
            env, autoreset_seed
        )
        env_info = {
            "final_info": raw_infos,
            "agents": reset_info["agents"],
        }
    else:
        final_obs = np.zeros((n, 16), dtype=np.float32)
        final_state = np.zeros((16 * n,), dtype=np.float32)
        obs = step_obs
        state = step_state
        alive_mask = step_alive
        adj = step_adj
        edge_attr = step_edge_attr
        env_info = {"agents": raw_infos}

    return (
        obs,
        rewards,
        terminations,
        truncations,
        state,
        alive_mask,
        adj,
        edge_attr,
        episode_done,
        final_obs,
        final_state,
        env_info,
    )


def _collate_step_results(
    results: List[Tuple[Any, ...]],
    num_envs: int,
    num_agents: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Stacks per-environment step tuples into contiguous ``(E, N, ...)`` arrays."""
    obs = np.empty((num_envs, num_agents, 16), dtype=np.float32)
    rewards = np.empty((num_envs, num_agents), dtype=np.float32)
    terminations = np.empty((num_envs, num_agents), dtype=bool)
    truncations = np.empty((num_envs, num_agents), dtype=bool)
    state = np.empty((num_envs, 16 * num_agents), dtype=np.float32)
    alive_mask = np.empty((num_envs, num_agents), dtype=bool)
    adj = np.empty((num_envs, num_agents, num_agents), dtype=np.float32)
    edge_attr = np.empty((num_envs, num_agents, num_agents, 3), dtype=np.float32)
    episode_done = np.empty((num_envs,), dtype=bool)
    final_obs = np.empty((num_envs, num_agents, 16), dtype=np.float32)
    final_state = np.empty((num_envs, 16 * num_agents), dtype=np.float32)
    env_infos: List[Dict[str, Any]] = []

    for e, row in enumerate(results):
        obs[e] = row[0]
        rewards[e] = row[1]
        terminations[e] = row[2]
        truncations[e] = row[3]
        state[e] = row[4]
        alive_mask[e] = row[5]
        adj[e] = row[6]
        edge_attr[e] = row[7]
        episode_done[e] = row[8]
        final_obs[e] = row[9]
        final_state[e] = row[10]
        env_infos.append(row[11])

    infos: Dict[str, Any] = {
        "state": state,
        "alive_mask": alive_mask,
        "adj": adj,
        "edge_attr": edge_attr,
        "episode_done": episode_done,
        "final_observation": final_obs,
        "final_state": final_state,
        "env_infos": env_infos,
    }
    return obs, rewards, terminations, truncations, infos


def _collate_reset_results(
    results: List[Tuple[Any, ...]],
    num_envs: int,
    num_agents: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Stacks per-environment reset tuples into contiguous ``(E, N, ...)`` arrays."""
    obs = np.empty((num_envs, num_agents, 16), dtype=np.float32)
    state = np.empty((num_envs, 16 * num_agents), dtype=np.float32)
    alive_mask = np.empty((num_envs, num_agents), dtype=bool)
    adj = np.empty((num_envs, num_agents, num_agents), dtype=np.float32)
    edge_attr = np.empty((num_envs, num_agents, num_agents, 3), dtype=np.float32)
    env_infos: List[Dict[str, Any]] = []

    for e, row in enumerate(results):
        obs[e] = row[0]
        state[e] = row[1]
        alive_mask[e] = row[2]
        adj[e] = row[3]
        edge_attr[e] = row[4]
        env_infos.append(row[5])

    infos: Dict[str, Any] = {
        "state": state,
        "alive_mask": alive_mask,
        "adj": adj,
        "edge_attr": edge_attr,
        "env_infos": env_infos,
    }
    return obs, infos


def _worker_loop(
    conn: Any,
    env_count: int,
    env_kwargs: Dict[str, Any],
) -> None:
    """Subprocess worker loop managing ``env_count`` headless ``ScrollSwarmEnv`` instances."""
    envs: List[ScrollSwarmEnv] = []
    try:
        envs = [
            ScrollSwarmEnv(**env_kwargs, render_mode=None) for _ in range(env_count)
        ]
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                seeds: List[Optional[int]] = payload
                res = [_reset_single_env(env, s) for env, s in zip(envs, seeds)]
                conn.send(("ok", res))
            elif cmd == "step":
                actions_chunk, autoreset_seeds = payload
                res = [
                    _step_and_autoreset(env, actions_chunk[i], autoreset_seeds[i])
                    for i, env in enumerate(envs)
                ]
                conn.send(("ok", res))
            elif cmd == "close":
                break
            else:
                raise ValueError(f"Unknown worker command: {cmd!r}")
    except EOFError:
        pass
    except Exception as exc:
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        for env in envs:
            env.close()
        conn.close()


class BaseVectorScrollSwarm(ABC):
    """
    Abstract base class for vectorized CTDE & graph-aware ScrollSwarm runners.

    Manages ``num_envs`` parallel corridors of ``num_agents`` UAVs and exposes
    batched ``(E, N, ...)`` spaces, auto-reset seeding, and block-diagonal PyG
    graph conversion utilities.
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int = 5,
        max_cycles: int = 1000,
        map_config: Optional[MapConfig] = None,
        comm_topology: Union[TopologyType, str] = TopologyType.CHAIN,
        target_reward: float = 1.0,
        crash_penalty: float = 10.0,
        step_reward: float = 0.01,
        team_reward: bool = False,
        backend: str = "auto",
    ):
        """
        Initializes shared spaces and environment configuration.

        Args:
            num_envs: Number of parallel corridors ``E`` (must be >= 1).
            num_agents: Number of UAVs ``N`` per corridor in ``[1, 10]``.
            max_cycles: Episode truncation horizon per corridor.
            map_config: Optional ``MapConfig`` applied to all corridors.
            comm_topology: Graph topology for inter-drone message passing.
            target_reward: Reward for photographing a target.
            crash_penalty: Penalty applied on fatal collision.
            step_reward: Per-step survival reward.
            team_reward: Whether target rewards are shared among living drones.
            backend: Geometry backend (``"auto"``, ``"numpy"``, ``"numba"``, ``"cpp"``).
        """
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if not (1 <= num_agents <= 10):
            raise ValueError(f"num_agents must be between 1 and 10, got {num_agents}")

        self.num_envs = int(num_envs)
        self.num_agents = int(num_agents)
        self._env_kwargs: Dict[str, Any] = {
            "num_agents": self.num_agents,
            "max_cycles": max_cycles,
            "map_config": map_config,
            "comm_topology": comm_topology,
            "target_reward": target_reward,
            "crash_penalty": crash_penalty,
            "step_reward": step_reward,
            "team_reward": team_reward,
            "backend": backend,
        }
        self._closed = False
        self._episode_counter = 0
        self._base_seed: Optional[int] = None

        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=np.float32
        )
        self.single_action_space = spaces.Discrete(3)
        self.single_state_space = spaces.Box(
            low=-1.0, high=1.0, shape=(16 * self.num_agents,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, self.num_agents, 16),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            np.full((self.num_envs, self.num_agents), 3, dtype=np.int64)
        )
        self.state_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, 16 * self.num_agents),
            dtype=np.float32,
        )

    def _initial_seeds(self, seed: Optional[int]) -> List[Optional[int]]:
        """Generates distinct deterministic per-corridor seeds when ``seed`` is set."""
        self._base_seed = seed
        self._episode_counter = self.num_envs
        if seed is None:
            return [None] * self.num_envs
        return [int(seed) + e for e in range(self.num_envs)]

    def _next_autoreset_seeds(self) -> List[Optional[int]]:
        """Generates fresh deterministic seeds for any environments that auto-reset."""
        if self._base_seed is None:
            return [None] * self.num_envs
        seeds = [
            int(self._base_seed) + self._episode_counter + e
            for e in range(self.num_envs)
        ]
        self._episode_counter += self.num_envs
        return seeds

    def _validate_actions(self, actions: np.ndarray) -> np.ndarray:
        """Validates and normalizes an ``(E, N)`` action array."""
        arr = np.asarray(actions, dtype=np.int64)
        expected = (self.num_envs, self.num_agents)
        if arr.shape != expected:
            raise ValueError(f"Expected actions shape {expected}, got {arr.shape}")
        return arr

    @staticmethod
    def batch_graphs(
        adj: np.ndarray,
        edge_attr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts dense ``(E, N, N)`` graphs into a block-diagonal PyG-ready COO batch.

        Each corridor ``e`` gets node indices offset by ``e * N``, guaranteeing
        strict inter-corridor edge isolation.

        Args:
            adj: Binary adjacency tensor of shape ``(E, N, N)``.
            edge_attr: Dense spatial feature tensor of shape ``(E, N, N, 3)``.

        Returns:
            edge_index: ``int64`` array of shape ``(2, E_total)`` with ``[src, tgt]``.
            edge_attr_flat: ``float32`` array of shape ``(E_total, 3)``.
            batch: ``int64`` graph membership vector of shape ``(E * N,)``.
        """
        num_envs, num_agents, _ = adj.shape
        env_idx, tgt_idx, src_idx = np.where(adj > 0.5)

        if env_idx.size == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr_flat = np.zeros((0, 3), dtype=np.float32)
        else:
            offset = env_idx.astype(np.int64) * num_agents
            src_flat = offset + src_idx.astype(np.int64)
            tgt_flat = offset + tgt_idx.astype(np.int64)
            edge_index = np.stack([src_flat, tgt_flat], axis=0)
            edge_attr_flat = edge_attr[env_idx, tgt_idx, src_idx].astype(
                np.float32, copy=False
            )

        batch = np.repeat(np.arange(num_envs, dtype=np.int64), num_agents)
        return edge_index, edge_attr_flat, batch

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets all parallel corridors and returns batched observations and infos."""

    @abstractmethod
    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Steps all parallel corridors synchronously and returns batched transitions."""

    @abstractmethod
    def close(self) -> None:
        """Releases environment or worker process resources."""

    def __enter__(self) -> "BaseVectorScrollSwarm":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class SyncVectorScrollSwarm(BaseVectorScrollSwarm):
    """
    Single-process synchronous vectorized runner for ``ScrollSwarmEnv``.

    Steps ``num_envs`` independent corridors sequentially in Python and packs
    observations, CTDE states, liveness masks, and communication graphs into
    contiguous ``(E, N, ...)`` NumPy arrays for batched neural policy inference.
    """

    def __init__(self, num_envs: int, **env_kwargs: Any):
        super().__init__(num_envs=num_envs, **env_kwargs)
        self.envs: List[ScrollSwarmEnv] = [
            ScrollSwarmEnv(**self._env_kwargs, render_mode=None)
            for _ in range(self.num_envs)
        ]
        logger.info(
            "Initialized SyncVectorScrollSwarm (num_envs=%d, num_agents=%d, backend=%s)",
            self.num_envs,
            self.num_agents,
            self._env_kwargs["backend"],
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Cannot reset a closed SyncVectorScrollSwarm.")
        seeds = self._initial_seeds(seed)
        results = [_reset_single_env(env, s) for env, s in zip(self.envs, seeds)]
        return _collate_reset_results(results, self.num_envs, self.num_agents)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Cannot step a closed SyncVectorScrollSwarm.")
        action_arr = self._validate_actions(actions)
        autoreset_seeds = self._next_autoreset_seeds()
        results = [
            _step_and_autoreset(env, action_arr[e], autoreset_seeds[e])
            for e, env in enumerate(self.envs)
        ]
        return _collate_step_results(results, self.num_envs, self.num_agents)

    def close(self) -> None:
        if not self._closed:
            for env in self.envs:
                env.close()
            self._closed = True


class AsyncVectorScrollSwarm(BaseVectorScrollSwarm):
    """
    Multiprocess parallel vectorized runner for ``ScrollSwarmEnv``.

    Distributes ``num_envs`` independent corridors across ``num_workers`` CPU
    processes (using the cross-platform ``spawn`` context) so geometry kernels
    execute concurrently across physical CPU cores.
    """

    def __init__(
        self,
        num_envs: int,
        num_workers: Optional[int] = None,
        **env_kwargs: Any,
    ):
        super().__init__(num_envs=num_envs, **env_kwargs)
        cpu_count = os.cpu_count() or 1
        requested_workers = (
            min(self.num_envs, cpu_count) if num_workers is None else int(num_workers)
        )
        if requested_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {requested_workers}")
        self.num_workers = min(self.num_envs, requested_workers)

        # Partition num_envs evenly across num_workers
        base, rem = divmod(self.num_envs, self.num_workers)
        self._worker_slices: List[slice] = []
        cursor = 0
        for w in range(self.num_workers):
            count = base + (1 if w < rem else 0)
            self._worker_slices.append(slice(cursor, cursor + count))
            cursor += count

        ctx = mp.get_context("spawn")
        self._parent_conns: List[Any] = []
        self._workers: List[Any] = []
        for sl in self._worker_slices:
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_loop,
                args=(child_conn, sl.stop - sl.start, self._env_kwargs),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._parent_conns.append(parent_conn)
            self._workers.append(proc)

        logger.info(
            "Initialized AsyncVectorScrollSwarm (num_envs=%d, num_workers=%d, num_agents=%d, backend=%s)",
            self.num_envs,
            self.num_workers,
            self.num_agents,
            self._env_kwargs["backend"],
        )

    def _gather_worker_results(self) -> List[Tuple[Any, ...]]:
        """Collects and flattens ordered results from all worker pipes."""
        merged: List[Tuple[Any, ...]] = []
        for conn in self._parent_conns:
            status, payload = conn.recv()
            if status == "error":
                raise RuntimeError(f"AsyncVectorScrollSwarm worker failed: {payload}")
            merged.extend(payload)
        return merged

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Cannot reset a closed AsyncVectorScrollSwarm.")
        seeds = self._initial_seeds(seed)
        for conn, sl in zip(self._parent_conns, self._worker_slices):
            conn.send(("reset", seeds[sl]))
        results = self._gather_worker_results()
        return _collate_reset_results(results, self.num_envs, self.num_agents)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Cannot step a closed AsyncVectorScrollSwarm.")
        action_arr = self._validate_actions(actions)
        autoreset_seeds = self._next_autoreset_seeds()
        for conn, sl in zip(self._parent_conns, self._worker_slices):
            conn.send(("step", (action_arr[sl], autoreset_seeds[sl])))
        results = self._gather_worker_results()
        return _collate_step_results(results, self.num_envs, self.num_agents)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for conn in self._parent_conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for proc in self._workers:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        for conn in self._parent_conns:
            try:
                conn.close()
            except OSError:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
