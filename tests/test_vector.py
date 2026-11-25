"""
Unit and parity tests for SyncVectorScrollSwarm and AsyncVectorScrollSwarm.
"""

import numpy as np
import pytest

from scrollswarm import (
    BaseVectorScrollSwarm,
    SyncVectorScrollSwarm,
    AsyncVectorScrollSwarm,
)


def test_vector_validation_and_spaces():
    """Verifies constructor checks, Gymnasium space shapes, and post-close guards."""
    with pytest.raises(ValueError):
        SyncVectorScrollSwarm(num_envs=0, num_agents=3)
    with pytest.raises(ValueError):
        SyncVectorScrollSwarm(num_envs=2, num_agents=11)
    with pytest.raises(ValueError):
        AsyncVectorScrollSwarm(num_envs=2, num_agents=3, num_workers=0)

    with SyncVectorScrollSwarm(num_envs=3, num_agents=4, backend="numpy") as vec:
        assert vec.observation_space.shape == (3, 4, 16)
        assert vec.action_space.shape == (3, 4)
        assert vec.state_space.shape == (3, 64)
        assert vec.single_observation_space.shape == (16,)
        assert vec.single_action_space.n == 3

        sample_act = vec.action_space.sample()
        assert sample_act.shape == (3, 4)

        vec.reset(seed=42)
        with pytest.raises(ValueError):
            vec.step(np.zeros((3, 3), dtype=np.int64))

    with pytest.raises(RuntimeError):
        vec.reset()
    with pytest.raises(RuntimeError):
        vec.step(np.zeros((3, 4), dtype=np.int64))


def test_sync_vector_reset_and_step_shapes():
    """Verifies batched shapes, dtypes, and CTDE/graph tensor consistency."""
    with SyncVectorScrollSwarm(
        num_envs=4, num_agents=3, max_cycles=50, comm_topology="chain", backend="numpy"
    ) as vec:
        obs, infos = vec.reset(seed=7)
        assert obs.shape == (4, 3, 16)
        assert obs.dtype == np.float32
        assert infos["state"].shape == (4, 48)
        assert infos["alive_mask"].shape == (4, 3)
        assert np.all(infos["alive_mask"])
        assert infos["adj"].shape == (4, 3, 3)
        assert infos["edge_attr"].shape == (4, 3, 3, 3)
        assert len(infos["env_infos"]) == 4

        actions = np.ones((4, 3), dtype=np.int64)
        next_obs, rewards, terms, truncs, step_infos = vec.step(actions)
        assert next_obs.shape == (4, 3, 16)
        assert rewards.shape == (4, 3)
        assert rewards.dtype == np.float32
        assert terms.shape == (4, 3)
        assert terms.dtype == bool
        assert truncs.shape == (4, 3)
        assert truncs.dtype == bool
        assert step_infos["episode_done"].shape == (4,)
        assert not np.any(step_infos["episode_done"])


def test_vector_partial_death_and_autoreset():
    """Verifies partial-swarm crash masking, graph pruning, and seamless auto-reset."""
    with SyncVectorScrollSwarm(
        num_envs=2, num_agents=3, max_cycles=8, comm_topology="chain", backend="numpy"
    ) as vec:
        vec.reset(seed=10)

        # Force agent_0 in env 0 into top boundary while env 1 stays healthy
        env0 = vec.envs[0]
        a0 = env0._agent_map["agent_0"]
        a0.y = float(env0.scrolling_map.top_heights[int(a0.x)]) - 2.0

        obs, rewards, terms, truncs, infos = vec.step(np.ones((2, 3), dtype=np.int64))
        assert terms[0, 0] is np.bool_(True)
        assert not infos["alive_mask"][0, 0]
        assert infos["alive_mask"][0, 1]
        assert np.all(infos["alive_mask"][1])
        # Crashed agent's graph row/col and CTDE slice must be zeroed
        assert np.all(infos["adj"][0, 0, :] == 0.0)
        assert np.all(infos["adj"][0, :, 0] == 0.0)
        assert np.all(infos["state"][0, :16] == 0.0)
        assert not np.any(infos["episode_done"])

        # Step until max_cycles=8 triggers auto-reset in both corridors
        last_infos = infos
        for _ in range(7):
            obs, rewards, terms, truncs, last_infos = vec.step(
                np.ones((2, 3), dtype=np.int64)
            )

        assert np.all(truncs[:, 1])  # surviving agents truncated at step 8
        assert np.all(last_infos["episode_done"])
        assert np.all(last_infos["alive_mask"])  # auto-reset restores full living fleet
        assert vec.envs[0].step_count == 0
        assert vec.envs[1].step_count == 0
        assert np.any(last_infos["final_observation"][1] != 0.0)
        assert np.any(last_infos["final_state"][1] != 0.0)


def test_batch_graphs_block_diagonal_isolation():
    """Verifies block-diagonal PyG graph batching keeps corridors strictly isolated."""
    with SyncVectorScrollSwarm(
        num_envs=3, num_agents=4, comm_topology="ring", backend="numpy"
    ) as vec:
        _, infos = vec.reset(seed=42)
        # Disable agent 1 in corridor 1 and verify edge pruning persists in batched graph
        vec.envs[1]._agent_map["agent_1"].alive = False
        from scrollswarm.vector import _extract_dense_graph

        adj1, attr1 = _extract_dense_graph(vec.envs[1])
        infos["adj"][1] = adj1
        infos["edge_attr"][1] = attr1

        edge_index, edge_attr_flat, batch = BaseVectorScrollSwarm.batch_graphs(
            infos["adj"], infos["edge_attr"]
        )
        assert batch.shape == (12,)
        np.testing.assert_array_equal(
            batch, np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        )
        assert edge_index.shape[0] == 2
        assert edge_attr_flat.shape == (edge_index.shape[1], 3)

        # Every edge must connect nodes belonging to the same corridor
        src, tgt = edge_index[0], edge_index[1]
        assert np.all(batch[src] == batch[tgt])
        # Dead node index 1*4 + 1 = 5 must have zero incident edges
        assert not np.any(src == 5)
        assert not np.any(tgt == 5)


def test_async_matches_sync_deterministically():
    """Verifies AsyncVectorScrollSwarm matches SyncVectorScrollSwarm bit-exactly across auto-reset."""
    kwargs = dict(
        num_envs=4, num_agents=3, max_cycles=10, comm_topology="chain", backend="numpy"
    )

    with (
        SyncVectorScrollSwarm(**kwargs) as sync_vec,
        AsyncVectorScrollSwarm(**kwargs, num_workers=2) as async_vec,
    ):
        sync_obs, sync_info = sync_vec.reset(seed=123)
        async_obs, async_info = async_vec.reset(seed=123)

        np.testing.assert_array_equal(async_obs, sync_obs)
        np.testing.assert_array_equal(async_info["state"], sync_info["state"])
        np.testing.assert_array_equal(async_info["adj"], sync_info["adj"])

        for step_idx in range(15):  # Crosses max_cycles=10 auto-reset boundary
            actions = np.full((4, 3), step_idx % 3, dtype=np.int64)
            s_out = sync_vec.step(actions)
            a_out = async_vec.step(actions)

            np.testing.assert_array_equal(a_out[0], s_out[0])
            np.testing.assert_array_equal(a_out[1], s_out[1])
            np.testing.assert_array_equal(a_out[2], s_out[2])
            np.testing.assert_array_equal(a_out[3], s_out[3])
            for key in (
                "state",
                "alive_mask",
                "adj",
                "edge_attr",
                "episode_done",
                "final_observation",
                "final_state",
            ):
                np.testing.assert_array_equal(a_out[4][key], s_out[4][key])
