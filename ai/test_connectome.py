"""Unit and learner-integration tests for the cached sparse controller."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from ai.config import SACConfig
from ai.connectome.acquire import load_neuprint_token, preprocess_selected_network
from ai.connectome.graph import ConnectomeGraph, GraphArtifactError
from ai.connectome.policy import FlyConnectomePolicy, MOTOR_CHANNELS
from ai.sac import SACAgent, TensorBatch


def _selection() -> dict:
    return {
        "nodes": {"body_ids": list(range(100, 108))},
        "input_groups": {"sensory": {"body_ids": [100, 101]}},
        "motor_groups": {name: {"body_ids": [102 + index]} for index, name in enumerate(MOTOR_CHANNELS)},
    }


def _graph(tmp_path) -> ConnectomeGraph:
    nodes = pd.DataFrame({"bodyId": list(range(100, 108)), "type": ["fixture"] * 8, "somaSide": ["L", "R"] * 4})
    edges = pd.DataFrame({
        "bodyId_pre": [100, 101, 102, 103, 104, 105, 106, 107, 107],
        "bodyId_post": [102, 103, 104, 105, 106, 107, 100, 101, 101],
        "weight": [2, 4, 3, 6, 5, 1, 2, 1, 3],
    })
    return preprocess_selected_network(nodes, edges, _selection(), tmp_path, max_neurons=8, max_edges=20, provenance={"fixture": True})


def test_token_loading_requires_environment_and_never_accepts_placeholder() -> None:
    with pytest.raises(RuntimeError, match="NEUPRINT_TOKEN"):
        load_neuprint_token({})
    assert load_neuprint_token({"NEUPRINT_TOKEN": "private-value"}) == "private-value"


def test_preprocessing_preserves_direction_mapping_and_cache(tmp_path) -> None:
    graph = _graph(tmp_path)
    assert graph.body_id_to_index[100] == 0
    assert graph.edge_index[:, 0].tolist() == [0, 2]  # body 100 -> body 102, never reversed.
    loaded = ConnectomeGraph.load(tmp_path / "connectome_graph.npz")
    assert loaded.body_id_to_index == graph.body_id_to_index
    assert set(loaded.motor_groups) == set(MOTOR_CHANNELS)
    edges = pd.read_parquet(tmp_path / "edges.parquet")
    assert {"bodyId_pre", "bodyId_post", "weight", "normalized_weight"}.issubset(edges.columns)
    assert (edges.groupby("bodyId_post")["normalized_weight"].sum() <= 1.00001).all()


def test_corrupted_graph_file_fails_clearly(tmp_path) -> None:
    path = tmp_path / "connectome_graph.npz"
    path.write_text("not an npz")
    (tmp_path / "connectome_metadata.json").write_text("{}")
    with pytest.raises(GraphArtifactError, match="could not load"):
        ConnectomeGraph.load(path)


def test_missing_or_empty_neuron_groups_are_rejected(tmp_path) -> None:
    graph = _graph(tmp_path)
    with pytest.raises(GraphArtifactError, match="motor groups"):
        FlyConnectomePolicy(3, 3, ConnectomeGraph(graph.body_ids, graph.edge_index, graph.weights, input_groups=graph.input_groups))
    with pytest.raises(GraphArtifactError, match="empty"):
        ConnectomeGraph(graph.body_ids, graph.edge_index, graph.weights, motor_groups=graph.motor_groups, input_groups={"sensory": ()})


def test_sparse_message_passing_is_batched_and_deterministic(tmp_path) -> None:
    graph = _graph(tmp_path)
    torch.manual_seed(17)
    policy = FlyConnectomePolicy(3, 3, graph, hidden_dim=8, propagation_steps=3)
    observations = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    first, _ = policy.sample(observations, deterministic=True)
    second, _ = policy.sample(observations, deterministic=True)
    assert policy.adjacency.is_sparse
    assert first.shape == (2, 3)
    assert torch.allclose(first, second)
    assert torch.isfinite(first).all()


def test_six_channel_antagonistic_decoder_dead_zone_and_bounds(tmp_path) -> None:
    policy = FlyConnectomePolicy(3, 3, _graph(tmp_path), hidden_dim=8, action_dead_zone=0.1, max_horizontal_speed=0.2, max_vertical_speed=0.3)
    neutral = policy.decode_motor_channels(torch.zeros((1, 6)))
    assert torch.equal(neutral, torch.zeros((1, 3)))
    channels = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 0.0]])
    action = policy.decode_motor_channels(channels)
    assert action[0, 0] > 0 and action[0, 1] < 0 and action[0, 2] > 0
    assert torch.all(action.abs() <= torch.tensor([0.2, 0.3, 1.0]))


def test_graph_actor_trains_one_sac_step_without_neuprint(tmp_path) -> None:
    graph = _graph(tmp_path)
    agent = SACAgent(SACConfig(state_dim=3, action_dim=3, hidden_dim=8, controller_type="fly_connectome", graph_path=str(tmp_path / "connectome_graph.npz"), seed=3))
    batch = TensorBatch(
        states=torch.zeros((4, 3)), actions=torch.zeros((4, 3)), rewards=torch.ones((4, 1)), next_states=torch.ones((4, 3)), dones=torch.zeros((4, 1)),
    )
    metrics = agent.update(batch)
    assert {"actor_loss", "alpha_loss", "entropy", "critic_one_loss", "critic_two_loss", "critic_one_q", "critic_two_q", "alpha", "actor_log_std_horizontal", "actor_log_std_vertical", "actor_log_std_gripper"}.issubset(metrics)
    assert all(np.isfinite(value) for value in metrics.values())


def test_random_graph_preserves_node_and_edge_counts(tmp_path) -> None:
    graph = _graph(tmp_path)
    random_graph = graph.randomized(seed=9)
    assert random_graph.node_count == graph.node_count
    assert random_graph.edge_count == graph.edge_count
    assert np.array_equal(np.bincount(random_graph.edge_index[0], minlength=8), np.bincount(graph.edge_index[0], minlength=8))
    assert np.array_equal(np.bincount(random_graph.edge_index[1], minlength=8), np.bincount(graph.edge_index[1], minlength=8))
