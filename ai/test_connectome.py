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


def test_dual_loop_actor_starts_neutral_and_changes_only_with_policy_signal(tmp_path) -> None:
    policy = FlyConnectomePolicy(
        4,
        3,
        _graph(tmp_path),
        hidden_dim=8,
        residual_alpha=(0.2, 0.15, 0.25),
        tactile_observation_indices=(0, 1, 2, 3),
    )
    observations = torch.zeros((2, 4))
    final, fly_base, residual, _ = policy.sample_decomposed(observations, deterministic=True)

    assert torch.allclose(fly_base, torch.zeros_like(fly_base), atol=1e-6)
    assert torch.allclose(final, torch.zeros_like(final), atol=1e-6)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)


def test_dual_loop_actor_exposes_exact_base_residual_composition(tmp_path) -> None:
    policy = FlyConnectomePolicy(
        4,
        3,
        _graph(tmp_path),
        hidden_dim=8,
        residual_alpha=(0.2, 0.15, 0.25),
        tactile_observation_indices=(0, 1, 2, 3),
    )
    observations = torch.tensor([[0.2, -0.3, 0.4, -0.5], [-0.1, 0.6, -0.2, 0.3]])
    final, fly_base, residual, log_probability = policy.sample_decomposed(observations, deterministic=True)

    expected = torch.clamp(fly_base + torch.tensor([0.2, 0.15, 0.25]) * residual, -1.0, 1.0)
    assert final.shape == fly_base.shape == residual.shape == (2, 3)
    assert log_probability.shape == (2, 1)
    assert torch.allclose(final, expected)
    assert torch.isfinite(final).all()

def test_connectome_base_head_is_nominal_and_geometry_independent(tmp_path) -> None:
    policy = FlyConnectomePolicy(
        30,
        3,
        _graph(tmp_path),
        hidden_dim=8,
        residual_alpha=(0.0, 0.0, 0.0),
        tactile_observation_indices=(17, 16, 18, 26),
    )
    with torch.no_grad():
        policy.base_head[0].weight.zero_()
        policy.base_head[0].bias.copy_(torch.tensor([0.25, -0.40, 0.60]))
    observations = torch.zeros((2, 30), dtype=torch.float32)
    observations[0, 10] = 0.80
    observations[1, 10] = -0.80

    _, fly_base, _, _ = policy.sample_decomposed(observations, deterministic=True)

    expected = torch.tanh(torch.tensor([0.25, -0.40, 0.60]))
    assert torch.allclose(fly_base[0], expected)
    assert torch.allclose(fly_base[1], expected)


def test_fly_base_tracks_object_with_signed_closed_loop_reflex(tmp_path) -> None:
    policy = FlyConnectomePolicy(
        30,
        3,
        _graph(tmp_path),
        hidden_dim=8,
        base_policy="closed_loop",
        residual_alpha=(0.0, 0.0, 0.0),
        tactile_observation_indices=(17, 16, 18, 26),
    )
    observations = torch.zeros((3, 30), dtype=torch.float32)
    observations[0, 10] = 0.4
    observations[0, 11] = 0.5
    observations[0, 20] = 1.0
    observations[1, 10] = -0.4
    observations[1, 11] = 0.5
    observations[1, 20] = 1.0
    observations[2, 10] = 0.0
    observations[2, 11] = 0.5
    observations[2, 20] = 1.0

    _, fly_base, _, _ = policy.sample_decomposed(observations, deterministic=True)

    assert fly_base[0, 0] < 0
    assert fly_base[1, 0] > 0
    assert fly_base[2, 0] == pytest.approx(0.0, abs=1e-6)
    assert fly_base[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert fly_base[1, 1] == pytest.approx(0.0, abs=1e-6)
    assert fly_base[2, 1] < 0
    assert fly_base[2, 2] > 0


def test_fly_base_is_pure_per_step_closed_loop_reflex(tmp_path) -> None:
    policy = FlyConnectomePolicy(
        30,
        3,
        _graph(tmp_path),
        hidden_dim=8,
        base_policy="closed_loop",
        residual_alpha=(0.0, 0.0, 0.0),
        tactile_observation_indices=(17, 16, 18, 26),
    )
    observations = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)

    _, fly_base, _, _ = policy.sample_decomposed(observations, deterministic=True)

    assert fly_base[0, 0] < 0.0
    assert fly_base[1, 0] > 0.0
    assert fly_base[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert fly_base[1, 1] == pytest.approx(0.0, abs=1e-6)


def test_fly_base_raises_lift_and_transport_after_attachment(tmp_path) -> None:
    policy = FlyConnectomePolicy(
        30,
        3,
        _graph(tmp_path),
        hidden_dim=8,
        base_policy="closed_loop",
        residual_alpha=(0.0, 0.0, 0.0),
        tactile_observation_indices=(17, 16, 18, 26),
    )
    attached_low = torch.zeros((1, 30), dtype=torch.float32)
    attached_low[:, 1] = 0.20
    attached_low[:, 12] = 0.10
    attached_low[:, 15] = 1.0

    attached_high = torch.zeros((1, 30), dtype=torch.float32)
    attached_high[:, 1] = 0.80
    attached_high[:, 12] = 0.25
    attached_high[:, 15] = 1.0

    _, low_base, _, _ = policy.sample_decomposed(attached_low, deterministic=True)
    _, high_base, _, _ = policy.sample_decomposed(attached_high, deterministic=True)

    assert low_base[0, 1] > 0.0
    assert high_base[0, 0] > 0.0
    assert high_base[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert high_base[0, 2] > 0.8


def test_fly_base_descends_and_releases_near_target(tmp_path) -> None:
    policy = FlyConnectomePolicy(
        30,
        3,
        _graph(tmp_path),
        hidden_dim=8,
        base_policy="closed_loop",
        residual_alpha=(0.0, 0.0, 0.0),
        tactile_observation_indices=(17, 16, 18, 26),
    )
    ready_to_lower = torch.zeros((1, 30), dtype=torch.float32)
    ready_to_lower[:, 1] = 0.70
    ready_to_lower[:, 12] = 0.02
    ready_to_lower[:, 15] = 1.0

    ready_to_release = torch.zeros((1, 30), dtype=torch.float32)
    ready_to_release[:, 1] = 0.15
    ready_to_release[:, 12] = 0.02
    ready_to_release[:, 15] = 1.0

    _, lower_base, _, _ = policy.sample_decomposed(ready_to_lower, deterministic=True)
    _, release_base, _, _ = policy.sample_decomposed(ready_to_release, deterministic=True)

    assert lower_base[0, 1] < 0.0
    assert lower_base[0, 2] > 0.8
    assert release_base[0, 2] < 0.0
    assert release_base[0, 1] == pytest.approx(-0.3, abs=1e-6)


def test_fly_base_uses_phase_state_for_release_gate(tmp_path) -> None:
    policy = FlyConnectomePolicy(
        30,
        3,
        _graph(tmp_path),
        hidden_dim=8,
        base_policy="closed_loop",
        residual_alpha=(0.0, 0.15, 0.20),
        tactile_observation_indices=(17, 16, 18, 26),
    )
    with torch.no_grad():
        policy.residual_head[-1].bias.copy_(torch.tensor([10.0, 10.0, 10.0]))
    phase_release_high = torch.zeros((1, 30), dtype=torch.float32)
    phase_release_high[:, 1] = 0.70
    phase_release_high[:, 12] = 0.02
    phase_release_high[:, 15] = -1.0  # released; still settling in PhaseReleaseObject
    phase_release_high[:, 19] = 7.0 / 4.5 - 1.0

    final, release_base, _, _ = policy.sample_decomposed(phase_release_high, deterministic=True)

    assert release_base[0, 2] < 0.0
    assert release_base[0, 1] == pytest.approx(-0.3, abs=1e-6)
    assert torch.allclose(final[0, 0], release_base[0, 0])
    assert final[0, 1].item() < 0.0


def test_tactile_slip_changes_residual_grip_without_changing_fly_base(tmp_path) -> None:
    policy = FlyConnectomePolicy(
        4,
        3,
        _graph(tmp_path),
        hidden_dim=8,
        tactile_observation_indices=(0, 1, 2, 3),
    )
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        # Route only tactile[slipSeverity] through one residual hidden unit to
        # the grip output. The connectome/base latent stays identically zero.
        policy.residual_head[0].weight[0, 8] = 1.0
        policy.residual_head[2].weight[2, 0] = 1.0
    observations = torch.zeros((2, 4))
    observations[1, 0] = 1.0

    final, fly_base, residual, _ = policy.sample_decomposed(observations, deterministic=True)

    assert torch.allclose(fly_base[0], fly_base[1])
    assert residual[1, 2] > residual[0, 2]
    assert final[1, 2] > final[0, 2]


def test_phase_gated_decoder_uses_transport_head_only_after_attachment(tmp_path) -> None:
    """The graph state is shared; the selected decoder changes by task state."""
    policy = FlyConnectomePolicy(
        30,
        3,
        _graph(tmp_path),
        hidden_dim=8,
        phase_gated_decoder=True,
        object_attached_observation_index=15,
        phase_observation_index=19,
        transport_phase_threshold=0.0,
    )
    with torch.no_grad():
        policy.motor_gain.fill_(1.0)
        policy.motor_bias.zero_()
        policy.transport_motor_gain.fill_(1.0)
        policy.transport_motor_bias.copy_(torch.tensor([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    channels = torch.zeros((2, 6))
    observations = torch.zeros((2, 30))
    observations[0, 15] = -1.0  # detached: acquisition decoder
    observations[0, 19] = -1.0
    observations[1, 15] = 1.0   # attached and MoveToTarget: transport decoder
    observations[1, 19] = 0.0

    selected = policy._apply_motor_decoder_heads(channels, observations)
    actions = policy.decode_motor_channels(selected)
    assert torch.allclose(actions[0], torch.zeros(3))
    assert actions[1, 0] > 0


def test_graph_actor_trains_one_sac_step_without_neuprint(tmp_path) -> None:
    graph = _graph(tmp_path)
    agent = SACAgent(SACConfig(state_dim=30, action_dim=3, hidden_dim=8, controller_type="fly_connectome", graph_path=str(tmp_path / "connectome_graph.npz"), seed=3))
    batch = TensorBatch(
        states=torch.zeros((4, 30)), actions=torch.zeros((4, 3)), rewards=torch.ones((4, 1)), next_states=torch.ones((4, 30)), dones=torch.zeros((4, 1)),
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


def test_connectome_decoder_only_training_unlocks_after_stability_window(tmp_path) -> None:
    graph = _graph(tmp_path)
    config = SACConfig(
        state_dim=3,
        action_dim=3,
        hidden_dim=8,
        controller_type="fly_connectome",
        graph_path=str(tmp_path / "connectome_graph.npz"),
        seed=7,
        train_edge_gains=False,
        freeze_topology=True,
        full_actor_unlock_step=32,
        base_warmup_steps=32,
        phase_gated_decoder=True,
        object_attached_observation_index=1,
        phase_observation_index=2,
    )
    agent = SACAgent(config)
    agent.training_step = 0
    agent._configure_actor_trainability()
    assert not agent.actor.neuron_bias.requires_grad
    assert not agent.actor.leak_logit.requires_grad
    assert not agent.actor.edge_log_gains.requires_grad
    assert agent.actor.sensory_encoder[0].weight.requires_grad
    assert agent.actor.motor_gain.requires_grad
    assert agent.actor.motor_bias.requires_grad
    assert agent.actor.transport_motor_gain.requires_grad
    assert agent.actor.transport_motor_bias.requires_grad
    assert not agent.actor.log_std.requires_grad
    assert all(parameter.requires_grad for parameter in agent.actor.base_head.parameters())
    assert not any(parameter.requires_grad for parameter in agent.actor.residual_head.parameters())
    learning_rates = sorted(group["lr"] for group in agent.actor_optimizer.param_groups)
    assert learning_rates == pytest.approx([config.learning_rate, config.learning_rate])

    agent.training_step = 64
    agent._configure_actor_trainability()
    assert agent.actor.neuron_bias.requires_grad
    assert agent.actor.leak_logit.requires_grad
    assert agent.actor.edge_log_gains.requires_grad
    assert agent.actor.sensory_encoder[0].weight.requires_grad
    assert agent.actor.log_std.requires_grad
    assert all(parameter.requires_grad for parameter in agent.actor.residual_head.parameters())
    learning_rates = sorted(group["lr"] for group in agent.actor_optimizer.param_groups)
    assert learning_rates == pytest.approx([config.learning_rate * 0.1, config.learning_rate])
