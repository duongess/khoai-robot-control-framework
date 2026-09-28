"""Regression tests for the canonical gantry Parametric-SAC law."""

from __future__ import annotations

import torch

from ai.connectome.dynamic_policy import DEFAULT_GANTRY_REFLEX_CONFIG
from ai.connectome.policy import FlyConnectomePolicy
from ai.test_connectome import _graph


def test_default_gantry_law_uses_error_features_and_varies_theta_by_state(tmp_path) -> None:
    torch.manual_seed(31)
    policy = FlyConnectomePolicy(30, 3, _graph(tmp_path), hidden_dim=8)
    registered = policy.register_default_gantry_reflex_law()
    assert registered["channels"][0]["signal_indices"] == {"x": 10}
    assert registered["channels"][1]["signal_indices"] == {"x": 11}

    # Coordinate errors belong at 10/11; raw position columns 0/1 are held
    # fixed so only the authoritative signed-error mapping can affect action.
    observations = torch.zeros((2, 30))
    observations[:, 0:2] = 0.9
    observations[0, 10], observations[1, 10] = 0.40, -0.40
    observations[0, 11], observations[1, 11] = 0.50, 0.20
    parameters = policy.bounded_reflex_parameters(observations)
    action, reflex, residual, _ = policy.sample_decomposed(observations, deterministic=True)

    assert not torch.allclose(parameters["channel_0.a"][0], parameters["channel_0.a"][1])
    assert action[0, 0] < 0 and action[1, 0] > 0
    assert torch.allclose(action[:, 1], torch.zeros_like(action[:, 1]))  # far from object: hold transit height
    assert torch.allclose(action, reflex)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.all(policy.parameter_log_std > policy.min_log_std)


def test_default_grip_law_closes_on_contact(tmp_path) -> None:
    """Grip is positive reinforcement: contact must increase clamp force."""
    policy = FlyConnectomePolicy(30, 3, _graph(tmp_path), hidden_dim=8)
    registered = policy.register_default_gantry_reflex_law()
    grip = registered["channels"][2]
    assert grip["expression_type"] == "positive_linear"
    assert grip["signal_indices"] == {"x": 20}

    # Eliminate state-dependent variation so this test isolates the law itself.
    with torch.no_grad():
        policy.parameter_head.weight.zero_()
    state = torch.zeros((2, 30))
    state[1, 20] = 1.0
    action, _, _, _ = policy.sample_decomposed(state, deterministic=True)

    assert action[0, 2] > 0  # positive holding baseline b
    assert action[1, 2] > action[0, 2]  # contact adds +a, never subtracts force


def test_default_gantry_task_flow_lifts_carries_and_releases(tmp_path) -> None:
    policy = FlyConnectomePolicy(30, 3, _graph(tmp_path), hidden_dim=8)
    policy.register_default_gantry_reflex_law()
    with torch.no_grad():
        policy.parameter_head.weight.zero_()

    state = torch.zeros((4, 30))
    # Attached below carry height: lift while preserving clamp force.
    state[0, 15], state[0, 1] = 1.0, 0.20
    # Attached at carry height: track the target horizontally and hold height.
    state[1, 15], state[1, 1], state[1, 12] = 1.0, 0.70, 0.25
    # Environment PhaseLowerAtTarget = 6.
    state[2, 15], state[2, 1], state[2, 12], state[2, 19] = 1.0, 0.70, 0.20, 6.0 / 4.5 - 1.0
    # Environment PhaseReleaseObject = 7; release can follow detach.
    state[3, 1], state[3, 19] = 0.20, 7.0 / 4.5 - 1.0

    action, _, _, _ = policy.sample_decomposed(state, deterministic=True)

    assert action[0, 1] > 0 and action[0, 2] > 0
    assert action[1, 0] > 0 and action[1, 1] < 0 and action[1, 2] > 0
    assert action[2, 0] > 0 and action[2, 1] <= -0.5 and action[2, 2] > 0
    assert action[3, 1] < 0 and action[3, 2] < 0


def test_phase_routing_keeps_action_on_engine_gradient_path(tmp_path) -> None:
    policy = FlyConnectomePolicy(30, 3, _graph(tmp_path), hidden_dim=8)
    policy.register_default_gantry_reflex_law()
    state = torch.zeros((1, 30), requires_grad=False)
    state[:, 15], state[:, 1], state[:, 12] = 1.0, 0.70, 0.10
    action, _, _, _ = policy.sample_decomposed(state, deterministic=True)
    assert action[0, 1].abs() > 1e-5  # transport is routed, not hard-coded to zero
    action.sum().backward()
    assert policy.parameter_head.weight.grad is not None
    assert policy.parameter_head.weight.grad.abs().sum() > 0


def test_approach_gate_descends_only_when_horizontally_aligned(tmp_path) -> None:
    policy = FlyConnectomePolicy(30, 3, _graph(tmp_path), hidden_dim=8)
    policy.register_default_gantry_reflex_law()
    state = torch.zeros((2, 30))
    state[0, 10], state[0, 11] = -0.40, 0.50  # far left of object
    state[1, 10], state[1, 11] = -0.05, 0.50  # within the 10cm gate

    action, _, _, _ = policy.sample_decomposed(state, deterministic=True)

    assert action[0, 0] > 0 and torch.allclose(action[0, 1], torch.tensor(0.0))
    assert action[1, 0] > 0 and action[1, 1] < 0
