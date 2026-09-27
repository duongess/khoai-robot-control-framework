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
    assert torch.all(action[:, 1] < 0)  # both grippers are above the object
    assert torch.allclose(action, reflex)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.all(policy.parameter_log_std > policy.min_log_std)
