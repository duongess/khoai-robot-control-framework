"""Safety priors and signed-error behaviour for Parametric SAC tracking."""

from __future__ import annotations

import pytest
import torch

from ai.connectome.policy import FlyConnectomePolicy
from ai.test_connectome import _graph


def _parameter(name: str, minimum: float, maximum: float, default: float) -> dict:
    return {"name": name, "min_val": minimum, "max_val": maximum, "default": default}


def test_signed_tracking_moves_toward_object_and_seeds_vertical_descent(tmp_path) -> None:
    policy = FlyConnectomePolicy(3, 3, _graph(tmp_path), hidden_dim=8)
    policy.register_reflex_law({"channels": [
        {
            "name": "horizontal",
            "action_index": 0,
            "expression_type": "linear",
            "signal_indices": {"x": 0},
            "parameters": [_parameter("a", 0.1, 2.0, 1.0), _parameter("b", -0.2, 0.2, 0.0)],
        },
        {
            "name": "vertical",
            "action_index": 1,
            "expression_type": "linear",
            "signal_indices": {"x": 1},
            # The submitted default is deliberately not trusted: registration
            # replaces it with a safe descent prior inside this interval.
            "parameters": [_parameter("a", 0.1, 2.0, 1.0), _parameter("b", -0.8, -0.3, -0.3)],
        },
    ]})
    observation = torch.tensor([[0.4, 0.5, 0.0]])
    action, reflex, residual, _ = policy.sample_decomposed(observation, deterministic=True)
    parameters = policy.bounded_reflex_parameters(observation)

    assert action[0, 0].item() < 0.0  # CarriageX > ObjectX: move left.
    assert action[0, 1].item() < 0.0  # Gripper above object: move down.
    assert reflex[0, 1].item() < 0.0
    assert torch.equal(residual, torch.zeros_like(residual))
    assert -0.8 <= parameters["vertical.b"].item() <= -0.3
    assert torch.all(policy.parameter_log_std > policy.min_log_std)


def test_vertical_tracking_rejects_a_bias_range_without_descent_prior(tmp_path) -> None:
    policy = FlyConnectomePolicy(3, 3, _graph(tmp_path), hidden_dim=8)
    with pytest.raises(ValueError, match="safe descent prior"):
        policy.register_reflex_law({"channels": [{
            "action_index": 1,
            "expression_type": "linear",
            "parameters": [_parameter("a", 0.1, 2.0, 1.0), _parameter("b", 0.0, 1.0, 0.5)],
        }]})
