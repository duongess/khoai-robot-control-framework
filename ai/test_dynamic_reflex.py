"""Tests for web-configured, differentiable reflex laws."""

from __future__ import annotations

import pytest
import torch

from ai.connectome.dynamic_reflex import DynamicReflexEngine, ReflexConfigError, ReflexFunctionConfig
from ai.connectome.policy import FlyConnectomePolicy
from ai.test_connectome import _graph


def _parameter(name: str, minimum: float, maximum: float, default: float = 1.0) -> dict:
    return {"name": name, "min_val": minimum, "max_val": maximum, "default": default}


def test_registered_linear_law_resizes_head_bounds_parameters_and_clamps_action(tmp_path) -> None:
    policy = FlyConnectomePolicy(3, 3, _graph(tmp_path), hidden_dim=8)
    policy.register_reflex_law({"channels": [{
        "name": "horizontal",
        "action_index": 0,
        "expression_type": "linear",
        "signal_indices": {"x": 0},
        "parameters": [_parameter("a", 0.0, 2.0), _parameter("b", -2.0, 2.0, 0.0)],
    }]})
    assert policy.parameter_head.out_features == 2
    with torch.no_grad():
        policy.parameter_head.weight.zero_()
        policy.parameter_head.bias.copy_(torch.tensor([20.0, 20.0]))
    observation = torch.tensor([[4.0, 0.0, 0.0]])
    parameters = policy.bounded_reflex_parameters(observation)
    action, base, _, _ = policy.sample_decomposed(observation, deterministic=True)
    assert 1.99 < parameters["horizontal.a"].item() <= 2.0
    assert 1.99 < parameters["horizontal.b"].item() <= 2.0
    assert base[0, 0].item() == pytest.approx(-1.0)  # 2 * 4 + 2, then hardware clamp.
    assert action[0, 0].item() == pytest.approx(-1.0)
    assert policy.reflex_parameter_telemetry(observation)[0]["name"] == "horizontal.a"


def test_impedance_pd_is_vectorized_over_a_batch() -> None:
    config = ReflexFunctionConfig.from_dict({"channels": [{
        "expression_type": "impedance_pd",
        "parameters": [_parameter("kp", 0, 10), _parameter("kd", 0, 10), _parameter("bias", -2, 2, 0)],
    }]})
    output = DynamicReflexEngine(config).evaluate(
        {"err": torch.tensor([2.0, -1.0]), "vel": torch.tensor([0.5, 3.0])},
        {"0.kp": torch.tensor([3.0, 4.0]), "0.kd": torch.tensor([2.0, 1.0]), "0.bias": torch.tensor([1.0, -2.0])},
    )
    assert torch.allclose(output[:, 0], torch.tensor([-6.0, -1.0]))


def test_custom_json_formula_is_safe_differentiable_and_checkpointed(tmp_path) -> None:
    payload = {"channels": [{
        "name": "custom",
        "expression_type": "custom_eval",
        "formula_str": "a * x + b",
        "parameters": [_parameter("a", -2, 2), _parameter("b", -1, 1, 0)],
    }]}
    policy = FlyConnectomePolicy(3, 3, _graph(tmp_path), hidden_dim=8)
    policy.register_reflex_law(payload)
    state = policy.state_dict()
    restored = FlyConnectomePolicy(3, 3, _graph(tmp_path), hidden_dim=8)
    restored.load_state_dict(state)
    assert restored.get_extra_state()["reflex_config"] == ReflexFunctionConfig.from_dict(payload).to_dict()
    with pytest.raises(ReflexConfigError, match="unsafe formula syntax|only tanh"):
        policy.register_reflex_law({"channels": [{
            "expression_type": "custom_eval",
            "formula_str": "__import__('os').system('echo unsafe')",
            "parameters": [],
        }]})
    with pytest.raises(ReflexConfigError, match="unsafe formula syntax"):
        policy.register_reflex_law({"channels": [{
            "expression_type": "custom_eval",
            "formula_str": "x.__class__",
            "parameters": [],
        }]})
