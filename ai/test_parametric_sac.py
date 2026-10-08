"""End-to-end assertions for SAC tuning registered reflex parameters only."""

from __future__ import annotations

import torch

from ai.config import SACConfig
from ai.connectome.dynamic_policy import DEFAULT_GANTRY_REFLEX_CONFIG
from ai.sac import SACAgent, TensorBatch
from ai.test_connectome import _graph


def _law() -> dict:
    return {"channels": [{
        "name": "lower_gripper",
        "action_index": 1,
        "expression_type": "linear",
        "signal_indices": {"x": 0},
        "parameters": [
            {"name": "a", "min_val": -2.0, "max_val": 2.0, "default": 1.0},
            {"name": "b", "min_val": -0.5, "max_val": 0.5, "default": 0.0},
        ],
    }]}


def _agent(tmp_path) -> SACAgent:
    _graph(tmp_path)
    agent = SACAgent(SACConfig(
        state_dim=3,
        action_dim=3,
        hidden_dim=8,
        controller_type="fly_connectome",
        graph_path=str(tmp_path / "connectome_graph.npz"),
        full_actor_unlock_step=0,
        base_warmup_steps=0,
    ))
    agent.register_reflex_law(_law())
    return agent


def test_parametric_sac_samples_parameters_not_a_motor_residual(tmp_path) -> None:
    agent = _agent(tmp_path)
    states = torch.tensor([[0.25, 0.0, 0.0], [-0.50, 0.0, 0.0]])
    final, reflex_action, residual = agent.act_with_actor_components(agent.actor, states, deterministic=False)

    assert agent.actor.reflex_parameter_count == 2
    assert agent.target_entropy == -2.0
    assert torch.allclose(final, reflex_action)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.all(final.abs() <= 1.0)
    optimizer_parameters = {id(parameter) for group in agent.actor_optimizer.param_groups for parameter in group["params"]}
    assert id(agent.actor.parameter_head.weight) in optimizer_parameters
    assert id(agent.actor.parameter_log_std) in optimizer_parameters


def test_parametric_sac_trains_and_restores_registered_parameter_head(tmp_path) -> None:
    agent = _agent(tmp_path)
    batch = TensorBatch(
        states=torch.zeros((4, 3)),
        actions=torch.zeros((4, 3)),
        rewards=torch.ones((4, 1)),
        next_states=torch.full((4, 3), 0.1),
        dones=torch.zeros((4, 1)),
    )
    metrics = agent.update(batch)
    assert torch.isfinite(torch.tensor(metrics["actor_loss"]))
    checkpoint = agent.checkpoint_state()

    restored = _agent(tmp_path)
    restored.load_checkpoint_state(checkpoint)
    assert restored.actor.reflex_parameter_count == 2
    assert restored.target_entropy == -2.0
    assert restored.actor.get_extra_state() == agent.actor.get_extra_state()


def test_dense_parametric_sac_trains_and_round_trips_all_six_coefficients() -> None:
    config = SACConfig(
        state_dim=30,
        action_dim=3,
        hidden_dim=16,
        controller_type="parametric_mlp",
        base_warmup_steps=0,
    )
    agent = SACAgent(config)
    agent.register_reflex_law(DEFAULT_GANTRY_REFLEX_CONFIG)
    batch = TensorBatch(
        states=torch.randn((8, 30)),
        actions=torch.tanh(torch.randn((8, 3))),
        rewards=torch.randn((8, 1)),
        next_states=torch.randn((8, 30)),
        dones=torch.zeros((8, 1)),
    )

    metrics = agent.update(batch)
    checkpoint = agent.checkpoint_state()
    restored = SACAgent(config)
    restored.register_reflex_law(DEFAULT_GANTRY_REFLEX_CONFIG)
    restored.load_checkpoint_state(checkpoint)

    states = torch.randn((2, 30))
    expected = agent.act_with_actor_components(agent.actor, states, deterministic=True)
    actual = restored.act_with_actor_components(restored.actor, states, deterministic=True)
    assert torch.isfinite(torch.tensor(metrics["actor_loss"]))
    assert agent.actor.reflex_parameter_count == restored.actor.reflex_parameter_count == 6
    assert all(torch.allclose(left, right) for left, right in zip(expected, actual, strict=True))
    assert torch.equal(actual[1], torch.zeros_like(actual[1]))
    assert actual[2].abs().sum() > 0
