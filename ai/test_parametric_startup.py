"""A new graph learner publishes Parametric SAC before its first prediction."""

from __future__ import annotations

import pytest
import torch

from ai.config import LearnerConfig
from ai.grpc_server import LearnerServicer
from ai.test_connectome import _graph
from ai.test_grpc_server import AbortContext
from gen.python.learner.v1 import environment_pb2, learner_pb2


def test_graph_learner_boot_registers_default_gantry_reflex_before_snapshot(tmp_path) -> None:
    _graph(tmp_path)
    servicer = LearnerServicer(LearnerConfig(
        state_dim=30,
        action_dim=3,
        controller_type="fly_connectome",
        graph_path=str(tmp_path / "connectome_graph.npz"),
        full_actor_unlock_step=0,
        base_warmup_steps=0,
    ))
    actor = servicer._agent.actor
    snapshot = servicer._actor_snapshots[1]

    assert actor.reflex_parameter_count == 6
    assert snapshot.reflex_parameter_count == 6
    assert snapshot.get_extra_state()["reflex_config"] is not None
    assert len(snapshot.get_extra_state()["reflex_config"]["channels"]) == 3

    state = torch.zeros((1, 30))
    state[:, 10] = -0.01  # 6 cm: carriage aligned within the 15 cm descent gate
    state[:, 11] = 0.5   # gripper above object: drive down
    action, reflex, residual, _ = snapshot.sample_decomposed(state, deterministic=True)
    assert action[0, 0] > 0
    assert action[0, 1] < 0
    assert torch.allclose(action, reflex)
    assert torch.equal(residual, torch.zeros_like(residual))

    response = servicer.PredictBatch(
        learner_pb2.PredictBatchRequest(states=[environment_pb2.State(values=state[0].tolist())]),
        AbortContext(),
    )
    assert list(response.actions[0].values) == pytest.approx(list(response.fly_base_actions[0].values))
    assert list(response.residual_actions[0].values) == [0.0, 0.0, 0.0]


def test_dense_parametric_learner_forwards_all_six_coefficients_without_zero_fallback() -> None:
    servicer = LearnerServicer(LearnerConfig(
        state_dim=30,
        action_dim=3,
        controller_type="parametric_mlp",
        deterministic_inference=True,
    ))
    actor = servicer._agent.actor
    snapshot = servicer._actor_snapshots[1]

    assert actor.reflex_parameter_count == 6
    assert snapshot.reflex_parameter_count == 6
    assert snapshot.parameter_head.out_features == 6

    # Remove state variation so the registered defaults reveal the exact theta
    # ordering: [a_x, b_x, a_y, b_y, a_g, b_g]. None may be synthesized as zero.
    with torch.no_grad():
        snapshot.parameter_head.weight.zero_()
    state = torch.zeros((1, 30))
    state[:, 10] = -0.01
    state[:, 11] = 0.5
    state[:, 20] = 1.0
    parameters = snapshot.bounded_reflex_parameters(state)
    theta = torch.stack([
        parameters["channel_0.a"], parameters["channel_0.b"],
        parameters["channel_1.a"], parameters["channel_1.b"],
        parameters["channel_2.a"], parameters["channel_2.b"],
    ], dim=1)
    action, fly_base, pure_rl, _ = snapshot.sample_decomposed(state, deterministic=True)

    assert theta.shape == (1, 6)
    assert theta[0, 2] > 0 and theta[0, 3] < 0
    assert theta[0, 4] > 0 and theta[0, 5] > 0
    assert action[0, 1] < 0 and action[0, 2] > 0
    assert torch.equal(fly_base, torch.zeros_like(fly_base))
    assert torch.allclose(pure_rl, action)

    response = servicer.PredictBatch(
        learner_pb2.PredictBatchRequest(states=[environment_pb2.State(values=state[0].tolist())]),
        AbortContext(),
    )
    assert list(response.fly_base_actions[0].values) == [0.0, 0.0, 0.0]
    assert response.residual_actions[0].values[1] < 0
    assert response.residual_actions[0].values[2] > 0
    assert list(response.actions[0].values) == pytest.approx(list(response.residual_actions[0].values))
