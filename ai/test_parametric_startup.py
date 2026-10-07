"""A new graph learner publishes Parametric SAC before its first prediction."""

from __future__ import annotations

import torch

from ai.config import LearnerConfig
from ai.grpc_server import LearnerServicer
from ai.test_connectome import _graph


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
