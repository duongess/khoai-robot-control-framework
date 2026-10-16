import torch

from ai.config import SACConfig
from ai.sac import SACAgent


def test_heuristic_guided_uses_scripted_phase_base_and_bounded_sac_residual() -> None:
    agent = SACAgent(SACConfig(
        state_dim=30,
        action_dim=3,
        hidden_dim=8,
        controller_type="heuristic_guided",
        base_policy="scripted_phases",
    ))
    state = torch.zeros((1, 30))
    state[:, 10] = -0.2  # Object is right of the carriage.
    state[:, 20] = 1.0   # Contact closes the gripper in acquire phase.
    final, base, residual = agent.act_with_actor_components(agent.actor, state)

    assert base[0, 0] > 0
    assert base[0, 2] == 0.9
    assert torch.all(residual.abs() <= 1)
    assert torch.all(final.abs() <= 1)
