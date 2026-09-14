import math

import torch

from ai.config import SACConfig
from ai.networks import GaussianActor
from ai.sac import SACAgent, TensorBatch


def test_gaussian_actor_returns_expected_tensor_shapes():
    actor = GaussianActor(state_dim=6, action_dim=3, hidden_dim=32)

    actions, log_probabilities = actor.sample(torch.randn(4, 6))

    assert actions.shape == (4, 3)
    assert log_probabilities.shape == (4, 1)


def test_gaussian_actor_actions_are_bounded():
    actor = GaussianActor(state_dim=3, action_dim=3, hidden_dim=32)

    actions, _ = actor.sample(torch.randn(256, 3))

    assert torch.all(actions >= -1)
    assert torch.all(actions <= 1)


def test_sac_agent_performs_one_training_update():
    agent = SACAgent(SACConfig(state_dim=3, action_dim=3, hidden_dim=32, seed=7))
    batch = TensorBatch(
        states=torch.randn(8, 3),
        actions=torch.tanh(torch.randn(8, 3)),
        rewards=torch.randn(8, 1),
        next_states=torch.randn(8, 3),
        dones=torch.zeros(8, 1),
    )
    critic_before = next(agent.critic_one.parameters()).detach().clone()

    metrics = agent.update(batch)

    assert not torch.equal(critic_before, next(agent.critic_one.parameters()))
    assert all(math.isfinite(value) for value in metrics.values())
    assert math.isfinite(metrics["entropy"])
