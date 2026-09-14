"""Neural networks used by the Soft Actor-Critic learner."""

import torch
from torch import nn
from torch.distributions import Normal


class GaussianActor(nn.Module):
    """A Gaussian policy with tanh-squashed continuous actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(states)
        return self.mean(features), self.log_std(features).clamp(-20, 2)

    def sample(self, states: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(states)
        distribution = Normal(mean, log_std.exp())
        raw_action = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw_action)

        if deterministic:
            log_probability = torch.zeros_like(action)
        else:
            log_probability = distribution.log_prob(raw_action)
            log_probability -= torch.log(1 - action.pow(2) + 1e-6)

        return action, log_probability.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    """A state-action value estimator."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((states, actions), dim=-1))
