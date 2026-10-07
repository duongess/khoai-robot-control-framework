"""Neural networks used by the Soft Actor-Critic learner."""

import torch
from torch import nn
from torch.distributions import Normal


class GaussianActor(nn.Module):
    """A Gaussian policy with tanh-squashed continuous actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, min_log_std: float = -3.0) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.min_log_std = min_log_std

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(states)
        return self.mean(features), self.log_std(features).clamp(self.min_log_std, 2)

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


class DenseParametricActor(nn.Module):
    """Dense feature extractor feeding the same registered reflex law as the graph actor."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, min_log_std: float = -3.0) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.min_log_std = min_log_std
        self.feature_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

    def _connectome_latent(self, observation: torch.Tensor) -> torch.Tensor:
        return self.backbone(observation)

    def register_reflex_law(self, config_dict: dict) -> dict:
        from ai.connectome.dynamic_policy import register_reflex_law
        return register_reflex_law(self, config_dict)

    @property
    def reflex_parameter_count(self) -> int:
        from ai.connectome.dynamic_policy import reflex_parameter_count
        return reflex_parameter_count(self)

    def sample_decomposed(self, observation: torch.Tensor, deterministic: bool = False):
        from ai.connectome.dynamic_policy import _sample_parametric
        action, _, log_probability, _ = _sample_parametric(self, observation, deterministic)
        return action, action, torch.zeros_like(action), log_probability

    def sample(self, observation: torch.Tensor, deterministic: bool = False):
        action, _, _, log_probability = self.sample_decomposed(observation, deterministic)
        return action, log_probability

    def forward(self, observation: torch.Tensor):
        from ai.connectome.dynamic_policy import _sample_parametric
        action, log_std, _, _ = _sample_parametric(self, observation, True)
        return action, log_std

    def get_extra_state(self):
        from ai.connectome.dynamic_policy import get_extra_state
        return get_extra_state(self)

    def set_extra_state(self, payload):
        from ai.connectome.dynamic_policy import set_extra_state
        set_extra_state(self, payload)
