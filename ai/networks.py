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


class ParametricMLPPolicy(nn.Module):
    """Dense context encoder for SAC policies whose actions are reflex parameters.

    The dynamic-reflex integration attaches the bounded parameter head and
    evaluates ``f(x; theta)``. Unlike :class:`GaussianActor`, this policy never
    substitutes a three-value motor vector for the six-value compliance vector.
    """

    # Go's pure_rl mode consumes the residual protocol stream. The value in that
    # stream is the complete evaluated reflex action, not a zero correction.
    parametric_action_stream = "residual"

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, min_log_std: float = -3.0) -> None:
        super().__init__()
        if state_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("state, action, and hidden dimensions must be positive")
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.context_dim = hidden_dim
        self.min_log_std = min_log_std
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.parameter_head: nn.Linear | None = None
        self.parameter_log_std: nn.Parameter | None = None
        self.register_buffer("reflex_parameter_min", torch.empty(0))
        self.register_buffer("reflex_parameter_max", torch.empty(0))
        self._dynamic_reflex_registry = None

    def _policy_context(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError(f"observation must have shape (batch, {self.observation_dim})")
        if not torch.isfinite(observation).all():
            raise ValueError("observation must contain finite values")
        return self.backbone(observation)


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
