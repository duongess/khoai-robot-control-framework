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


class HeuristicGuidedPolicy(GaussianActor):
    """SAC residual policy guided by the factory's fixed pick-and-place phases.

    The scripted component is deliberately stateless: every action is computed
    from the current observation and phase supplied by the plant.  SAC can only
    make bounded corrections around that safe nominal trajectory.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, min_log_std: float = -3.0) -> None:
        super().__init__(state_dim, action_dim, hidden_dim, min_log_std)
        if action_dim != 3:
            raise ValueError("heuristic_guided requires the three-channel gantry action")
        self.register_buffer("residual_alpha", torch.tensor((0.20, 0.15, 0.30)))

    @staticmethod
    def _scripted_phases(states: torch.Tensor) -> torch.Tensor:
        base = torch.zeros((len(states), 3), dtype=states.dtype, device=states.device)
        if states.shape[1] < 2:
            return base
        err_object_x = states[:, 10] if states.shape[1] > 10 else torch.zeros_like(states[:, 0])
        err_target_x = states[:, 12] if states.shape[1] > 12 else err_object_x
        attached = states[:, 15] > 0.5 if states.shape[1] > 15 else torch.zeros_like(err_object_x, dtype=torch.bool)
        contact = states[:, 20] > 0.5 if states.shape[1] > 20 else torch.zeros_like(attached)
        phase = states[:, 19] if states.shape[1] > 19 else torch.zeros_like(err_object_x)
        phase_code = torch.round((phase + 1.0) * 4.5).to(torch.int64)

        object_centered = err_object_x.abs() < 0.03
        target_centered = err_target_x.abs() < 0.03
        acquire_x = (-2.0 * err_object_x).clamp(-1.0, 1.0)
        target_x = (2.0 * err_target_x).clamp(-1.0, 1.0)
        base[:, 0] = torch.where(attached, target_x, acquire_x)
        base[:, 1] = torch.where(~attached & object_centered, torch.full_like(err_object_x, -0.65), torch.zeros_like(err_object_x))
        base[:, 2] = torch.where(contact | attached, torch.full_like(err_object_x, 0.90), torch.zeros_like(err_object_x))

        lifting = attached & (phase_code < 5)
        lowering = attached & (phase_code == 6)
        releasing = attached & (phase_code >= 7)
        base[:, 0] = torch.where(lifting, torch.zeros_like(err_object_x), base[:, 0])
        base[:, 1] = torch.where(lifting, torch.full_like(err_object_x, 0.85), base[:, 1])
        base[:, 1] = torch.where(lowering & target_centered, torch.full_like(err_object_x, -0.75), base[:, 1])
        base[:, 1] = torch.where(releasing, torch.full_like(err_object_x, -0.30), base[:, 1])
        base[:, 2] = torch.where(releasing, torch.full_like(err_object_x, -1.0), base[:, 2])
        return base.clamp(-1.0, 1.0)

    def sample_decomposed(
        self, states: torch.Tensor, deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(states)
        distribution = Normal(mean, log_std.exp())
        raw_residual = mean if deterministic else distribution.rsample()
        residual = torch.tanh(raw_residual)
        base = self._scripted_phases(states)
        action = torch.clamp(base + self.residual_alpha * residual, -1.0, 1.0)
        if deterministic:
            log_probability = torch.zeros((len(states), 1), dtype=states.dtype, device=states.device)
        else:
            log_probability = distribution.log_prob(raw_residual)
            log_probability -= torch.log(1 - residual.pow(2) + 1e-6)
            log_probability = log_probability.sum(dim=-1, keepdim=True)
        return action, base, residual, log_probability

    def sample(self, states: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        action, _, _, log_probability = self.sample_decomposed(states, deterministic)
        return action, log_probability


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
