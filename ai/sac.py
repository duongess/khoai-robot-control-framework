"""A minimal Soft Actor-Critic implementation for one-dimensional actions."""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as functional

from ai.config import SACConfig
from ai.networks import Critic, GaussianActor


@dataclass
class TensorBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class SACAgent:
    """Trains directly on transition batches supplied by the gRPC learner service."""

    def __init__(self, config: SACConfig) -> None:
        self.config = config
        self.device = torch.device("cpu")
        torch.manual_seed(config.seed)

        self.actor = GaussianActor(config.state_dim, config.hidden_dim).to(self.device)
        self.critic_one = Critic(config.state_dim, config.hidden_dim).to(self.device)
        self.critic_two = Critic(config.state_dim, config.hidden_dim).to(self.device)
        self.target_critic_one = deepcopy(self.critic_one).to(self.device)
        self.target_critic_two = deepcopy(self.critic_two).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_one_optimizer = torch.optim.Adam(self.critic_one.parameters(), lr=config.learning_rate)
        self.critic_two_optimizer = torch.optim.Adam(self.critic_two.parameters(), lr=config.learning_rate)
        self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)

    def act(self, states: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        self._validate_states(states)
        with torch.no_grad():
            actions, _ = self.actor.sample(states.to(self.device), deterministic=deterministic)
        return actions.cpu()

    def update(self, batch: TensorBatch) -> dict[str, float]:
        states, actions, rewards, next_states, dones = self._prepare_batch(batch)

        with torch.no_grad():
            next_actions, next_log_probability = self.actor.sample(next_states)
            target_q = torch.minimum(
                self.target_critic_one(next_states, next_actions),
                self.target_critic_two(next_states, next_actions),
            ) - self.alpha.detach() * next_log_probability
            target_value = rewards + self.config.gamma * (1 - dones) * target_q

        critic_one_loss = functional.mse_loss(self.critic_one(states, actions), target_value)
        critic_two_loss = functional.mse_loss(self.critic_two(states, actions), target_value)
        self._step_optimizer(self.critic_one_optimizer, critic_one_loss)
        self._step_optimizer(self.critic_two_optimizer, critic_two_loss)

        sampled_actions, log_probability = self.actor.sample(states)
        actor_q = torch.minimum(
            self.critic_one(states, sampled_actions),
            self.critic_two(states, sampled_actions),
        )
        actor_loss = (self.alpha.detach() * log_probability - actor_q).mean()
        self._step_optimizer(self.actor_optimizer, actor_loss)

        alpha_loss = -(self.log_alpha * (log_probability + self.config.target_entropy).detach()).mean()
        self._step_optimizer(self.alpha_optimizer, alpha_loss)
        self._soft_update_targets()

        return {
            "actor_loss": float(actor_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "critic_one_loss": float(critic_one_loss.detach()),
            "critic_two_loss": float(critic_two_loss.detach()),
        }

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _prepare_batch(self, batch: TensorBatch) -> tuple[torch.Tensor, ...]:
        tensors = tuple(
            tensor.to(self.device, dtype=torch.float32)
            for tensor in (batch.states, batch.actions, batch.rewards, batch.next_states, batch.dones)
        )
        states, actions, rewards, next_states, dones = tensors
        self._validate_states(states)
        self._validate_states(next_states)
        expected_shape = (states.shape[0], 1)
        for name, tensor in (("actions", actions), ("rewards", rewards), ("dones", dones)):
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}")
        if states.shape[0] == 0:
            raise ValueError("transition batch must not be empty")
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("transition batch contains non-finite values")
        if (actions < -1).any() or (actions > 1).any():
            raise ValueError("actions must be between -1 and 1")
        return tensors

    def _validate_states(self, states: torch.Tensor) -> None:
        if states.ndim != 2 or states.shape[1] != self.config.state_dim:
            raise ValueError(f"states must have shape (batch, {self.config.state_dim})")
        if not torch.isfinite(states).all():
            raise ValueError("states must contain finite values")

    @staticmethod
    def _step_optimizer(optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _soft_update_targets(self) -> None:
        with torch.no_grad():
            for target, source in (
                (self.target_critic_one, self.critic_one),
                (self.target_critic_two, self.critic_two),
            ):
                for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
                    target_parameter.lerp_(source_parameter, self.config.tau)
