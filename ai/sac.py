"""A minimal Soft Actor-Critic implementation for one-dimensional actions."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional

from ai.config import SACConfig
from ai.connectome.graph import ConnectomeGraph
from ai.connectome.policy import FlyConnectomePolicy, RandomGraphPolicy
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

        self.actor = self._build_actor(config).to(self.device)
        self.critic_one = Critic(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic_two = Critic(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_critic_one = deepcopy(self.critic_one).to(self.device)
        self.target_critic_two = deepcopy(self.critic_two).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_one_optimizer = torch.optim.Adam(self.critic_one.parameters(), lr=config.learning_rate)
        self.critic_two_optimizer = torch.optim.Adam(self.critic_two.parameters(), lr=config.learning_rate)
        self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)

    def act(self, states: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Evaluate the currently trainable actor."""
        return self.act_with_actor(self.actor, states, deterministic=deterministic)

    def act_with_actor(self, actor: nn.Module, states: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Evaluate an immutable actor snapshot without touching live weights."""
        self._validate_states(states)
        with torch.no_grad():
            actions, _ = actor.sample(states.to(self.device), deterministic=deterministic)
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

        with torch.no_grad():
            _, current_log_std = self.actor(states)
            critic_one_q = self.critic_one(states, actions).mean()
            critic_two_q = self.critic_two(states, actions).mean()
        return {
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "entropy": float((-log_probability).mean().detach()),
            "critic_one_loss": float(critic_one_loss.detach()),
            "critic_two_loss": float(critic_two_loss.detach()),
            "critic_one_q": float(critic_one_q),
            "critic_two_q": float(critic_two_q),
            "alpha": float(self.alpha.detach()),
            "actor_log_std_horizontal": float(current_log_std[:, 0].mean()),
            "actor_log_std_vertical": float(current_log_std[:, 1].mean()) if self.config.action_dim > 1 else 0.0,
            "actor_log_std_gripper": float(current_log_std[:, 2].mean()) if self.config.action_dim > 2 else 0.0,
        }

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def checkpoint_state(self) -> dict[str, Any]:
        """Return every trainable SAC component needed for an exact resume."""
        return {
            "sac_config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "critic_one": self.critic_one.state_dict(),
            "critic_two": self.critic_two.state_dict(),
            "target_critic_one": self.target_critic_one.state_dict(),
            "target_critic_two": self.target_critic_two.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_one_optimizer": self.critic_one_optimizer.state_dict(),
            "critic_two_optimizer": self.critic_two_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().clone(),
            "torch_rng_state": torch.get_rng_state(),
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore a checkpoint created by :meth:`checkpoint_state` safely."""
        expected_config = asdict(self.config)
        if state.get("sac_config") != expected_config:
            raise ValueError("checkpoint SAC configuration does not match the selected model")
        required = (
            "actor", "critic_one", "critic_two", "target_critic_one", "target_critic_two",
            "actor_optimizer", "critic_one_optimizer", "critic_two_optimizer", "alpha_optimizer",
            "log_alpha", "torch_rng_state",
        )
        if any(key not in state for key in required):
            raise ValueError("checkpoint SAC state is incomplete")
        log_alpha = state["log_alpha"]
        if not isinstance(log_alpha, torch.Tensor) or log_alpha.shape != self.log_alpha.shape or not torch.isfinite(log_alpha).all():
            raise ValueError("checkpoint contains an invalid entropy temperature")
        try:
            self.actor.load_state_dict(state["actor"])
            self.critic_one.load_state_dict(state["critic_one"])
            self.critic_two.load_state_dict(state["critic_two"])
            self.target_critic_one.load_state_dict(state["target_critic_one"])
            self.target_critic_two.load_state_dict(state["target_critic_two"])
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_one_optimizer.load_state_dict(state["critic_one_optimizer"])
            self.critic_two_optimizer.load_state_dict(state["critic_two_optimizer"])
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
            self.log_alpha.data.copy_(log_alpha.to(self.device))
            torch.set_rng_state(state["torch_rng_state"])
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"could not restore SAC checkpoint state: {error}") from error

    @staticmethod
    def _build_actor(config: SACConfig) -> nn.Module:
        if config.controller_type == "mlp":
            return GaussianActor(config.state_dim, config.action_dim, config.hidden_dim, config.min_log_std)
        graph = ConnectomeGraph.load(config.graph_path or "")
        arguments = dict(
            observation_dim=config.state_dim,
            action_dim=config.action_dim,
            graph=graph,
            hidden_dim=config.hidden_dim,
            propagation_steps=config.propagation_steps,
            train_edge_gains=config.train_edge_gains,
            activation=config.activation,
            action_dead_zone=config.action_dead_zone,
            min_log_std=config.min_log_std,
            max_horizontal_speed=config.max_horizontal_speed,
            max_vertical_speed=config.max_vertical_speed,
            max_gripper_command=config.max_gripper_command,
        )
        if config.controller_type == "random_graph":
            return RandomGraphPolicy(**arguments, seed=config.seed)
            
        return FlyConnectomePolicy(**arguments)

    def _prepare_batch(self, batch: TensorBatch) -> tuple[torch.Tensor, ...]:
        tensors = tuple(
            tensor.to(self.device, dtype=torch.float32)
            for tensor in (batch.states, batch.actions, batch.rewards, batch.next_states, batch.dones)
        )
        states, actions, rewards, next_states, dones = tensors
        self._validate_states(states)
        self._validate_states(next_states)
        if tuple(actions.shape) != (states.shape[0], self.config.action_dim):
            raise ValueError(f"actions must have shape {(states.shape[0], self.config.action_dim)}")
        for name, tensor in (("rewards", rewards), ("dones", dones)):
            if tuple(tensor.shape) != (states.shape[0], 1):
                raise ValueError(f"{name} must have shape {(states.shape[0], 1)}")
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
