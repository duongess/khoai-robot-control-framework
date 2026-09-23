"""Sparse, connectome-constrained SAC actor policies."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.distributions import Normal

from ai.connectome.graph import ConnectomeGraph, GraphArtifactError


MOTOR_CHANNELS = (
    "front_left",
    "front_right",
    "middle_left",
    "middle_right",
    "hind_left",
    "hind_right",
)


class FlyConnectomePolicy(nn.Module):
    """Gaussian actor whose hidden state propagates only over cached graph edges.

    Observation features are encoded into an explicitly configured sensory group;
    they are never injected into a motor group.  The six graph readouts are an
    engineered robot decoder, not a claim about fly motor semantics.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        graph: ConnectomeGraph,
        hidden_dim: int = 64,
        propagation_steps: int = 4,
        train_edge_gains: bool = True,
        activation: str = "tanh",
        input_group: str = "sensory",
        action_dead_zone: float = 0.1,
        min_log_std: float = -3.0,
        max_horizontal_speed: float = 1.0,
        max_vertical_speed: float = 1.0,
        max_gripper_command: float = 1.0,
        phase_gated_decoder: bool = False,
        object_attached_observation_index: int = -1,
        phase_observation_index: int = -1,
        transport_phase_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0 or hidden_dim <= 0 or propagation_steps <= 0:
            raise ValueError("dimensions and propagation_steps must be positive")
        if activation not in {"tanh", "hardtanh"}:
            raise ValueError("activation must be tanh or hardtanh")
        if not 0 <= action_dead_zone < 1:
            raise ValueError("action_dead_zone must be in [0, 1)")
        if min(max_horizontal_speed, max_vertical_speed, max_gripper_command) <= 0:
            raise ValueError("action limits must be positive")
        if phase_gated_decoder:
            phase_indices = (object_attached_observation_index, phase_observation_index)
            if any(index < 0 or index >= observation_dim for index in phase_indices):
                raise ValueError("phase-gated decoder observation indices must be within observation_dim")
            if object_attached_observation_index == phase_observation_index:
                raise ValueError("phase-gated decoder observation indices must be distinct")
            if not torch.isfinite(torch.tensor(transport_phase_threshold)):
                raise ValueError("transport_phase_threshold must be finite")
        missing = [name for name in MOTOR_CHANNELS if name not in graph.motor_groups]
        if missing:
            raise GraphArtifactError(f"graph is missing required motor groups: {', '.join(missing)}")
        if input_group not in graph.input_groups:
            raise GraphArtifactError(f"graph is missing configured input group {input_group!r}")
        sensory_indices = tuple(graph.input_groups[input_group])
        motor_nodes = {index for group in graph.motor_groups.values() for index in group}
        if motor_nodes.intersection(sensory_indices):
            raise GraphArtifactError("sensory input group must not overlap motor groups")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.node_count = graph.node_count
        self.propagation_steps = propagation_steps
        self.action_dead_zone = action_dead_zone
        self.min_log_std = min_log_std
        self.phase_gated_decoder = phase_gated_decoder
        self.object_attached_observation_index = object_attached_observation_index
        self.phase_observation_index = phase_observation_index
        self.transport_phase_threshold = transport_phase_threshold
        self.register_buffer("adjacency", graph.torch_adjacency())
        self.register_buffer("base_edge_weights", torch.tensor(graph.weights, dtype=torch.float32))
        self.register_buffer("edge_indices", torch.tensor(graph.edge_index[[1, 0]], dtype=torch.long))
        self.register_buffer("sensory_indices", torch.tensor(sensory_indices, dtype=torch.long))
        self.sensory_encoder = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, len(sensory_indices)))
        self.neuron_bias = nn.Parameter(torch.zeros(graph.node_count))
        self.leak_logit = nn.Parameter(torch.zeros(graph.node_count))
        self.edge_log_gains = nn.Parameter(torch.zeros(graph.edge_count), requires_grad=train_edge_gains)
        self.motor_gain = nn.Parameter(torch.ones(len(MOTOR_CHANNELS)))
        self.motor_bias = nn.Parameter(torch.zeros(len(MOTOR_CHANNELS)))
        # The sparse neural core remains shared.  Only this engineered output
        # adapter changes once an object is attached and the task reaches
        # MoveToTarget (the force-control observation encodes that phase as 0).
        if phase_gated_decoder:
            self.transport_motor_gain = nn.Parameter(torch.ones(len(MOTOR_CHANNELS)))
            self.transport_motor_bias = nn.Parameter(torch.zeros(len(MOTOR_CHANNELS)))
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))
        self.register_buffer("action_limits", torch.tensor([max_horizontal_speed, max_vertical_speed, max_gripper_command], dtype=torch.float32))
        self.activation = torch.tanh if activation == "tanh" else (lambda value: torch.clamp(value, -1.0, 1.0))
        self._motor_group_indices: dict[str, torch.Tensor] = {}
        for name in MOTOR_CHANNELS:
            indices = torch.tensor(graph.motor_groups[name], dtype=torch.long)
            self.register_buffer(f"motor_{name}_indices", indices)
            self._motor_group_indices[name] = indices

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tanh-safe action means and bounded Gaussian log standard deviations."""
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError(f"observation must have shape (batch, {self.observation_dim})")
        if not torch.isfinite(observation).all():
            raise ValueError("observation must contain finite values")
        batch = observation.shape[0]
        state = observation.new_zeros((batch, self.node_count))
        encoded = self.sensory_encoder(observation)
        state[:, self.sensory_indices] = encoded
        adjacency = self._weighted_adjacency()
        leak = torch.sigmoid(self.leak_logit).unsqueeze(0)
        for _ in range(self.propagation_steps):
            messages = torch.sparse.mm(adjacency, state.T).T
            candidate = self.activation(messages + self.neuron_bias)
            state = (1.0 - leak) * state + leak * candidate
            if not torch.isfinite(state).all():
                raise RuntimeError("non-finite connectome state; inspect graph weights and training settings")
            state = torch.clamp(state, -1.0, 1.0)
        channels = torch.stack([state[:, self._motor_group_indices[name]].mean(dim=1) for name in MOTOR_CHANNELS], dim=1)
        channels = self._apply_motor_decoder_heads(channels, observation)
        mean = self.decode_motor_channels(channels)
        if self.action_dim > 3:
            mean = torch.cat((mean, mean.new_zeros((batch, self.action_dim - 3))), dim=1)
        elif self.action_dim < 3:
            mean = mean[:, : self.action_dim]
        return mean, self.log_std.clamp(self.min_log_std, 2).expand_as(mean)

    def _apply_motor_decoder_heads(self, channels: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        """Select a learned motor readout without blending contradictory actions.

        Before a secure attachment, and through lifting, the acquisition head is
        used.  Once the object is attached and the phase feature has reached
        the configured transport threshold, a separately trainable transport
        head drives the *same* sparse connectome state.  The task's phase is an
        observation, so SAC can still learn both behaviours end to end.
        """
        approach_channels = torch.tanh(channels * self.motor_gain + self.motor_bias)
        if not self.phase_gated_decoder:
            return approach_channels
        transport_channels = torch.tanh(channels * self.transport_motor_gain + self.transport_motor_bias)
        transport_mask = (
            (observation[:, self.object_attached_observation_index] > 0)
            & (observation[:, self.phase_observation_index] >= self.transport_phase_threshold)
        )
        return torch.where(transport_mask.unsqueeze(1), transport_channels, approach_channels)

    def sample(self, observation: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(observation)
        distribution = Normal(mean, log_std.exp())
        raw_action = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw_action)
        if deterministic:
            return action, torch.zeros((len(action), 1), dtype=action.dtype, device=action.device)
        log_probability = distribution.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_probability.sum(dim=-1, keepdim=True)

    def decode_motor_channels(self, channels: torch.Tensor) -> torch.Tensor:
        """Decode six engineered channels to normalized robot-arm commands.

        ``front_right - front_left`` controls horizontal movement; middle and hind
        pairs respectively control vertical movement and the gripper.  The result
        remains within the existing generic SAC action contract ``[-1, 1]``.
        """
        if channels.ndim != 2 or channels.shape[1] != len(MOTOR_CHANNELS):
            raise ValueError("channels must have shape (batch, 6)")
        actions = torch.stack((
            channels[:, 1] - channels[:, 0],
            channels[:, 3] - channels[:, 2],
            channels[:, 4] - channels[:, 5],
        ), dim=1)
        actions = torch.tanh(actions) * self.action_limits
        # The environment owns the hardware dead zone and reports when it
        # removes a command. Keeping the actor output continuous here preserves
        # trainable small corrections for SAC.
        return actions.clamp(-1.0, 1.0)

    def _weighted_adjacency(self) -> torch.Tensor:
        values = self.base_edge_weights * torch.exp(self.edge_log_gains.clamp(-4, 4))
        # Sparse coalescing handles parallel edges while preserving source->target.
        return torch.sparse_coo_tensor(self.edge_indices, values, self.adjacency.shape, device=values.device, check_invariants=False).coalesce()


class RandomGraphPolicy(FlyConnectomePolicy):
    """Degree-preserving random-topology baseline with the same actor budget."""

    def __init__(self, *args, graph: ConnectomeGraph, seed: int = 0, **kwargs) -> None:
        super().__init__(*args, graph=graph.randomized(seed), **kwargs)
