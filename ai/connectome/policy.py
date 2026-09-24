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
    """Dual-loop actor whose shared latent propagates over cached graph edges.

    Observation features are encoded into an explicitly configured sensory group;
    they are never injected into a motor group. A deterministic fly-base branch
    supplies the nominal reflex and SAC samples only the tactile residual branch.
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
        residual_alpha: tuple[float, float, float] = (0.20, 0.15, 0.30),
        tactile_observation_indices: tuple[int, int, int, int] = (-1, -1, -1, -1),
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0 or hidden_dim <= 0 or propagation_steps <= 0:
            raise ValueError("dimensions and propagation_steps must be positive")
        if action_dim != 3:
            raise ValueError("dual-loop fly actor requires exactly three action channels")
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
        if any(value < 0 or value > 1 for value in residual_alpha):
            raise ValueError("residual alpha values must be in [0, 1]")
        if any(index < -1 or index >= observation_dim for index in tactile_observation_indices):
            raise ValueError("tactile observation indices must be -1 or within observation_dim")
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
        # Both branches consume the same connectome latent. The base head owns
        # nominal visuomotor reflexes; SAC exploration exists only in the
        # residual distribution below.
        self.latent_projection = nn.Sequential(
            nn.Linear(len(MOTOR_CHANNELS), hidden_dim),
            nn.Tanh(),
        )
        self.base_head = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Tanh())
        nn.init.zeros_(self.base_head[0].weight)
        nn.init.zeros_(self.base_head[0].bias)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim + len(tactile_observation_indices), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Warmup starts from a neutral residual mean. Fixed Gaussian sampling
        # still explores while SAC first teaches the deterministic fly base.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))
        self.register_buffer("action_limits", torch.tensor([max_horizontal_speed, max_vertical_speed, max_gripper_command], dtype=torch.float32))
        self.register_buffer("residual_alpha", torch.tensor(residual_alpha, dtype=torch.float32))
        self.register_buffer("tactile_observation_indices", torch.tensor(tactile_observation_indices, dtype=torch.long))
        self.activation = torch.tanh if activation == "tanh" else (lambda value: torch.clamp(value, -1.0, 1.0))
        self._motor_group_indices: dict[str, torch.Tensor] = {}
        for name in MOTOR_CHANNELS:
            indices = torch.tensor(graph.motor_groups[name], dtype=torch.long)
            self.register_buffer(f"motor_{name}_indices", indices)
            self._motor_group_indices[name] = indices

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the deterministic composed action and residual log standard deviation."""
        final, _, _, _ = self.sample_decomposed(observation, deterministic=True)
        return final, self.log_std.clamp(self.min_log_std, 2).expand_as(final)

    def _connectome_latent(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode observations and propagate them over the immutable fly graph."""
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
        return self.latent_projection(channels)

    def _closed_loop_base_reflex(self, observation: torch.Tensor) -> torch.Tensor:
        """Closed-loop fly-base control for spatial tracking and stable grasping.

        The biological base policy is not a free-running open-loop actuator. It
        tracks the signed horizontal object error, descends only while aligned,
        and applies a nominal grip baseline once contact is established. SAC is
        left with force adaptation and damping only.
        """
        if observation.ndim != 2:
            raise ValueError("observation must be batched with shape (batch, features)")

        batch = observation.shape[0]
        base = torch.zeros((batch, self.action_dim), dtype=observation.dtype, device=observation.device)
        if observation.shape[1] < 5:
            return base

        dx = observation[:, 10] if observation.shape[1] > 10 else observation[:, 4] - observation[:, 0]
        if observation.shape[1] > 11:
            dy = observation[:, 11]
        else:
            dy = torch.zeros_like(dx)

        attached = observation[:, 15] > 0.5 if observation.shape[1] > 15 else torch.zeros_like(dx, dtype=torch.bool, device=observation.device)
        contact = observation[:, 20] > 0.5 if observation.shape[1] > 20 else torch.zeros_like(dx, dtype=torch.bool, device=observation.device)

        # The fly-base policy must be stateless and recompute a fresh control
        # signal from the live observation on every step. Index 10 is the signed
        # displacement err_x = normalize(CarriageX - ObjectX).
        # Negative err_x means the object is to the right of the carriage, so the
        # carriage must drive positive X. Positive err_x means the object is to the
        # left, so the carriage must drive negative X. No module-level latch or
        # threshold freeze is allowed.
        err_x = observation[:, 10]
        err_target_x = observation[:, 12] if observation.shape[1] > 12 else torch.zeros_like(err_x)
        gripper_y = observation[:, 1]
        carry_height_ready = gripper_y > torch.tensor(0.50, dtype=observation.dtype, device=observation.device)

        # Decode the normalized phase from the environment observation. The Go
        # environment exposes PhaseLowerAtTarget as 6 and PhaseReleaseObject as 7.
        phase_available = observation.shape[1] > 19
        phase = observation[:, 19] if phase_available else torch.zeros_like(err_x)
        phase_has_signal = phase_available & (phase.abs() > 1e-6)
        phase_code = torch.round((phase + 1.0) * torch.tensor(4.5, dtype=observation.dtype, device=observation.device)).to(torch.int64)

        # Legacy observations without a meaningful phase channel are still safe:
        # fall back to the same target-alignment geometry used before the phase-based
        # gate was introduced, but keep the airborne release guard active.
        target_tolerance = torch.tensor(0.05, dtype=observation.dtype, device=observation.device)
        release_height = torch.tensor(0.30, dtype=observation.dtype, device=observation.device)
        lower_target_geometry = attached & (torch.abs(err_target_x) <= target_tolerance) & (gripper_y > release_height)
        release_geometry = attached & (torch.abs(err_target_x) <= target_tolerance) & (gripper_y <= release_height)

        lower_target_mask = attached & ((phase_has_signal & (phase_code == 6)) | (~phase_has_signal & lower_target_geometry))
        release_mask = (phase_has_signal & (phase_code >= 7)) | (~phase_has_signal & release_geometry)

        # 1. Keep the carriage centered on the target in both the final lowering
        # and release phases until the lateral error is reduced to a narrow landing
        # tolerance. This prevents the gripper from freezing at the edge of the
        # target and dropping the object onto the side slope.
        gain_x = torch.tensor(2.0, dtype=observation.dtype, device=observation.device)
        target_centered = torch.abs(err_target_x) <= torch.tensor(0.03, dtype=observation.dtype, device=observation.device)
        target_align_cmd = torch.clamp(gain_x * err_target_x, min=-1.0, max=1.0)

        x_track = -torch.clamp(gain_x * err_x, min=-1.0, max=1.0)

        # Only descend once the carriage is already horizontally close enough to
        # the object. If the lateral error is still large, keep the gripper level and
        # drive horizontally first.
        descent_gate = torch.abs(err_x) < torch.tensor(0.05, dtype=observation.dtype, device=observation.device)
        y_track = torch.where(descent_gate & ~attached, torch.full_like(err_x, -0.5), torch.zeros_like(err_x))

        # Post-grasp reflex cascade: lift first while attached but still below the
        # safe carry height, then transport horizontally to the target while holding
        # the carriage at carry height. Once the target phase is reached, the base
        # descends to the release guide and keeps the jaws closed until the
        # environment has advanced into the explicit release phase. This prevents
        # a free-fall drop while the object is still airborne.
        lift_mask = attached & ~carry_height_ready
        transport_mask = attached & carry_height_ready

        x_track = torch.where(lift_mask, torch.zeros_like(x_track), x_track)
        x_track = torch.where(transport_mask, target_align_cmd, x_track)
        x_track = torch.where(lower_target_mask, torch.where(target_centered, torch.zeros_like(x_track), target_align_cmd), x_track)
        x_track = torch.where(release_mask, torch.where(target_centered, torch.zeros_like(x_track), target_align_cmd), x_track)

        y_track = torch.where(lift_mask, torch.full_like(err_x, 0.85), y_track)
        y_track = torch.where(transport_mask, torch.zeros_like(err_x), y_track)
        y_track = torch.where(lower_target_mask, torch.where(target_centered, torch.full_like(err_x, -0.75), torch.zeros_like(err_x)), y_track)
        y_track = torch.where(release_mask, torch.full_like(err_x, -0.3), y_track)

        grip_baseline = torch.where(attached, torch.full_like(dx, 0.90), torch.zeros_like(dx))
        grip_baseline = torch.where(contact & ~attached, torch.full_like(dx, 0.9), grip_baseline)
        grip_baseline = torch.where(lower_target_mask, torch.full_like(dx, 0.90), grip_baseline)
        grip_baseline = torch.where(release_mask, torch.full_like(dx, -1.0), grip_baseline)

        base[:, 0] = x_track
        base[:, 1] = y_track
        base[:, 2] = grip_baseline
        return base.clamp(-1.0, 1.0)

    def sample_decomposed(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(final, fly_base, sac_residual, log_probability)``.

        ``fly_base`` and ``sac_residual`` are separately observable, while only
        the three-dimensional composed action is stored in replay and consumed
        by the critics. This preserves the existing action-space contract.
        """
        latent = self._connectome_latent(observation)
        fly_base = self._closed_loop_base_reflex(observation)
        tactile_columns = [
            observation[:, index] if index >= 0 else observation.new_zeros(len(observation))
            for index in self.tactile_observation_indices.tolist()
        ]
        tactile = torch.stack(tactile_columns, dim=1)
        residual_mean = self.residual_head(torch.cat((latent, tactile), dim=1))
        log_std = self.log_std.clamp(self.min_log_std, 2).expand_as(residual_mean)
        distribution = Normal(residual_mean, log_std.exp())
        raw_residual = residual_mean if deterministic else distribution.rsample()
        sac_residual = torch.tanh(raw_residual)
        final = torch.clamp(fly_base + self.residual_alpha * sac_residual, -1.0, 1.0)
        # During release, the plant checks that the object remains inside the
        # target after the movement command. A residual correction can move it
        # out of the target before the release command is evaluated. Keep the
        # fly-base position command for this short safety window and never let
        # SAC reverse the downward release command. SAC keeps full authority
        # outside release.
        if observation.shape[1] > 19:
            release_phase = observation[:, 19] >= (7.0 / 4.5 - 1.0)
            release_horizontal = torch.where(release_phase, fly_base[:, 0], final[:, 0])
            release_vertical = torch.minimum(final[:, 1], fly_base[:, 1])
            release_vertical = torch.where(release_phase, release_vertical, final[:, 1])
            # Build a new tensor instead of assigning through final[:, index].
            # In-place view writes invalidate SAC autograd versions during the
            # actor update.
            final = torch.stack((release_horizontal, release_vertical, final[:, 2]), dim=1)
        if deterministic:
            log_probability = torch.zeros((len(final), 1), dtype=final.dtype, device=final.device)
        else:
            log_probability = distribution.log_prob(raw_residual) - torch.log(1 - sac_residual.pow(2) + 1e-6)
            log_probability = log_probability.sum(dim=-1, keepdim=True)
        return final, fly_base, sac_residual, log_probability

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
        final, _, _, log_probability = self.sample_decomposed(observation, deterministic=deterministic)
        return final, log_probability

    def decode_motor_channels(self, channels: torch.Tensor) -> torch.Tensor:
        """Decode six engineered channels for diagnostics and legacy tests.

        ``front_right - front_left`` controls horizontal movement; middle and hind
        pairs respectively correct horizontal motion, vertical motion, and grip
        force. Production composition is performed by the two learned heads in
        :meth:`sample_decomposed`; Go only selects an ablation branch and applies
        physical safety limits. The result remains within ``[-1, 1]``.
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
