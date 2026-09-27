"""Parametric-SAC integration for registered differentiable reflex laws."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as functional

from ai.connectome.dynamic_reflex import DynamicReflexEngine, ReflexConfigError, ReflexFunctionConfig
from ai.connectome.policy import FlyConnectomePolicy


# Canonical signed-error law for the 30-feature force-control observation.
DEFAULT_GANTRY_REFLEX_CONFIG: dict[str, Any] = {
    "channels": [
        {"action_index": 0, "expression_type": "linear", "signal_indices": {"x": 10}, "parameters": [{"name": "a", "min_val": 0.5, "max_val": 3.0, "default": 2.0}, {"name": "b", "min_val": -0.5, "max_val": 0.5, "default": 0.0}]},
        {"action_index": 1, "expression_type": "linear", "signal_indices": {"x": 11}, "parameters": [{"name": "a", "min_val": 0.5, "max_val": 2.0, "default": 1.0}, {"name": "b", "min_val": -0.8, "max_val": -0.3, "default": -0.5}]},
        {"action_index": 2, "expression_type": "positive_linear", "signal_indices": {"x": 20}, "parameters": [{"name": "a", "min_val": 0.5, "max_val": 2.0, "default": 1.0}, {"name": "b", "min_val": 0.0, "max_val": 1.0, "default": 0.5}]},
    ]
}
_DEFAULT_GANTRY_SIGNAL_INDICES = {0: 10, 1: 11, 2: 20}


@dataclass
class _RegistryState:
    config: ReflexFunctionConfig
    engine: DynamicReflexEngine


def _state(policy: FlyConnectomePolicy) -> _RegistryState | None:
    return getattr(policy, "_dynamic_reflex_registry", None)


def _apply_vertical_tracking_prior(config: ReflexFunctionConfig) -> ReflexFunctionConfig:
    """Seed the vertical tracking intercept toward descent, never ascent."""
    channels = []
    for channel_index, channel in enumerate(config.channels):
        action_index = channel.action_index if channel.action_index is not None else channel_index
        if action_index != 1 or channel.expression_type not in {"linear", "saturated_linear"}:
            channels.append(channel)
            continue
        parameters = []
        for spec in channel.parameters:
            if spec.name != "b":
                parameters.append(spec)
                continue
            low, high = max(spec.min_val, -0.8), min(spec.max_val, -0.3)
            if low > high:
                raise ReflexConfigError("vertical tracking bias b must overlap the safe descent prior [-0.8, -0.3]")
            parameters.append(replace(spec, default=min(max(-0.5, low), high)))
        channels.append(replace(channel, parameters=tuple(parameters)))
    return replace(config, channels=tuple(channels))


def register_reflex_law(policy: FlyConnectomePolicy, config_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Install a law whose *only* stochastic actor output is its parameters."""
    config = config_dict if isinstance(config_dict, ReflexFunctionConfig) else ReflexFunctionConfig.from_dict(config_dict)
    config = _apply_vertical_tracking_prior(config)
    action_indices = [channel.action_index if channel.action_index is not None else index for index, channel in enumerate(config.channels)]
    if len(set(action_indices)) != len(action_indices):
        raise ReflexConfigError("each reflex channel must control a distinct action_index")
    if any(index >= policy.action_dim for index in action_indices):
        raise ReflexConfigError(f"reflex action_index must be less than action_dim ({policy.action_dim})")
    engine = DynamicReflexEngine(config)  # Compile and validate before state changes.
    if config.parameter_count == 0:
        raise ReflexConfigError("a Parametric SAC reflex requires at least one free parameter")
    reference = policy.base_head[0].weight
    head = nn.Linear(reference.shape[1], config.parameter_count, device=reference.device, dtype=reference.dtype)
    low = torch.tensor([spec.min_val for channel in config.channels for spec in channel.parameters], dtype=reference.dtype, device=reference.device)
    high = torch.tensor([spec.max_val for channel in config.channels for spec in channel.parameters], dtype=reference.dtype, device=reference.device)
    defaults = torch.tensor([spec.default for channel in config.channels for spec in channel.parameters], dtype=reference.dtype, device=reference.device)
    with torch.no_grad():
        nn.init.orthogonal_(head.weight, gain=2.0)
        fraction = (defaults - low) / (high - low)
        head.bias.copy_(torch.logit(fraction.clamp(1e-5, 1 - 1e-5)))
    if "reflex_parameter_min" not in policy._buffers:
        policy.register_buffer("reflex_parameter_min", low)
        policy.register_buffer("reflex_parameter_max", high)
    else:
        policy.reflex_parameter_min = low
        policy.reflex_parameter_max = high
    policy.parameter_head = head
    policy.parameter_log_std = nn.Parameter(torch.full((config.parameter_count,), -1.0, device=reference.device, dtype=reference.dtype))
    policy._dynamic_reflex_registry = _RegistryState(config, engine)
    return config.to_dict()


def register_default_gantry_reflex_law(policy: FlyConnectomePolicy) -> dict[str, Any]:
    return register_reflex_law(policy, DEFAULT_GANTRY_REFLEX_CONFIG)


def clear_reflex_law(policy: FlyConnectomePolicy) -> None:
    policy.parameter_head = None
    policy.parameter_log_std = None
    policy.reflex_parameter_min = torch.empty(0, device=policy.base_head[0].weight.device)
    policy.reflex_parameter_max = torch.empty(0, device=policy.base_head[0].weight.device)
    policy._dynamic_reflex_registry = None


def reflex_parameter_count(policy: FlyConnectomePolicy) -> int:
    state = _state(policy)
    return 0 if state is None else state.config.parameter_count


def _parameter_mapping(policy: FlyConnectomePolicy, values: torch.Tensor) -> dict[str, torch.Tensor]:
    state = _state(policy)
    assert state is not None
    mapped: dict[str, torch.Tensor] = {}
    offset = 0
    for channel_index, channel in enumerate(state.config.channels):
        channel_name = channel.name or f"channel_{channel_index}"
        for spec in channel.parameters:
            mapped[DynamicReflexEngine.parameter_key(channel_index, spec.name)] = values[:, offset]
            mapped[f"{channel_name}.{spec.name}"] = values[:, offset]
            offset += 1
    return mapped


def _parameter_statistics(policy: FlyConnectomePolicy, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if _state(policy) is None or policy.parameter_head is None or policy.parameter_log_std is None:
        raise RuntimeError("no Parametric SAC reflex law is registered")
    latent = policy._connectome_latent(observation)
    return policy.parameter_head(latent), policy.parameter_log_std.clamp(policy.min_log_std, 2).expand(len(observation), -1)


def _bound_parameters(policy: FlyConnectomePolicy, raw_parameters: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    values = policy.reflex_parameter_min + (policy.reflex_parameter_max - policy.reflex_parameter_min) * torch.sigmoid(raw_parameters)
    return values, _parameter_mapping(policy, values)


def bounded_reflex_parameters(policy: FlyConnectomePolicy, observation: torch.Tensor) -> dict[str, torch.Tensor]:
    """Deterministic, bounded parameter telemetry at the supplied state."""
    mean, _ = _parameter_statistics(policy, observation)
    return _bound_parameters(policy, mean)[1]


def _evaluate_reflex(policy: FlyConnectomePolicy, observation: torch.Tensor, parameters: Mapping[str, torch.Tensor]) -> torch.Tensor:
    state = _state(policy)
    assert state is not None
    outputs: list[torch.Tensor] = []
    action_indices: list[int] = []
    for channel_index, (channel, tree) in enumerate(zip(state.config.channels, state.engine._trees, strict=True)):
        action_index = channel.action_index if channel.action_index is not None else channel_index
        default_signal_index = channel.signal_indices.get("x", channel.signal_indices.get("err", channel.signal_indices.get("error", _DEFAULT_GANTRY_SIGNAL_INDICES.get(action_index))))
        if default_signal_index is None or default_signal_index >= observation.shape[1]:
            raise ReflexConfigError(f"channel {action_index} requires an in-range explicit signal_indices mapping for this observation schema")
        fallback = observation[:, default_signal_index]
        signals = {"x": fallback, "err": fallback, "error": fallback, "vel": torch.zeros_like(fallback)}
        signals.update({name: observation[:, column] for name, column in channel.signal_indices.items()})
        channel_parameters = {spec.name: parameters[DynamicReflexEngine.parameter_key(channel_index, spec.name)] for spec in channel.parameters}
        value = state.engine._evaluate_node(tree.body, {**signals, **channel_parameters})
        if not isinstance(value, torch.Tensor):
            value = torch.full_like(fallback, float(value))
        if value.ndim != 1 or not torch.isfinite(value).all():
            raise RuntimeError("dynamic reflex produced an invalid action")
        outputs.append(value)
        action_indices.append(action_index)
    by_action = {action_index: outputs[index] for index, action_index in enumerate(action_indices)}
    return torch.stack([by_action.get(index, observation.new_zeros(len(observation))) for index in range(policy.action_dim)], dim=1).clamp(-1.0, 1.0)


def _uses_default_gantry_task_flow(policy: FlyConnectomePolicy) -> bool:
    """True only for the canonical three-channel gantry schema."""
    state = _state(policy)
    if state is None or len(state.config.channels) != 3:
        return False
    expected = ((0, "linear", 10), (1, "linear", 11), (2, "positive_linear", 20))
    return all(
        (channel.action_index if channel.action_index is not None else index, channel.expression_type, channel.signal_indices.get("x")) == item
        for index, (channel, item) in enumerate(zip(state.config.channels, expected, strict=True))
    )


def _apply_gantry_task_flow(
    observation: torch.Tensor, action: torch.Tensor, parameters: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Phase orchestration around parametric reflexes for pick, carry, and drop.

    Registered theta values remain gains and biases; phase selects the physical
    error and enforces the safe lift and release directions.
    """
    if observation.shape[1] <= 20:
        return action
    err_target_x = observation[:, 12]
    gripper_y = observation[:, 1]
    attached = observation[:, 15] > 0.5
    phase_code = torch.round((observation[:, 19] + 1.0) * 4.5).to(torch.int64)
    carry_ready = gripper_y > 0.50
    lift = attached & ~carry_ready
    lower_at_target = attached & (phase_code == 6)
    release = phase_code >= 7
    transport = attached & carry_ready & ~lower_at_target & ~release

    ax, bx = parameters["channel_0.a"], parameters["channel_0.b"]
    ay, by = parameters["channel_1.a"], parameters["channel_1.b"]
    ag, bg = parameters["channel_2.a"], parameters["channel_2.b"]
    target_x = (ax * err_target_x + bx).clamp(-1.0, 1.0)
    lift_y = (ay + by).clamp(min=0.35, max=1.0)
    lower_y = (-ay + by).clamp(min=-1.0, max=-0.35)
    release_y = (-ay + by).clamp(min=-1.0, max=-0.30)
    hold_grip = torch.maximum(action[:, 2], torch.full_like(action[:, 2], 0.50))
    open_grip = (-(ag + bg)).clamp(-1.0, -0.50)

    x, y, grip = action.unbind(dim=1)
    x = torch.where(lift, torch.zeros_like(x), x)
    x = torch.where(transport | lower_at_target | release, target_x, x)
    y = torch.where(lift, lift_y, y)
    y = torch.where(transport, torch.zeros_like(y), y)
    y = torch.where(lower_at_target, lower_y, y)
    y = torch.where(release, release_y, y)
    grip = torch.where(attached & ~release, hold_grip, grip)
    grip = torch.where(release, open_grip, grip)
    return torch.stack((x, y, grip), dim=1)


def _sample_parametric(policy: FlyConnectomePolicy, observation: torch.Tensor, deterministic: bool):
    mean, log_std = _parameter_statistics(policy, observation)
    distribution = Normal(mean, log_std.exp())
    raw_parameters = mean if deterministic else distribution.rsample()
    _, parameters = _bound_parameters(policy, raw_parameters)
    action = _evaluate_reflex(policy, observation, parameters)
    if deterministic:
        log_probability = torch.zeros((len(action), 1), dtype=action.dtype, device=action.device)
    else:
        # SAC's entropy is over the bounded controller parameters, not over a
        # separate motor residual. This is the change-of-variables Jacobian for
        # theta = min + (max - min) * sigmoid(raw_parameter).
        scale = policy.reflex_parameter_max - policy.reflex_parameter_min
        log_jacobian = torch.log(scale) + functional.logsigmoid(raw_parameters) + functional.logsigmoid(-raw_parameters)
        log_probability = (distribution.log_prob(raw_parameters) - log_jacobian).sum(dim=-1, keepdim=True)
    return action, log_std, log_probability, parameters


def reflex_parameter_telemetry(policy: FlyConnectomePolicy, observation: torch.Tensor) -> list[dict[str, Any]]:
    state = _state(policy)
    if state is None:
        return []
    values = bounded_reflex_parameters(policy, observation)
    output = []
    for channel_index, channel in enumerate(state.config.channels):
        channel_name = channel.name or f"channel_{channel_index}"
        for spec in channel.parameters:
            output.append({"name": f"{channel_name}.{spec.name}", "values": values[f"{channel_name}.{spec.name}"].detach().cpu().tolist(), "min_val": spec.min_val, "max_val": spec.max_val, "default": spec.default})
    return output


_original_sample_decomposed = FlyConnectomePolicy.sample_decomposed
_original_forward = FlyConnectomePolicy.forward
_original_load_state_dict = FlyConnectomePolicy.load_state_dict


def sample_decomposed(policy: FlyConnectomePolicy, observation: torch.Tensor, deterministic: bool = False):
    if _state(policy) is None:
        return _original_sample_decomposed(policy, observation, deterministic)
    action, _, log_probability, parameters = _sample_parametric(policy, observation, deterministic)
    if _uses_default_gantry_task_flow(policy):
        action = _apply_gantry_task_flow(observation, action, parameters)
    # The protocol retains the decomposition fields, but Parametric SAC has no
    # direct residual actuator branch. ``fly_base`` is f(x; theta_SAC).
    return action, action, torch.zeros_like(action), log_probability


def forward(policy: FlyConnectomePolicy, observation: torch.Tensor):
    if _state(policy) is None:
        return _original_forward(policy, observation)
    action, log_std, _, parameters = _sample_parametric(policy, observation, deterministic=True)
    if _uses_default_gantry_task_flow(policy):
        action = _apply_gantry_task_flow(observation, action, parameters)
    return action, log_std


def get_extra_state(policy: FlyConnectomePolicy) -> dict[str, Any]:
    state = _state(policy)
    return {"reflex_config": None if state is None else state.config.to_dict()}


def set_extra_state(policy: FlyConnectomePolicy, payload: Any) -> None:
    if payload is None:
        clear_reflex_law(policy)
    elif not isinstance(payload, Mapping) or set(payload) != {"reflex_config"}:
        raise ReflexConfigError("invalid persisted dynamic reflex state")
    elif payload["reflex_config"] is None:
        clear_reflex_law(policy)
    elif get_extra_state(policy)["reflex_config"] != payload["reflex_config"]:
        register_reflex_law(policy, payload["reflex_config"])


def load_state_dict(policy: FlyConnectomePolicy, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
    if "_extra_state" in state_dict:
        set_extra_state(policy, state_dict["_extra_state"])
    return _original_load_state_dict(policy, state_dict, strict=strict, assign=assign)


FlyConnectomePolicy.register_reflex_law = register_reflex_law
FlyConnectomePolicy.register_default_gantry_reflex_law = register_default_gantry_reflex_law
FlyConnectomePolicy.clear_reflex_law = clear_reflex_law
FlyConnectomePolicy.reflex_parameter_count = property(reflex_parameter_count)
FlyConnectomePolicy.bounded_reflex_parameters = bounded_reflex_parameters
FlyConnectomePolicy.reflex_parameter_telemetry = reflex_parameter_telemetry
FlyConnectomePolicy.export_reflex_parameter_telemetry = reflex_parameter_telemetry
FlyConnectomePolicy.sample_decomposed = sample_decomposed
FlyConnectomePolicy.forward = forward
FlyConnectomePolicy.get_extra_state = get_extra_state
FlyConnectomePolicy.set_extra_state = set_extra_state
FlyConnectomePolicy.load_state_dict = load_state_dict
