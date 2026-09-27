"""Validated, differentiable reflex laws configured by a client payload.

The module deliberately does not use ``eval``.  Custom formulae are parsed as
Python expressions and interpreted from a small AST allow-list into PyTorch
operations, keeping both the attack surface and the available math explicit.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


class ReflexConfigError(ValueError):
    """A web-client reflex configuration is malformed or unsafe."""


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    min_val: float
    max_val: float
    default: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.isidentifier() or self.name.startswith("_"):
            raise ReflexConfigError("parameter names must be public Python identifiers")
        if not all(math.isfinite(value) for value in (self.min_val, self.max_val, self.default)):
            raise ReflexConfigError(f"parameter {self.name!r} bounds and default must be finite")
        if self.min_val >= self.max_val:
            raise ReflexConfigError(f"parameter {self.name!r} must have min_val less than max_val")
        if not self.min_val <= self.default <= self.max_val:
            raise ReflexConfigError(f"parameter {self.name!r} default must be within its bounds")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParameterSpec":
        _require_mapping(payload, "parameter")
        _reject_unknown(payload, {"name", "min_val", "max_val", "default"}, "parameter")
        try:
            return cls(
                name=str(payload["name"]),
                min_val=float(payload["min_val"]),
                max_val=float(payload["max_val"]),
                default=float(payload.get("default", 1.0)),
            )
        except KeyError as error:
            raise ReflexConfigError(f"parameter is missing {error.args[0]!r}") from error
        except (TypeError, ValueError) as error:
            raise ReflexConfigError("parameter fields must have valid scalar values") from error

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "min_val": self.min_val, "max_val": self.max_val, "default": self.default}


@dataclass(frozen=True)
class ReflexChannelConfig:
    """One action channel and the observation symbols made available to it.

    ``signal_indices`` maps formula names (for example ``err`` and ``vel``) to
    observation columns.  Template laws use sensible defaults when a mapping is
    absent: ``x``/``err`` use the channel's action index and ``vel`` is zero.
    """

    expression_type: str
    formula_str: str | None = None
    parameters: tuple[ParameterSpec, ...] = field(default_factory=tuple)
    name: str | None = None
    action_index: int | None = None
    signal_indices: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.expression_type not in DynamicReflexEngine.SUPPORTED_TYPES:
            raise ReflexConfigError(f"unsupported expression_type {self.expression_type!r}")
        if self.expression_type == "custom_eval":
            if not isinstance(self.formula_str, str) or not self.formula_str.strip():
                raise ReflexConfigError("custom_eval requires a non-empty formula_str")
        elif self.formula_str is not None and not isinstance(self.formula_str, str):
            raise ReflexConfigError("formula_str must be a string when supplied")
        if self.name is not None and (not self.name or not self.name.replace("_", "a").isalnum()):
            raise ReflexConfigError("channel name may contain only letters, digits, and underscores")
        if self.action_index is not None and self.action_index < 0:
            raise ReflexConfigError("action_index must be non-negative")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ReflexConfigError("parameter names must be unique within a channel")
        for signal, index in self.signal_indices.items():
            if not signal.isidentifier() or signal.startswith("_") or not isinstance(index, int) or index < 0:
                raise ReflexConfigError("signal_indices must map public names to non-negative integer columns")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReflexChannelConfig":
        _require_mapping(payload, "reflex channel")
        allowed = {"expression_type", "formula_str", "parameters", "name", "action_index", "signal_indices", "inputs"}
        _reject_unknown(payload, allowed, "reflex channel")
        raw_signals = payload.get("signal_indices", payload.get("inputs", {}))
        if not isinstance(raw_signals, Mapping):
            raise ReflexConfigError("signal_indices must be an object")
        raw_parameters = payload.get("parameters", [])
        if not isinstance(raw_parameters, list):
            raise ReflexConfigError("parameters must be a list")
        try:
            return cls(
                expression_type=str(payload["expression_type"]),
                formula_str=payload.get("formula_str"),
                parameters=tuple(ParameterSpec.from_dict(item) for item in raw_parameters),
                name=payload.get("name"),
                action_index=payload.get("action_index"),
                signal_indices={str(key): value for key, value in raw_signals.items()},
            )
        except KeyError as error:
            raise ReflexConfigError(f"reflex channel is missing {error.args[0]!r}") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression_type": self.expression_type,
            "formula_str": self.formula_str,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "name": self.name,
            "action_index": self.action_index,
            "signal_indices": dict(self.signal_indices),
        }


@dataclass(frozen=True)
class ReflexFunctionConfig:
    channels: tuple[ReflexChannelConfig, ...]

    def __post_init__(self) -> None:
        if not self.channels:
            raise ReflexConfigError("a reflex configuration must contain at least one channel")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReflexFunctionConfig":
        _require_mapping(payload, "reflex configuration")
        _reject_unknown(payload, {"channels"}, "reflex configuration")
        channels = payload.get("channels")
        if not isinstance(channels, list):
            raise ReflexConfigError("reflex configuration channels must be a list")
        return cls(tuple(ReflexChannelConfig.from_dict(item) for item in channels))

    def to_dict(self) -> dict[str, Any]:
        return {"channels": [channel.to_dict() for channel in self.channels]}

    @property
    def parameter_count(self) -> int:
        return sum(len(channel.parameters) for channel in self.channels)


class DynamicReflexEngine:
    """Compile and evaluate template or safely parsed custom reflex laws."""

    SUPPORTED_TYPES = frozenset({"linear", "saturated_linear", "impedance_pd", "custom_eval"})
    _TEMPLATES = {
        "linear": "-a * x + b",
        "saturated_linear": "c * tanh(-a * x + b)",
        "impedance_pd": "-kp * err - kd * vel + bias",
    }
    _FUNCTIONS = frozenset({"tanh", "clamp", "abs"})
    _MAX_FORMULA_LENGTH = 512
    _MAX_AST_NODES = 96

    def __init__(self, config: ReflexFunctionConfig) -> None:
        self.config = config
        self._trees = tuple(self._parse(channel) for channel in config.channels)

    @staticmethod
    def formula_for(channel: ReflexChannelConfig) -> str:
        return channel.formula_str if channel.expression_type == "custom_eval" else DynamicReflexEngine._TEMPLATES[channel.expression_type]

    def _parse(self, channel: ReflexChannelConfig) -> ast.Expression:
        formula = self.formula_for(channel)
        if len(formula) > self._MAX_FORMULA_LENGTH:
            raise ReflexConfigError("formula_str exceeds the 512-character limit")
        try:
            parsed = ast.parse(formula, mode="eval")
        except SyntaxError as error:
            raise ReflexConfigError("formula_str is not a valid expression") from error
        if sum(1 for _ in ast.walk(parsed)) > self._MAX_AST_NODES:
            raise ReflexConfigError("formula_str is too complex")
        self._validate_ast(parsed)
        parameter_names = {parameter.name for parameter in channel.parameters}
        variables = {node.id for node in ast.walk(parsed) if isinstance(node, ast.Name)} - self._FUNCTIONS
        missing = parameter_names - variables
        if missing:
            raise ReflexConfigError(f"formula does not use declared parameter(s): {', '.join(sorted(missing))}")
        unknown = variables - parameter_names - set(channel.signal_indices) - {"x", "err", "error", "vel"}
        if unknown:
            raise ReflexConfigError(f"formula contains undeclared signal(s): {', '.join(sorted(unknown))}")
        return parsed

    def _validate_ast(self, node: ast.AST) -> None:
        for current in ast.walk(node):
            if isinstance(current, ast.Expression | ast.Load | ast.BinOp | ast.UnaryOp | ast.Call | ast.Name | ast.Constant):
                continue
            if isinstance(current, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)):
                continue
            raise ReflexConfigError(f"unsafe formula syntax: {type(current).__name__}")
        for current in ast.walk(node):
            if isinstance(current, ast.Constant):
                if not isinstance(current.value, (int, float)) or isinstance(current.value, bool) or not math.isfinite(float(current.value)):
                    raise ReflexConfigError("formula constants must be finite numbers")
            elif isinstance(current, ast.Name) and (not current.id.isidentifier() or current.id.startswith("_")):
                raise ReflexConfigError("formula names must be public identifiers")
            elif isinstance(current, ast.Call):
                if not isinstance(current.func, ast.Name) or current.func.id not in self._FUNCTIONS or current.keywords:
                    raise ReflexConfigError("only tanh, abs, and clamp positional calls are allowed")
                if current.func.id in {"tanh", "abs"} and len(current.args) != 1:
                    raise ReflexConfigError(f"{current.func.id} accepts exactly one argument")
                if current.func.id == "clamp" and len(current.args) not in {1, 3}:
                    raise ReflexConfigError("clamp accepts value or value, min, max")

    def evaluate(self, signals: Mapping[str, torch.Tensor], parameters: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate all configured channels and return ``(batch, channels)``."""
        outputs = []
        for index, (channel, tree) in enumerate(zip(self.config.channels, self._trees, strict=True)):
            channel_parameters = {spec.name: parameters[self.parameter_key(index, spec.name)] for spec in channel.parameters}
            environment = {**signals, **channel_parameters}
            try:
                value = self._evaluate_node(tree.body, environment)
            except KeyError as error:
                raise ReflexConfigError(f"formula requires unavailable signal {error.args[0]!r}") from error
            if not isinstance(value, torch.Tensor):
                reference = next(iter(channel_parameters.values()), next(iter(signals.values()), None))
                if reference is None:
                    raise ReflexConfigError("a formula requires at least one tensor signal or parameter")
                value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device).expand_as(reference)
            if value.ndim != 1:
                raise ReflexConfigError("a reflex formula must evaluate to one value per batch item")
            if not torch.isfinite(value).all():
                raise RuntimeError("dynamic reflex produced non-finite values")
            outputs.append(value)
        return torch.stack(outputs, dim=1)

    @staticmethod
    def parameter_key(channel_index: int, parameter_name: str) -> str:
        return f"{channel_index}.{parameter_name}"

    def _evaluate_node(self, node: ast.AST, environment: Mapping[str, torch.Tensor]) -> torch.Tensor | float:
        if isinstance(node, ast.Name):
            return environment[node.id]
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.UnaryOp):
            operand = self._evaluate_node(node.operand, environment)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp):
            left, right = self._evaluate_node(node.left, environment), self._evaluate_node(node.right, environment)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        if isinstance(node, ast.Call):
            values = [self._evaluate_node(argument, environment) for argument in node.args]
            name = node.func.id  # validated above
            if name == "tanh":
                return torch.tanh(values[0]) if isinstance(values[0], torch.Tensor) else math.tanh(values[0])
            if name == "abs":
                return torch.abs(values[0]) if isinstance(values[0], torch.Tensor) else abs(values[0])
            if len(values) == 1:
                return torch.clamp(values[0], -1.0, 1.0) if isinstance(values[0], torch.Tensor) else max(-1.0, min(1.0, values[0]))
            return torch.clamp(values[0], min=values[1], max=values[2])
        raise ReflexConfigError("unsafe formula syntax")


def _require_mapping(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ReflexConfigError(f"{name} must be a JSON object")


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ReflexConfigError(f"{name} contains unsupported field(s): {', '.join(sorted(unknown))}")
