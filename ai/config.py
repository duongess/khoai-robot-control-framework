"""Configuration for the local SAC learner."""

from dataclasses import dataclass
import os
from pathlib import Path

@dataclass(frozen=True)
class SACConfig:
    state_dim: int
    action_dim: int = 3
    hidden_dim: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    target_entropy: float = -3.0
    seed: int = 0
    controller_type: str = "mlp"
    graph_path: str | None = None
    propagation_steps: int = 4
    train_edge_gains: bool = True
    freeze_topology: bool = True
    activation: str = "tanh"
    action_dead_zone: float = 0.1
    max_horizontal_speed: float = 1.0
    max_vertical_speed: float = 1.0
    max_gripper_command: float = 1.0

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state dimension must be positive")
        if self.action_dim <= 0:
            raise ValueError("action dimension must be positive")
        if self.controller_type not in {"mlp", "fly_connectome", "random_graph"}:
            raise ValueError("controller_type must be mlp, fly_connectome, or random_graph")
        if self.controller_type != "mlp" and not self.graph_path:
            raise ValueError("graph_path is required for graph controllers")
        if self.graph_path and self.controller_type != "mlp" and not Path(self.graph_path).is_file():
            raise ValueError(f"connectome graph does not exist: {self.graph_path}")
        if not self.freeze_topology:
            raise ValueError("freeze_topology must remain true; creating graph edges is unsupported")


@dataclass(frozen=True)
class LearnerConfig:
    # The force-control task emits 22 values. Keep the dataclass default aligned
    # with from_environment() so an in-process servicer cannot start with a
    # different observation schema than the gRPC process.
    state_dim: int = 22
    action_dim: int = 3
    controller_type: str = "mlp"
    graph_path: str | None = None
    propagation_steps: int = 4
    train_edge_gains: bool = True
    freeze_topology: bool = True
    activation: str = "tanh"
    action_dead_zone: float = 0.1
    max_horizontal_speed: float = 1.0
    max_vertical_speed: float = 1.0
    max_gripper_command: float = 1.0
    deterministic_inference: bool = False

    @classmethod
    def from_environment(cls) -> "LearnerConfig":
        try:
            graph_path = os.environ.get("LEARNER_GRAPH_PATH")
            return cls(
                state_dim=int(os.environ.get("LEARNER_STATE_DIM", "22")),
                action_dim=int(os.environ.get("LEARNER_ACTION_DIM", "3")),
                controller_type=os.environ.get("LEARNER_CONTROLLER", "mlp"),
                graph_path=graph_path,
                propagation_steps=int(os.environ.get("LEARNER_PROPAGATION_STEPS", "4")),
                train_edge_gains=os.environ.get("LEARNER_TRAIN_EDGE_GAINS", "true").lower() == "true",
                freeze_topology=os.environ.get("LEARNER_FREEZE_TOPOLOGY", "true").lower() == "true",
                activation=os.environ.get("LEARNER_ACTIVATION", "tanh"),
                action_dead_zone=float(os.environ.get("LEARNER_ACTION_DEAD_ZONE", "0.1")),
                max_horizontal_speed=float(os.environ.get("LEARNER_MAX_HORIZONTAL_SPEED", "1.0")),
                max_vertical_speed=float(os.environ.get("LEARNER_MAX_VERTICAL_SPEED", "1.0")),
                max_gripper_command=float(os.environ.get("LEARNER_MAX_GRIPPER_COMMAND", "1.0")),
                deterministic_inference=os.environ.get("LEARNER_DETERMINISTIC_INFERENCE", "false").lower() == "true",
            )
        except ValueError as error:
            raise ValueError("LEARNER_STATE_DIM and LEARNER_ACTION_DIM must be integers") from error
