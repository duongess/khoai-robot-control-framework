"""Configuration for the local SAC learner."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

@dataclass(frozen=True)
class SACConfig:
    state_dim: int
    # In force-control residual mode the three actor outputs are bounded
    # corrections (delta X velocity, delta Y velocity, delta grip setpoint),
    # not direct macro motor commands. Their physical scales live in the Go
    # environment so every actor implementation shares one action contract.
    action_dim: int = 3
    hidden_dim: int = 128
    # The force-control task uses a 0.1 s physics step and contains a 30 s
    # secure-hold lesson. A 0.99 per-step discount has only a ~10 s effective
    # horizon, making a later break penalty almost invisible to SAC.
    gamma: float = 0.999
    tau: float = 0.005
    learning_rate: float = 3e-4
    target_entropy: float = -3.0
    seed: int = 0
    controller_type: str = "mlp"
    graph_path: str | None = None
    propagation_steps: int = 4
    train_edge_gains: bool = True
    freeze_topology: bool = True
    full_actor_unlock_step: int = 128
    # The force-control task has one sparse connectome core, then distinct
    # learned readouts for acquiring an object and transporting it.  This is a
    # phase-conditioned decoder, not two independent policies whose actions
    # would be averaged together.
    phase_gated_decoder: bool = False
    object_attached_observation_index: int = -1
    phase_observation_index: int = -1
    transport_phase_threshold: float = 0.0
    activation: str = "tanh"
    action_dead_zone: float = 0.001
    # Maintain a modest variance floor during stochastic data collection. This
    # does not affect explicit deterministic evaluation requests.
    min_log_std: float = -3.0
    max_horizontal_speed: float = 1.0
    max_vertical_speed: float = 1.0
    max_gripper_command: float = 1.0

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state dimension must be positive")
        if self.action_dim <= 0:
            raise ValueError("action dimension must be positive")
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must be in (0, 1)")
        if self.controller_type not in {"mlp", "fly_connectome", "random_graph"}:
            raise ValueError("controller_type must be mlp, fly_connectome, or random_graph")
        if self.controller_type != "mlp" and not self.graph_path:
            raise ValueError("graph_path is required for graph controllers")
        if self.graph_path and self.controller_type != "mlp" and not Path(self.graph_path).is_file():
            raise ValueError(f"connectome graph does not exist: {self.graph_path}")
        if not self.freeze_topology:
            raise ValueError("freeze_topology must remain true; creating graph edges is unsupported")
        if self.full_actor_unlock_step < 0:
            raise ValueError("full_actor_unlock_step must be non-negative")
        if self.phase_gated_decoder:
            if self.controller_type == "mlp":
                raise ValueError("phase_gated_decoder requires a graph controller")
            indices = (self.object_attached_observation_index, self.phase_observation_index)
            if any(index < 0 or index >= self.state_dim for index in indices):
                raise ValueError("phase-gated decoder observation indices must be within state_dim")
            if self.object_attached_observation_index == self.phase_observation_index:
                raise ValueError("phase-gated decoder observation indices must be distinct")
            if not math.isfinite(self.transport_phase_threshold):
                raise ValueError("transport_phase_threshold must be finite")
        if self.min_log_std > 2:
            raise ValueError("min_log_std must not exceed 2")


@dataclass(frozen=True)
class LearnerConfig:
    # The force-control task emits 30 values. Keep the dataclass default aligned
    # with from_environment() so an in-process servicer cannot start with a
    # different observation schema than the gRPC process.
    state_dim: int = 30
    action_dim: int = 3
    controller_type: str = "mlp"
    graph_path: str | None = None
    propagation_steps: int = 4
    train_edge_gains: bool = True
    freeze_topology: bool = True
    full_actor_unlock_step: int = 128
    phase_gated_decoder: bool = False
    object_attached_observation_index: int = -1
    phase_observation_index: int = -1
    transport_phase_threshold: float = 0.0
    activation: str = "tanh"
    action_dead_zone: float = 0.001
    min_log_std: float = -3.0
    max_horizontal_speed: float = 1.0
    max_vertical_speed: float = 1.0
    max_gripper_command: float = 1.0
    # At the fixed 0.1 s control period, 0.999 keeps meaningful credit for a
    # 30 s secure hold and its possible later safety failure.
    gamma: float = 0.999
    deterministic_inference: bool = False
    # Keep the local learner responsive on development machines. These values
    # control PyTorch compute threads, not the number of simulated workers.
    torch_num_threads: int = 1
    torch_num_interop_threads: int = 1
    # Episodes pin an immutable actor snapshot. Bound their cache so a long
    # training run cannot retain one full model per gradient update forever.
    max_policy_snapshots: int = 128
    # Per-request inference/training logs are expensive at a 20 Hz control
    # rate, especially when the terminal is rendering a browser dashboard.
    log_every_n_requests: int = 100
    # Checkpoints are local training artifacts. The model name supplied on the
    # command line is validated separately and is never treated as a path.
    checkpoint_dir: str = "data/models"

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must be in (0, 1)")
        if self.torch_num_threads <= 0 or self.torch_num_interop_threads <= 0:
            raise ValueError("PyTorch thread counts must be positive")
        if self.max_policy_snapshots < 2:
            raise ValueError("max_policy_snapshots must be at least 2")
        if self.log_every_n_requests <= 0:
            raise ValueError("log_every_n_requests must be positive")
        if not self.checkpoint_dir.strip():
            raise ValueError("checkpoint_dir must not be empty")
        if self.full_actor_unlock_step < 0:
            raise ValueError("full_actor_unlock_step must be non-negative")
        if self.phase_gated_decoder:
            if self.controller_type == "mlp":
                raise ValueError("phase_gated_decoder requires a graph controller")
            indices = (self.object_attached_observation_index, self.phase_observation_index)
            if any(index < 0 or index >= self.state_dim for index in indices):
                raise ValueError("phase-gated decoder observation indices must be within state_dim")
            if self.object_attached_observation_index == self.phase_observation_index:
                raise ValueError("phase-gated decoder observation indices must be distinct")
            if not math.isfinite(self.transport_phase_threshold):
                raise ValueError("transport_phase_threshold must be finite")

    @classmethod
    def from_environment(cls) -> "LearnerConfig":
        try:
            graph_path = os.environ.get("LEARNER_GRAPH_PATH")
            controller_type = os.environ.get("LEARNER_CONTROLLER", "mlp")
            default_phase_gating = "true" if controller_type in {"fly_connectome", "random_graph"} else "false"
            return cls(
                state_dim=int(os.environ.get("LEARNER_STATE_DIM", "30")),
                action_dim=int(os.environ.get("LEARNER_ACTION_DIM", "3")),
                controller_type=controller_type,
                graph_path=graph_path,
                propagation_steps=int(os.environ.get("LEARNER_PROPAGATION_STEPS", "4")),
                train_edge_gains=os.environ.get("LEARNER_TRAIN_EDGE_GAINS", "true").lower() == "true",
                freeze_topology=os.environ.get("LEARNER_FREEZE_TOPOLOGY", "true").lower() == "true",
                full_actor_unlock_step=int(os.environ.get("LEARNER_FULL_ACTOR_UNLOCK_STEP", "128")),
                phase_gated_decoder=os.environ.get("LEARNER_PHASE_GATED_DECODER", default_phase_gating).lower() == "true",
                object_attached_observation_index=int(os.environ.get("LEARNER_OBJECT_ATTACHED_OBSERVATION_INDEX", "15")),
                phase_observation_index=int(os.environ.get("LEARNER_PHASE_OBSERVATION_INDEX", "19")),
                transport_phase_threshold=float(os.environ.get("LEARNER_TRANSPORT_PHASE_THRESHOLD", "0.0")),
                activation=os.environ.get("LEARNER_ACTIVATION", "tanh"),
                action_dead_zone=float(os.environ.get("LEARNER_ACTION_DEAD_ZONE", "0.001")),
                min_log_std=float(os.environ.get("LEARNER_MIN_LOG_STD", "-3.0")),
                max_horizontal_speed=float(os.environ.get("LEARNER_MAX_HORIZONTAL_SPEED", "1.0")),
                max_vertical_speed=float(os.environ.get("LEARNER_MAX_VERTICAL_SPEED", "1.0")),
                max_gripper_command=float(os.environ.get("LEARNER_MAX_GRIPPER_COMMAND", "1.0")),
                gamma=float(os.environ.get("LEARNER_GAMMA", "0.999")),
                deterministic_inference=os.environ.get("LEARNER_DETERMINISTIC_INFERENCE", "false").lower() == "true",
                torch_num_threads=int(os.environ.get("LEARNER_TORCH_NUM_THREADS", "1")),
                torch_num_interop_threads=int(os.environ.get("LEARNER_TORCH_NUM_INTEROP_THREADS", "1")),
                max_policy_snapshots=int(os.environ.get("LEARNER_MAX_POLICY_SNAPSHOTS", "128")),
                log_every_n_requests=int(os.environ.get("LEARNER_LOG_EVERY_N_REQUESTS", "100")),
                checkpoint_dir=os.environ.get("LEARNER_CHECKPOINT_DIR", "data/models"),
            )
        except ValueError as error:
            raise ValueError("learner environment configuration contains an invalid numeric value") from error
