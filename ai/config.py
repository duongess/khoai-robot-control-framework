"""Configuration for the local SAC learner."""

from dataclasses import dataclass
import os


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

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state dimension must be positive")
        if self.action_dim <= 0:
            raise ValueError("action dimension must be positive")


@dataclass(frozen=True)
class LearnerConfig:
    state_dim: int = 19
    action_dim: int = 3

    @classmethod
    def from_environment(cls) -> "LearnerConfig":
        try:
            return cls(state_dim=int(os.environ.get("LEARNER_STATE_DIM", "19")), action_dim=int(os.environ.get("LEARNER_ACTION_DIM", "3")))
        except ValueError as error:
            raise ValueError("LEARNER_STATE_DIM and LEARNER_ACTION_DIM must be integers") from error
