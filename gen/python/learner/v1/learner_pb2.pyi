from learner.v1 import environment_pb2 as _environment_pb2
from learner.v1 import transition_pb2 as _transition_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PredictBatchRequest(_message.Message):
    __slots__ = ("environment", "states", "policy_version")
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    STATES_FIELD_NUMBER: _ClassVar[int]
    POLICY_VERSION_FIELD_NUMBER: _ClassVar[int]
    environment: _environment_pb2.EnvironmentDescriptor
    states: _containers.RepeatedCompositeFieldContainer[_environment_pb2.State]
    policy_version: int
    def __init__(self, environment: _Optional[_Union[_environment_pb2.EnvironmentDescriptor, _Mapping]] = ..., states: _Optional[_Iterable[_Union[_environment_pb2.State, _Mapping]]] = ..., policy_version: _Optional[int] = ...) -> None: ...

class PredictBatchResponse(_message.Message):
    __slots__ = ("actions", "policy_version", "fly_base_actions", "residual_actions")
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    POLICY_VERSION_FIELD_NUMBER: _ClassVar[int]
    FLY_BASE_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    RESIDUAL_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    actions: _containers.RepeatedCompositeFieldContainer[_environment_pb2.Action]
    policy_version: int
    fly_base_actions: _containers.RepeatedCompositeFieldContainer[_environment_pb2.Action]
    residual_actions: _containers.RepeatedCompositeFieldContainer[_environment_pb2.Action]
    def __init__(self, actions: _Optional[_Iterable[_Union[_environment_pb2.Action, _Mapping]]] = ..., policy_version: _Optional[int] = ..., fly_base_actions: _Optional[_Iterable[_Union[_environment_pb2.Action, _Mapping]]] = ..., residual_actions: _Optional[_Iterable[_Union[_environment_pb2.Action, _Mapping]]] = ...) -> None: ...

class TrainBatchRequest(_message.Message):
    __slots__ = ("batch", "policy_version")
    BATCH_FIELD_NUMBER: _ClassVar[int]
    POLICY_VERSION_FIELD_NUMBER: _ClassVar[int]
    batch: _transition_pb2.TransitionBatch
    policy_version: int
    def __init__(self, batch: _Optional[_Union[_transition_pb2.TransitionBatch, _Mapping]] = ..., policy_version: _Optional[int] = ...) -> None: ...

class TrainBatchResponse(_message.Message):
    __slots__ = ("accepted", "samples_seen", "policy_version", "actor_loss", "critic_loss", "alpha_loss", "entropy", "training_step", "critic_one_q", "critic_two_q", "alpha", "actor_log_std_horizontal", "actor_log_std_vertical", "actor_log_std_gripper")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    SAMPLES_SEEN_FIELD_NUMBER: _ClassVar[int]
    POLICY_VERSION_FIELD_NUMBER: _ClassVar[int]
    ACTOR_LOSS_FIELD_NUMBER: _ClassVar[int]
    CRITIC_LOSS_FIELD_NUMBER: _ClassVar[int]
    ALPHA_LOSS_FIELD_NUMBER: _ClassVar[int]
    ENTROPY_FIELD_NUMBER: _ClassVar[int]
    TRAINING_STEP_FIELD_NUMBER: _ClassVar[int]
    CRITIC_ONE_Q_FIELD_NUMBER: _ClassVar[int]
    CRITIC_TWO_Q_FIELD_NUMBER: _ClassVar[int]
    ALPHA_FIELD_NUMBER: _ClassVar[int]
    ACTOR_LOG_STD_HORIZONTAL_FIELD_NUMBER: _ClassVar[int]
    ACTOR_LOG_STD_VERTICAL_FIELD_NUMBER: _ClassVar[int]
    ACTOR_LOG_STD_GRIPPER_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    samples_seen: int
    policy_version: int
    actor_loss: float
    critic_loss: float
    alpha_loss: float
    entropy: float
    training_step: int
    critic_one_q: float
    critic_two_q: float
    alpha: float
    actor_log_std_horizontal: float
    actor_log_std_vertical: float
    actor_log_std_gripper: float
    def __init__(self, accepted: _Optional[bool] = ..., samples_seen: _Optional[int] = ..., policy_version: _Optional[int] = ..., actor_loss: _Optional[float] = ..., critic_loss: _Optional[float] = ..., alpha_loss: _Optional[float] = ..., entropy: _Optional[float] = ..., training_step: _Optional[int] = ..., critic_one_q: _Optional[float] = ..., critic_two_q: _Optional[float] = ..., alpha: _Optional[float] = ..., actor_log_std_horizontal: _Optional[float] = ..., actor_log_std_vertical: _Optional[float] = ..., actor_log_std_gripper: _Optional[float] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("ready", "policy_version", "training_step", "device", "model_name", "controller_type", "active_model_name")
    READY_FIELD_NUMBER: _ClassVar[int]
    POLICY_VERSION_FIELD_NUMBER: _ClassVar[int]
    TRAINING_STEP_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    CONTROLLER_TYPE_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    policy_version: int
    training_step: int
    device: str
    model_name: str
    controller_type: str
    active_model_name: str
    def __init__(self, ready: _Optional[bool] = ..., policy_version: _Optional[int] = ..., training_step: _Optional[int] = ..., device: _Optional[str] = ..., model_name: _Optional[str] = ..., controller_type: _Optional[str] = ..., active_model_name: _Optional[str] = ...) -> None: ...

class SaveCheckpointRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SaveCheckpointResponse(_message.Message):
    __slots__ = ("model_name", "policy_version", "training_step")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    POLICY_VERSION_FIELD_NUMBER: _ClassVar[int]
    TRAINING_STEP_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    policy_version: int
    training_step: int
    def __init__(self, model_name: _Optional[str] = ..., policy_version: _Optional[int] = ..., training_step: _Optional[int] = ...) -> None: ...

class RecordEpisodeResultRequest(_message.Message):
    __slots__ = ("episode_id", "success", "episode_reward")
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    EPISODE_REWARD_FIELD_NUMBER: _ClassVar[int]
    episode_id: str
    success: bool
    episode_reward: float
    def __init__(self, episode_id: _Optional[str] = ..., success: _Optional[bool] = ..., episode_reward: _Optional[float] = ...) -> None: ...

class RecordEpisodeResultResponse(_message.Message):
    __slots__ = ("stop_training", "completed_episodes", "rolling_success_rate", "consecutive_successes", "reason")
    STOP_TRAINING_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_EPISODES_FIELD_NUMBER: _ClassVar[int]
    ROLLING_SUCCESS_RATE_FIELD_NUMBER: _ClassVar[int]
    CONSECUTIVE_SUCCESSES_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    stop_training: bool
    completed_episodes: int
    rolling_success_rate: float
    consecutive_successes: int
    reason: str
    def __init__(self, stop_training: _Optional[bool] = ..., completed_episodes: _Optional[int] = ..., rolling_success_rate: _Optional[float] = ..., consecutive_successes: _Optional[int] = ..., reason: _Optional[str] = ...) -> None: ...
