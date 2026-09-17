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
    __slots__ = ("actions", "policy_version")
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    POLICY_VERSION_FIELD_NUMBER: _ClassVar[int]
    actions: _containers.RepeatedCompositeFieldContainer[_environment_pb2.Action]
    policy_version: int
    def __init__(self, actions: _Optional[_Iterable[_Union[_environment_pb2.Action, _Mapping]]] = ..., policy_version: _Optional[int] = ...) -> None: ...

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
    __slots__ = ("ready", "policy_version", "training_step", "device", "model_name")
    READY_FIELD_NUMBER: _ClassVar[int]
    POLICY_VERSION_FIELD_NUMBER: _ClassVar[int]
    TRAINING_STEP_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    policy_version: int
    training_step: int
    device: str
    model_name: str
    def __init__(self, ready: _Optional[bool] = ..., policy_version: _Optional[int] = ..., training_step: _Optional[int] = ..., device: _Optional[str] = ..., model_name: _Optional[str] = ...) -> None: ...

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
