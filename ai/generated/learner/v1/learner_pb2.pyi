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
    policy_version: str
    def __init__(self, environment: _Optional[_Union[_environment_pb2.EnvironmentDescriptor, _Mapping]] = ..., states: _Optional[_Iterable[_Union[_environment_pb2.State, _Mapping]]] = ..., policy_version: _Optional[str] = ...) -> None: ...

class PredictBatchResponse(_message.Message):
    __slots__ = ("actions", "policy_version")
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    POLICY_VERSION_FIELD_NUMBER: _ClassVar[int]
    actions: _containers.RepeatedCompositeFieldContainer[_environment_pb2.Action]
    policy_version: str
    def __init__(self, actions: _Optional[_Iterable[_Union[_environment_pb2.Action, _Mapping]]] = ..., policy_version: _Optional[str] = ...) -> None: ...

class TrainBatchRequest(_message.Message):
    __slots__ = ("batch", "policy_version")
    BATCH_FIELD_NUMBER: _ClassVar[int]
    POLICY_VERSION_FIELD_NUMBER: _ClassVar[int]
    batch: _transition_pb2.TransitionBatch
    policy_version: str
    def __init__(self, batch: _Optional[_Union[_transition_pb2.TransitionBatch, _Mapping]] = ..., policy_version: _Optional[str] = ...) -> None: ...

class TrainBatchResponse(_message.Message):
    __slots__ = ("accepted", "samples_seen", "policy_version")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    SAMPLES_SEEN_FIELD_NUMBER: _ClassVar[int]
    POLICY_VERSION_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    samples_seen: int
    policy_version: str
    def __init__(self, accepted: _Optional[bool] = ..., samples_seen: _Optional[int] = ..., policy_version: _Optional[str] = ...) -> None: ...
