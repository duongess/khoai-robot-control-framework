from learner.v1 import environment_pb2 as _environment_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Transition(_message.Message):
    __slots__ = ("state", "action", "reward", "next_state", "terminated", "truncated", "episode_id", "step")
    STATE_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    NEXT_STATE_FIELD_NUMBER: _ClassVar[int]
    TERMINATED_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    state: _environment_pb2.State
    action: _environment_pb2.Action
    reward: float
    next_state: _environment_pb2.State
    terminated: bool
    truncated: bool
    episode_id: str
    step: int
    def __init__(self, state: _Optional[_Union[_environment_pb2.State, _Mapping]] = ..., action: _Optional[_Union[_environment_pb2.Action, _Mapping]] = ..., reward: _Optional[float] = ..., next_state: _Optional[_Union[_environment_pb2.State, _Mapping]] = ..., terminated: _Optional[bool] = ..., truncated: _Optional[bool] = ..., episode_id: _Optional[str] = ..., step: _Optional[int] = ...) -> None: ...

class TransitionBatch(_message.Message):
    __slots__ = ("transitions", "environment")
    TRANSITIONS_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    transitions: _containers.RepeatedCompositeFieldContainer[Transition]
    environment: _environment_pb2.EnvironmentDescriptor
    def __init__(self, transitions: _Optional[_Iterable[_Union[Transition, _Mapping]]] = ..., environment: _Optional[_Union[_environment_pb2.EnvironmentDescriptor, _Mapping]] = ...) -> None: ...
