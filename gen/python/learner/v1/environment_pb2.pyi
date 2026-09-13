from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class State(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ...) -> None: ...

class EnvironmentDescriptor(_message.Message):
    __slots__ = ("name", "version", "state_dimension", "action_dimension", "continuous_actions")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    STATE_DIMENSION_FIELD_NUMBER: _ClassVar[int]
    ACTION_DIMENSION_FIELD_NUMBER: _ClassVar[int]
    CONTINUOUS_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    state_dimension: int
    action_dimension: int
    continuous_actions: bool
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., state_dimension: _Optional[int] = ..., action_dimension: _Optional[int] = ..., continuous_actions: _Optional[bool] = ...) -> None: ...
