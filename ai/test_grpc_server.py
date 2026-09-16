import math

import pytest
import torch

from ai.config import LearnerConfig
from ai.grpc_server import LearnerServicer
from learner.v1 import environment_pb2, learner_pb2, transition_pb2


class AbortContext:
    def abort(self, _code, message):
        raise ValueError(message)


def test_health_and_batched_prediction_use_configured_dimensions():
    servicer = LearnerServicer(LearnerConfig(state_dim=3, action_dim=3))
    response = servicer.PredictBatch(learner_pb2.PredictBatchRequest(states=[environment_pb2.State(values=[0.0, 0.1, 0.2]), environment_pb2.State(values=[0.3, 0.4, 0.5])]), AbortContext())
    assert len(response.actions) == 2
    assert all(len(action.values) == 3 for action in response.actions)
    assert all(-1 <= value <= 1 for action in response.actions for value in action.values)
    assert servicer.HealthCheck(learner_pb2.HealthCheckRequest(), AbortContext()).ready


def test_prediction_rejects_invalid_and_non_finite_states():
    servicer = LearnerServicer(LearnerConfig(state_dim=3, action_dim=3))
    with pytest.raises(ValueError, match="dimension"):
        servicer.PredictBatch(learner_pb2.PredictBatchRequest(states=[environment_pb2.State(values=[0.0])]), AbortContext())
    with pytest.raises(ValueError, match="finite"):
        servicer.PredictBatch(learner_pb2.PredictBatchRequest(states=[environment_pb2.State(values=[math.nan, 0.0, 0.0])]), AbortContext())


@pytest.mark.parametrize("deterministic", [False, True])
def test_prediction_uses_explicit_collection_or_evaluation_mode(deterministic: bool):
    servicer = LearnerServicer(LearnerConfig(state_dim=3, action_dim=3, deterministic_inference=deterministic))
    called_with: list[bool] = []

    def act_with_actor(_actor, states, *, deterministic: bool):
        called_with.append(deterministic)
        return torch.zeros((len(states), 3))

    servicer._agent.act_with_actor = act_with_actor  # type: ignore[method-assign]
    servicer.PredictBatch(
        learner_pb2.PredictBatchRequest(states=[environment_pb2.State(values=[0.0, 0.1, 0.2])]),
        AbortContext(),
    )
    assert called_with == [deterministic]


def test_training_increments_policy_version():
    servicer = LearnerServicer(LearnerConfig(state_dim=3, action_dim=3))
    transitions = [transition_pb2.Transition(state=environment_pb2.State(values=[0.0, 0.1, 0.2]), action=environment_pb2.Action(values=[0.0, 0.0, 0.0]), reward=1.0, next_state=environment_pb2.State(values=[0.1, 0.2, 0.3])) for _ in range(4)]
    result = servicer.TrainBatch(learner_pb2.TrainBatchRequest(batch=transition_pb2.TransitionBatch(transitions=transitions)), AbortContext())
    assert result.accepted
    assert result.policy_version == 2
    assert result.training_step == 1


def test_policy_snapshot_remains_available_after_training():
    servicer = LearnerServicer(LearnerConfig(state_dim=3, action_dim=3))
    request = learner_pb2.PredictBatchRequest(states=[environment_pb2.State(values=[0.0, 0.1, 0.2])])
    initial = servicer.PredictBatch(request, AbortContext())
    assert initial.policy_version == 1
    transitions = [transition_pb2.Transition(state=environment_pb2.State(values=[0.0, 0.1, 0.2]), action=environment_pb2.Action(values=[0.0, 0.0, 0.0]), reward=1.0, next_state=environment_pb2.State(values=[0.1, 0.2, 0.3])) for _ in range(4)]
    trained = servicer.TrainBatch(learner_pb2.TrainBatchRequest(batch=transition_pb2.TransitionBatch(transitions=transitions)), AbortContext())
    assert trained.policy_version == 2
    pinned = servicer.PredictBatch(
        learner_pb2.PredictBatchRequest(states=[environment_pb2.State(values=[0.0, 0.1, 0.2])], policy_version=1),
        AbortContext(),
    )
    assert pinned.policy_version == 1


def test_policy_snapshot_cache_is_bounded():
    servicer = LearnerServicer(LearnerConfig(state_dim=3, action_dim=3, max_policy_snapshots=2))
    transitions = [transition_pb2.Transition(state=environment_pb2.State(values=[0.0, 0.1, 0.2]), action=environment_pb2.Action(values=[0.0, 0.0, 0.0]), reward=1.0, next_state=environment_pb2.State(values=[0.1, 0.2, 0.3])) for _ in range(4)]
    request = learner_pb2.TrainBatchRequest(batch=transition_pb2.TransitionBatch(transitions=transitions))
    servicer.TrainBatch(request, AbortContext())
    servicer.TrainBatch(request, AbortContext())
    servicer.TrainBatch(request, AbortContext())
    assert len(servicer._actor_snapshots) == 2
    assert set(servicer._actor_snapshots) == {3, 4}
