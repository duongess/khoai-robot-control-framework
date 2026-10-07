import math
from pathlib import Path

import pytest
import torch

from ai.config import LearnerConfig
from ai.checkpoints import validate_model_name
from ai.grpc_server import LearnerServicer
from ai.test_connectome import _graph
from gen.python.learner.v1 import environment_pb2, learner_pb2, transition_pb2


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


def test_learner_forwards_configured_long_horizon_discount():
    servicer = LearnerServicer(LearnerConfig(state_dim=3, action_dim=3, gamma=0.998))
    assert servicer._agent.config.gamma == pytest.approx(0.998)


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

    def act_with_actor_components(_actor, states, *, deterministic: bool):
        called_with.append(deterministic)
        actions = torch.zeros((len(states), 3))
        return actions, actions, actions

    servicer._agent.act_with_actor_components = act_with_actor_components  # type: ignore[method-assign]
    response = servicer.PredictBatch(
        learner_pb2.PredictBatchRequest(states=[environment_pb2.State(values=[0.0, 0.1, 0.2])]),
        AbortContext(),
    )
    assert called_with == [deterministic]
    assert len(response.actions) == len(response.fly_base_actions) == len(response.residual_actions) == 1


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


def test_checkpoint_round_trip_restores_complete_sac_training_state(tmp_path: Path):
    config = LearnerConfig(state_dim=3, action_dim=3, checkpoint_dir=str(tmp_path))
    servicer = LearnerServicer(config, model_name="grasp-v1")
    transitions = [transition_pb2.Transition(
        state=environment_pb2.State(values=[0.0, 0.1, 0.2]),
        action=environment_pb2.Action(values=[0.0, 0.0, 0.0]),
        reward=1.0,
        next_state=environment_pb2.State(values=[0.1, 0.2, 0.3]),
    ) for _ in range(4)]
    servicer.TrainBatch(learner_pb2.TrainBatchRequest(batch=transition_pb2.TransitionBatch(transitions=transitions)), AbortContext())
    saved_actor = {name: value.detach().clone() for name, value in servicer._agent.actor.state_dict().items()}

    saved = servicer.SaveCheckpoint(learner_pb2.SaveCheckpointRequest(), AbortContext())
    restored = LearnerServicer(config, model_name="grasp-v1")

    assert saved.model_name == "grasp-v1"
    assert (tmp_path / "grasp-v1.pt").is_file()
    assert restored._policy_version == servicer._policy_version
    assert restored._training_step == servicer._training_step
    assert restored._samples_seen == servicer._samples_seen
    assert all(torch.equal(value, restored._agent.actor.state_dict()[name]) for name, value in saved_actor.items())
    assert restored._agent.alpha.item() == pytest.approx(servicer._agent.alpha.item())


@pytest.mark.parametrize("name", ["../escape", "", "has space", "/tmp/model"])
def test_checkpoint_model_name_rejects_paths_and_unsafe_names(name: str):
    with pytest.raises(ValueError):
        validate_model_name(name)


def test_best_checkpoint_and_early_stopping_callback(tmp_path: Path):
    config = LearnerConfig(state_dim=3, action_dim=3, checkpoint_dir=str(tmp_path))
    servicer = LearnerServicer(config, model_name="grasp-v1")

    servicer.record_episode_result(True, 5.0)
    best_path = tmp_path / "best_model_checkpoint.pt"
    assert best_path.is_file()
    assert servicer._best_success_rate == pytest.approx(1.0)
    assert servicer._best_average_reward == pytest.approx(5.0)

    for _ in range(5):
        servicer.record_episode_result(True, 5.0)
    assert servicer._early_stopped
    assert servicer._high_success_streak >= 5

    previous_step = servicer._training_step
    transitions = [transition_pb2.Transition(
        state=environment_pb2.State(values=[0.0, 0.1, 0.2]),
        action=environment_pb2.Action(values=[0.0, 0.0, 0.0]),
        reward=1.0,
        next_state=environment_pb2.State(values=[0.1, 0.2, 0.3]),
    ) for _ in range(4)]
    result = servicer.TrainBatch(learner_pb2.TrainBatchRequest(batch=transition_pb2.TransitionBatch(transitions=transitions)), AbortContext())
    assert result.training_step == previous_step


def test_eval_mode_skips_training_and_loads_best_checkpoint(tmp_path: Path):
    config = LearnerConfig(state_dim=3, action_dim=3, checkpoint_dir=str(tmp_path), eval_mode=True)
    checkpoint_path = tmp_path / "best_model_checkpoint.pt"
    checkpoint_path.write_bytes(b"")

    servicer = LearnerServicer(config)
    assert servicer._eval_mode is True
    assert servicer._config.deterministic_inference is True

    previous_step = servicer._training_step
    transitions = [transition_pb2.Transition(
        state=environment_pb2.State(values=[0.0, 0.1, 0.2]),
        action=environment_pb2.Action(values=[0.0, 0.0, 0.0]),
        reward=1.0,
        next_state=environment_pb2.State(values=[0.1, 0.2, 0.3]),
    ) for _ in range(4)]
    result = servicer.TrainBatch(learner_pb2.TrainBatchRequest(batch=transition_pb2.TransitionBatch(transitions=transitions)), AbortContext())
    assert result.accepted is True
    assert result.training_step == previous_step
    assert servicer._agent.actor.training is False


def test_two_stage_training_controller_keeps_connectome_trainable_until_real_thresholds(tmp_path: Path):
    graph = _graph(tmp_path)
    graph_dir = tmp_path / "connectome_cache"
    graph.save(graph_dir)
    config = LearnerConfig(
        state_dim=30,
        action_dim=3,
        controller_type="fly_connectome",
        graph_path=str(graph_dir / "connectome_graph.npz"),
        checkpoint_dir=str(tmp_path),
    )
    servicer = LearnerServicer(config)
    for success in [True] * 4 + [False] + [True] * 4:
        servicer.record_episode_result(success, 1.0)

    assert not servicer._stage1_frozen
    assert all(parameter.requires_grad for parameter in servicer._agent.actor.sensory_encoder.parameters())
    assert all(parameter.requires_grad for parameter in servicer._agent.actor.base_head.parameters())
    assert servicer._agent.actor.edge_log_gains.requires_grad

    deployment_servicer = LearnerServicer(config)
    for _ in range(5):
        deployment_servicer.record_episode_result(True, 1.0)
    assert deployment_servicer._early_stopped
    assert deployment_servicer._config.deterministic_inference is True
    assert (tmp_path / "best_production_checkpoint.pt").is_file()
