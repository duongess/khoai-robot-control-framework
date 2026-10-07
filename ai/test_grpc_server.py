import math
from pathlib import Path

import pytest
import torch

from ai.config import LearnerConfig
from ai.checkpoints import validate_model_name
from ai.grpc_server import EARLY_STOP_MESSAGE, LearnerServicer
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


def test_best_checkpoint_uses_complete_50_episode_window_and_new_high(tmp_path: Path):
    config = LearnerConfig(state_dim=3, action_dim=3, checkpoint_dir=str(tmp_path))
    servicer = LearnerServicer(config, model_name="grasp-v1")

    best_path = tmp_path / "best_production_checkpoint.pt"
    for index in range(49):
        servicer.record_episode_result(index % 10 != 0, 5.0)
    assert not best_path.exists()
    servicer.record_episode_result(True, 5.0)
    assert servicer._best_success_rate == pytest.approx(0.90)
    assert best_path.is_file()
    first = torch.load(best_path, map_location="cpu", weights_only=True)
    assert first["evaluation"]["best_window_episodes"] == 50

    servicer.record_episode_result(True, 5.0)
    assert servicer._best_success_rate == pytest.approx(0.92)
    improved = torch.load(best_path, map_location="cpu", weights_only=True)
    assert improved["evaluation"]["best_success_rate"] == pytest.approx(0.92)
    assert not servicer._early_stopped
    assert not servicer._config.deterministic_inference

    previous_step = servicer._training_step
    transitions = [transition_pb2.Transition(
        state=environment_pb2.State(values=[0.0, 0.1, 0.2]),
        action=environment_pb2.Action(values=[0.0, 0.0, 0.0]),
        reward=1.0,
        next_state=environment_pb2.State(values=[0.1, 0.2, 0.3]),
    ) for _ in range(4)]
    result = servicer.TrainBatch(learner_pb2.TrainBatchRequest(batch=transition_pb2.TransitionBatch(transitions=transitions)), AbortContext())
    assert result.training_step == previous_step + 1
    assert result.policy_version == 2


def test_legacy_locked_checkpoint_resumes_training(tmp_path: Path):
    config = LearnerConfig(state_dim=3, action_dim=3, checkpoint_dir=str(tmp_path))
    servicer = LearnerServicer(config, model_name="legacy-locked")
    servicer.SaveCheckpoint(learner_pb2.SaveCheckpointRequest(), AbortContext())
    path = tmp_path / "legacy-locked.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["evaluation"]["early_stopped"] = True
    payload["evaluation"].pop("early_stopping_schema")
    payload["learner_config"]["deterministic_inference"] = True
    torch.save(payload, path)

    resumed = LearnerServicer(config, model_name="legacy-locked")
    assert not resumed._early_stopped
    assert not resumed._config.deterministic_inference
    assert all(parameter.requires_grad for parameter in resumed._agent.actor.parameters())
    assert resumed._agent.log_alpha.requires_grad

    transitions = [transition_pb2.Transition(
        state=environment_pb2.State(values=[0.0, 0.1, 0.2]),
        action=environment_pb2.Action(values=[0.0, 0.0, 0.0]),
        reward=1.0,
        next_state=environment_pb2.State(values=[0.1, 0.2, 0.3]),
    ) for _ in range(4)]
    result = resumed.TrainBatch(learner_pb2.TrainBatchRequest(batch=transition_pb2.TransitionBatch(transitions=transitions)), AbortContext())
    assert result.training_step == 1
    assert result.policy_version == 2


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

    episode = servicer.RecordEpisodeResult(
        learner_pb2.RecordEpisodeResultRequest(episode_id="eval-1", success=True, episode_reward=1.0), AbortContext()
    )
    assert episode.stop_training
    assert episode.completed_episodes == 0
    assert not (tmp_path / "best_production_checkpoint.pt").exists()


def test_connectome_stops_after_20_consecutive_successes_and_locks_weights(tmp_path: Path, capsys):
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

    assert all(parameter.requires_grad for parameter in servicer._agent.actor.sensory_encoder.parameters())
    assert all(parameter.requires_grad for parameter in servicer._agent.actor.base_head.parameters())
    assert servicer._agent.actor.edge_log_gains.requires_grad

    deployment_servicer = LearnerServicer(config)
    for _ in range(19):
        deployment_servicer.record_episode_result(True, 1.0)
    assert not deployment_servicer._early_stopped
    deployment_servicer.record_episode_result(True, 1.0)
    assert deployment_servicer._early_stopped
    assert deployment_servicer._stop_reason == "consecutive_successes"
    assert deployment_servicer._config.deterministic_inference
    assert EARLY_STOP_MESSAGE in capsys.readouterr().out
    sensory_weight = next(deployment_servicer._agent.actor.sensory_encoder.parameters())
    before = sensory_weight.detach().clone()
    transitions = [transition_pb2.Transition(
        state=environment_pb2.State(values=[0.1] * 30),
        action=environment_pb2.Action(values=[0.0, 0.0, 0.0]),
        reward=1.0,
        next_state=environment_pb2.State(values=[0.2] * 30),
    ) for _ in range(4)]
    result = deployment_servicer.TrainBatch(learner_pb2.TrainBatchRequest(batch=transition_pb2.TransitionBatch(transitions=transitions)), AbortContext())
    assert result.training_step == 0
    assert torch.equal(before, sensory_weight)
    assert (tmp_path / "best_production_checkpoint.pt").is_file()
    restored = LearnerServicer(config, model_name="best_production_checkpoint")
    assert restored._early_stopped
    assert restored._completed_episodes == 20


def test_rolling_95_percent_stops_after_50_episodes(tmp_path: Path):
    servicer = LearnerServicer(LearnerConfig(state_dim=3, action_dim=3, checkpoint_dir=str(tmp_path)))
    outcomes = [index not in {15, 35} for index in range(50)]
    for success in outcomes:
        servicer.record_episode_result(success, 1.0)
    assert servicer._completed_episodes == 50
    assert servicer._best_success_rate == pytest.approx(0.96)
    assert servicer._stop_reason == "rolling_success_rate"
    assert (tmp_path / "best_production_checkpoint.pt").is_file()


def test_hard_limit_stops_at_150_and_saves_fallback_checkpoint(tmp_path: Path):
    servicer = LearnerServicer(LearnerConfig(state_dim=3, action_dim=3, checkpoint_dir=str(tmp_path)))
    for index in range(150):
        servicer.record_episode_result(index % 2 == 0, 1.0)
    assert servicer._completed_episodes == 150
    assert servicer._stop_reason == "max_episodes"
    assert servicer._early_stopped
    assert (tmp_path / "best_production_checkpoint.pt").is_file()


def test_stop_restores_earlier_peak_actor_for_continuing_evaluation(tmp_path: Path):
    servicer = LearnerServicer(LearnerConfig(state_dim=3, action_dim=3, checkpoint_dir=str(tmp_path)))
    for index in range(50):
        servicer.record_episode_result(index % 10 != 0, 1.0)
    peak = torch.load(tmp_path / "best_production_checkpoint.pt", map_location="cpu", weights_only=True)["agent"]["actor"]
    with torch.no_grad():
        next(servicer._agent.actor.parameters()).add_(1.0)
    for _ in range(100):
        servicer.record_episode_result(False, 0.0)
    assert servicer._stop_reason == "max_episodes"
    assert servicer._policy_version == 2
    restored_actor = servicer._agent.actor.state_dict()
    assert all(torch.equal(restored_actor[name], value) for name, value in peak.items() if isinstance(value, torch.Tensor))
    assert torch.load(tmp_path / "best_production_checkpoint.pt", map_location="cpu", weights_only=True)["evaluation"]["early_stopped"]


def test_episode_result_rpc_is_idempotent_and_uses_completed_episodes(tmp_path: Path):
    servicer = LearnerServicer(LearnerConfig(state_dim=3, action_dim=3, checkpoint_dir=str(tmp_path)))
    request = learner_pb2.RecordEpisodeResultRequest(episode_id="run-1", success=True, episode_reward=2.0)
    first = servicer.RecordEpisodeResult(request, AbortContext())
    duplicate = servicer.RecordEpisodeResult(request, AbortContext())
    assert first.completed_episodes == duplicate.completed_episodes == 1
    assert first.consecutive_successes == duplicate.consecutive_successes == 1
    assert not first.stop_training
