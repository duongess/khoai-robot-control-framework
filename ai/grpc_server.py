"""Local gRPC server for the Soft Actor-Critic learner."""

import json
import math
import logging
import signal
import sys
import threading
from concurrent import futures
from collections import deque
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import grpc
import torch

# Generated modules import one another as ``learner.v1``.  Make their actual
# repository root available for both `python -m ai` and pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen" / "python"))

from ai.config import LearnerConfig, SACConfig
from ai.connectome.dynamic_policy import DEFAULT_GANTRY_REFLEX_CONFIG
from ai.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_save_checkpoint,
    checkpoint_path,
    load_checkpoint,
    validate_model_name,
)
from ai.sac import SACAgent, TensorBatch
from gen.python.learner.v1 import environment_pb2, learner_pb2, learner_pb2_grpc


ADDRESS = "127.0.0.1:50051"


def _upgrade_legacy_default_grip(config: object) -> dict | None:
    """Replace only the obsolete default grip channel in persisted registries."""
    if not isinstance(config, dict) or not isinstance(config.get("channels"), list):
        return None
    updated = deepcopy(config)
    for index, channel in enumerate(updated["channels"]):
        if not isinstance(channel, dict) or channel.get("action_index") != 2:
            continue
        signals = channel.get("signal_indices")
        if channel.get("expression_type") == "linear" and isinstance(signals, dict) and signals.get("x") == 17:
            updated["channels"][index] = deepcopy(DEFAULT_GANTRY_REFLEX_CONFIG["channels"][2])
            return updated
    return None


class JsonFormatter(logging.Formatter):
    """Formats learner events as concise JSON log records."""

    def format(self, record: logging.LogRecord) -> str:
        event = {"level": record.levelname.lower(), "event": record.getMessage()}
        event.update(getattr(record, "fields", {}))
        return json.dumps(event, separators=(",", ":"), sort_keys=True)


def configure_logging() -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("learner_server")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = configure_logging()


class LearnerServicer(learner_pb2_grpc.LearnerServiceServicer):
    """A local SAC learner that trains only on supplied transition batches."""

    def __init__(self, config: LearnerConfig | None = None, model_name: str | None = None) -> None:
        requested_config = config or LearnerConfig.from_environment()
        self._model_name = validate_model_name(model_name) if model_name else ""
        checkpoint = checkpoint_path(requested_config.checkpoint_dir, self._model_name) if self._model_name else None
        payload = load_checkpoint(checkpoint) if checkpoint and checkpoint.is_file() else None
        if payload is None:
            self._config = requested_config
        else:
            try:
                # Loading by name must work without making a user remember all
                # controller/graph environment variables from the original run.
                saved_config = LearnerConfig(**payload["learner_config"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"checkpoint {checkpoint.name} has an invalid learner configuration: {error}") from error
            # The caller may deliberately move their local checkpoint directory;
            # retain that location while restoring every training setting.
            self._config = replace(saved_config, checkpoint_dir=requested_config.checkpoint_dir)
        self._agent = SACAgent(SACConfig(
            state_dim=self._config.state_dim,
            action_dim=self._config.action_dim,
            gamma=self._config.gamma,
            target_entropy=-float(self._config.action_dim),
            controller_type=self._config.controller_type,
            base_policy=self._config.base_policy,
            graph_path=self._config.graph_path,
            propagation_steps=self._config.propagation_steps,
            train_edge_gains=self._config.train_edge_gains,
            freeze_topology=self._config.freeze_topology,
            full_actor_unlock_step=self._config.full_actor_unlock_step,
            phase_gated_decoder=self._config.phase_gated_decoder,
            object_attached_observation_index=self._config.object_attached_observation_index,
            phase_observation_index=self._config.phase_observation_index,
            transport_phase_threshold=self._config.transport_phase_threshold,
            activation=self._config.activation,
            action_dead_zone=self._config.action_dead_zone,
            min_log_std=self._config.min_log_std,
            max_horizontal_speed=self._config.max_horizontal_speed,
            max_vertical_speed=self._config.max_vertical_speed,
            max_gripper_command=self._config.max_gripper_command,
            residual_alpha_x=self._config.residual_alpha_x,
            residual_alpha_y=self._config.residual_alpha_y,
            residual_alpha_grip=self._config.residual_alpha_grip,
            base_warmup_steps=self._config.base_warmup_steps,
            base_learning_rate_multiplier=self._config.base_learning_rate_multiplier,
            slip_severity_observation_index=self._config.slip_severity_observation_index,
            grip_force_observation_index=self._config.grip_force_observation_index,
            vertical_acceleration_observation_index=self._config.vertical_acceleration_observation_index,
            previous_vertical_action_observation_index=self._config.previous_vertical_action_observation_index,
        ))
        # A new graph learner must never silently serve the legacy residual actor.
        # Checkpoints restore their own persisted schema in _restore_checkpoint.
        if payload is None and self._config.controller_type in {"fly_connectome", "random_graph"}:
            self._agent.register_reflex_law(DEFAULT_GANTRY_REFLEX_CONFIG)
        self._samples_seen = 0
        # Version zero is reserved by the protocol for "latest". Every actual
        # episode therefore receives a positive, immutable snapshot ID.
        self._policy_version = 1
        self._actor_snapshots = {self._policy_version: deepcopy(self._agent.actor).eval()}
        self._policy_lock = threading.RLock()
        # Training mutates the live SAC networks, whereas prediction uses
        # immutable actor snapshots. Keep those concerns separate so an update
        # does not hold the prediction lookup lock for its entire backward pass.
        self._training_lock = threading.Lock()
        self._training_step = 0
        self._predict_requests = 0
        self._train_requests = 0
        self._episode_results = deque(maxlen=100)
        self._episode_keys: set[tuple[object, ...]] = set()
        self._best_success_rate = 0.0
        self._best_average_reward = float("-inf")
        self._high_success_streak = 0
        self._early_stopped = False
        self._last_metrics = self._empty_metrics()
        if payload is not None:
            self._restore_checkpoint(payload, checkpoint)
            # Legacy checkpoints predate the registry. Upgrade them at boot
            # rather than reviving the residual branch for a new episode.
            if self._config.controller_type in {"fly_connectome", "random_graph"}:
                if not self._agent.actor.reflex_parameter_count:
                    self._agent.register_reflex_law(DEFAULT_GANTRY_REFLEX_CONFIG)
                    self._actor_snapshots = {self._policy_version: deepcopy(self._agent.actor).eval()}
                else:
                    migrated = _upgrade_legacy_default_grip(self._agent.actor.get_extra_state().get("reflex_config"))
                    if migrated is not None:
                        self._agent.register_reflex_law(migrated)
                        self._actor_snapshots = {self._policy_version: deepcopy(self._agent.actor).eval()}

    def PredictBatch(self, request, context):
        try:
            states = self._states_to_tensor(request.states, "states")
            # SAC must sample actions while it is collecting replay data. A
            # deterministic, untrained actor repeats one arbitrary vector (for
            # example, simultaneous right/down motion) and never explores a
            # corrective action. Evaluation can opt in explicitly via
            # LEARNER_DETERMINISTIC_INFERENCE=true.
            # Snapshots are immutable after publication, so hold the lock only
            # while looking one up. A prediction can then run concurrently with
            # the next training update instead of being stuck behind it.
            with self._policy_lock:
                requested_version = request.policy_version
                served_version = self._policy_version if requested_version == 0 else requested_version
                actor = self._actor_snapshots.get(served_version)
            if actor is None:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"policy snapshot {served_version} is unavailable; start a fresh episode",
                )
            with torch.inference_mode():
                mean, log_std = actor(states)
                predicted_actions, fly_base_actions, residual_actions = self._agent.act_with_actor_components(
                    actor, states, deterministic=self._config.deterministic_inference
                )
                predicted_actions = predicted_actions.clamp(-1, 1)
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        actions = [environment_pb2.Action(values=[float(value) for value in action]) for action in predicted_actions]
        fly_base = [environment_pb2.Action(values=[float(value) for value in action]) for action in fly_base_actions]
        residuals = [environment_pb2.Action(values=[float(value) for value in action]) for action in residual_actions]
        self._predict_requests += 1
        if self._predict_requests % self._config.log_every_n_requests == 0:
            LOGGER.info("predict_batch", extra={"fields": {
                "states": len(request.states),
                "requested_policy_version": request.policy_version,
                "policy_version": served_version,
                "deterministic": self._config.deterministic_inference,
                "action_mean": [float(value) for value in predicted_actions.mean(dim=0)],
                "action_std": [float(value) for value in predicted_actions.std(dim=0, unbiased=False)],
                "fly_base_mean": [float(value) for value in fly_base_actions.mean(dim=0)],
                "sac_residual_mean": [float(value) for value in residual_actions.mean(dim=0)],
                "actor_mean_pre_tanh": [float(value) for value in mean.detach().mean(dim=0)],
                "actor_log_std": [float(value) for value in log_std.detach().mean(dim=0)],
            }})
        return learner_pb2.PredictBatchResponse(
            actions=actions,
            policy_version=served_version,
            fly_base_actions=fly_base,
            residual_actions=residuals,
        )

    def TrainBatch(self, request, context):
        if not request.HasField("batch"):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "transition batch is required")
        try:
            batch = self._batch_to_tensors(request.batch.transitions)
            self._record_terminal_transitions(request.batch.transitions)
            # The Go runtime already applies backpressure, and this lock also
            # makes the service safe if another client submits a train request.
            # Prediction continues against its prior immutable snapshot while
            # the live actor/critics perform a backward pass.
            with self._training_lock:
                if self._early_stopped:
                    self._zero_optimizer_gradients()
                    metrics = dict(self._last_metrics)
                else:
                    metrics = self._agent.update(batch)
                if not self._early_stopped:
                    snapshot = deepcopy(self._agent.actor).eval()
                    with self._policy_lock:
                        self._policy_version += 1
                        self._actor_snapshots[self._policy_version] = snapshot
                        while len(self._actor_snapshots) > self._config.max_policy_snapshots:
                            del self._actor_snapshots[min(self._actor_snapshots)]
                # Keep counters in the same critical section as weights and
                # policy publication so a concurrent checkpoint is coherent.
                self._samples_seen += len(request.batch.transitions)
                if not self._early_stopped:
                    self._training_step += 1
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        self._last_metrics = dict(metrics)
        self._train_requests += 1
        if self._train_requests % self._config.log_every_n_requests == 0:
            LOGGER.info("train_batch", extra={"fields": {"received": len(request.batch.transitions), "samples_seen": self._samples_seen, **metrics}})
        return learner_pb2.TrainBatchResponse(
            accepted=True,
            samples_seen=self._samples_seen,
            policy_version=self._policy_version,
            actor_loss=metrics["actor_loss"],
            critic_loss=(metrics["critic_one_loss"] + metrics["critic_two_loss"]) / 2,
            alpha_loss=metrics["alpha_loss"],
            entropy=metrics["entropy"],
            training_step=self._training_step,
            critic_one_q=metrics["critic_one_q"],
            critic_two_q=metrics["critic_two_q"],
            alpha=metrics["alpha"],
            actor_log_std_horizontal=metrics["actor_log_std_horizontal"],
            actor_log_std_vertical=metrics["actor_log_std_vertical"],
            actor_log_std_gripper=metrics["actor_log_std_gripper"],
        )

    @staticmethod
    def _empty_metrics() -> dict[str, float]:
        return {"actor_loss": 0.0, "alpha_loss": 0.0, "entropy": 0.0, "critic_one_loss": 0.0, "critic_two_loss": 0.0, "critic_one_q": 0.0, "critic_two_q": 0.0, "alpha": 1.0, "actor_log_std_horizontal": 0.0, "actor_log_std_vertical": 0.0, "actor_log_std_gripper": 0.0}

    def _zero_optimizer_gradients(self) -> None:
        for optimizer in (self._agent.actor_optimizer, self._agent.critic_one_optimizer, self._agent.critic_two_optimizer, self._agent.alpha_optimizer):
            optimizer.zero_grad(set_to_none=True)

    def _rolling_success_rate(self) -> float:
        if not self._episode_results:
            return 0.0
        return sum(1 for success, _ in self._episode_results if success) / len(self._episode_results)

    def _rolling_average_reward(self) -> float:
        if not self._episode_results:
            return float("-inf")
        return sum(reward for _, reward in self._episode_results) / len(self._episode_results)

    def _freeze_learning_locked(self) -> None:
        self._zero_optimizer_gradients()
        for module in (self._agent.actor, self._agent.critic_one, self._agent.critic_two):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self._agent.log_alpha.requires_grad_(False)
        self._early_stopped = True
        LOGGER.warning("[EARLY STOPPING] Target accuracy achieved (>=99%%). Model frozen at peak performance.", extra={"fields": {"success_rate": self._rolling_success_rate(), "episodes": len(self._episode_results)}})

    def _evaluation_state(self) -> dict[str, object]:
        return {"best_success_rate": self._best_success_rate, "best_average_reward": self._best_average_reward, "high_success_streak": self._high_success_streak, "early_stopped": self._early_stopped, "episode_results": list(self._episode_results)}

    def _restore_evaluation_state(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        results = payload.get("episode_results")
        if isinstance(results, list):
            for item in results[-100:]:
                if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], bool) and isinstance(item[1], (int, float)) and math.isfinite(float(item[1])):
                    self._episode_results.append((item[0], float(item[1])))
        if isinstance(payload.get("best_success_rate"), (int, float)) and math.isfinite(float(payload["best_success_rate"])):
            self._best_success_rate = float(payload["best_success_rate"])
        if isinstance(payload.get("best_average_reward"), (int, float)) and math.isfinite(float(payload["best_average_reward"])):
            self._best_average_reward = float(payload["best_average_reward"])
        if isinstance(payload.get("high_success_streak"), int) and payload["high_success_streak"] >= 0:
            self._high_success_streak = payload["high_success_streak"]
        if payload.get("early_stopped") is True:
            self._freeze_learning_locked()

    def _save_best_checkpoint_locked(self, success_rate: float, average_reward: float) -> None:
        path = checkpoint_path(self._config.checkpoint_dir, "best_policy_checkpoint")
        evaluation = self._evaluation_state()
        evaluation["best_success_rate"] = success_rate
        evaluation["best_average_reward"] = average_reward
        payload = {"format_version": CHECKPOINT_FORMAT_VERSION, "model_name": "best_policy_checkpoint", "learner_config": asdict(self._config), "agent": self._agent.checkpoint_state(), "samples_seen": self._samples_seen, "policy_version": self._policy_version, "training_step": self._training_step, "evaluation": evaluation}
        atomic_save_checkpoint(path, payload)
        self._best_success_rate = success_rate
        self._best_average_reward = average_reward
        LOGGER.warning("[AUTO-SAVE] New best model saved with Success Rate: %.1f%%!", success_rate * 100.0, extra={"fields": {"path": str(path), "average_reward": average_reward}})

    def _record_episode_result_locked(self, success: bool, average_reward: float) -> None:
        if not math.isfinite(average_reward):
            raise ValueError("episode reward must be finite")
        self._episode_results.append((bool(success), float(average_reward)))
        success_rate = self._rolling_success_rate()
        average = self._rolling_average_reward()
        self._high_success_streak = self._high_success_streak + 1 if success_rate >= 0.99 else 0
        if success_rate >= 0.985 and average > self._best_average_reward:
            self._save_best_checkpoint_locked(success_rate, average)
        if self._high_success_streak >= 100 and not self._early_stopped:
            self._freeze_learning_locked()

    def record_episode_result(self, success: bool, average_reward: float) -> None:
        """Record an exact evaluation result from an external episode runner."""
        with self._training_lock:
            self._record_episode_result_locked(success, float(average_reward))

    @staticmethod
    def _terminal_success(transition) -> bool:
        if transition.truncated:
            return False
        values = list(transition.next_state.values)
        if len(values) > 19 and math.isfinite(float(values[19])):
            phase_code = round((float(values[19]) + 1.0) * 4.5)
            if phase_code == 8:
                return True
            if phase_code == 9:
                return False
        return float(transition.reward) > 0.0

    def _record_terminal_transitions(self, transitions) -> None:
        with self._training_lock:
            for transition in transitions:
                if not (transition.terminated or transition.truncated):
                    continue
                episode_id = transition.episode_id
                key = (episode_id, int(transition.step)) if episode_id else ("terminal", tuple(round(float(value), 6) for value in transition.next_state.values), round(float(transition.reward), 6), bool(transition.truncated))
                if key in self._episode_keys:
                    continue
                self._episode_keys.add(key)
                self._record_episode_result_locked(self._terminal_success(transition), float(transition.reward))
                if len(self._episode_keys) > 10000:
                    self._episode_keys.clear()

    def HealthCheck(self, request, context):
        return learner_pb2.HealthCheckResponse(
            ready=True,
            policy_version=self._policy_version,
            training_step=self._training_step,
            device=str(self._agent.device),
            model_name=self._model_name,
        )

    def SaveCheckpoint(self, request, context):
        try:
            model_name = validate_model_name(self._model_name)
            path = checkpoint_path(self._config.checkpoint_dir, model_name)
            # Follow the same lock ordering as TrainBatch. This creates a
            # consistent state across networks, optimizers and policy version.
            with self._training_lock:
                with self._policy_lock:
                    payload = {
                        "format_version": CHECKPOINT_FORMAT_VERSION,
                        "model_name": model_name,
                        "learner_config": asdict(self._config),
                        "agent": self._agent.checkpoint_state(),
                        "samples_seen": self._samples_seen,
                        "policy_version": self._policy_version,
                        "training_step": self._training_step,
                        "evaluation": self._evaluation_state(),
                    }
                    atomic_save_checkpoint(path, payload)
                    self._model_name = model_name
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        LOGGER.info("checkpoint_saved", extra={"fields": {
            "model_name": model_name,
            "policy_version": self._policy_version,
            "training_step": self._training_step,
        }})
        return learner_pb2.SaveCheckpointResponse(
            model_name=model_name,
            policy_version=self._policy_version,
            training_step=self._training_step,
        )

    def _restore_checkpoint(self, payload: dict, checkpoint: Path) -> None:
        try:
            self._agent.load_checkpoint_state(payload["agent"])
            self._samples_seen = self._checkpoint_counter(payload, "samples_seen")
            self._policy_version = self._checkpoint_counter(payload, "policy_version", minimum=1)
            self._training_step = self._checkpoint_counter(payload, "training_step")
            self._restore_evaluation_state(payload.get("evaluation"))
            # The staged actor schedule belongs to the learner counter, which
            # is stored outside SAC's tensor state. Restore it explicitly so a
            # resumed model is not accidentally re-frozen for 128 updates.
            self._agent.training_step = self._training_step
            self._agent._configure_actor_trainability()
            if self._early_stopped:
                self._freeze_learning_locked()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"could not restore checkpoint {checkpoint.name}: {error}") from error
        # Episode snapshots intentionally do not survive a process restart.
        # New Go episodes ask for this restored latest version first.
        self._actor_snapshots = {self._policy_version: deepcopy(self._agent.actor).eval()}
        LOGGER.info("checkpoint_loaded", extra={"fields": {
            "model_name": self._model_name,
            "policy_version": self._policy_version,
            "training_step": self._training_step,
        }})

    @staticmethod
    def _checkpoint_counter(payload: dict, key: str, minimum: int = 0) -> int:
        value = payload.get(key)
        if not isinstance(value, int) or value < minimum:
            raise ValueError(f"checkpoint {key} is invalid")
        return value

    def _states_to_tensor(self, states, field_name: str) -> torch.Tensor:
        values = [list(state.values) for state in states]
        if not values:
            raise ValueError(f"{field_name} must not be empty")
        if any(len(state) != self._config.state_dim for state in values):
            raise ValueError(f"{field_name} must contain states with dimension {self._config.state_dim}")
        tensor = torch.tensor(values, dtype=torch.float32)
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{field_name} must contain only finite values")
        return tensor

    def _batch_to_tensors(self, transitions) -> TensorBatch:
        if not transitions:
            raise ValueError("transition batch must not be empty")
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for index, transition in enumerate(transitions):
            if not transition.HasField("state") or not transition.HasField("action") or not transition.HasField("next_state"):
                raise ValueError(f"transition {index} is missing state, action, or next state")
            if len(transition.state.values) != self._config.state_dim or len(transition.next_state.values) != self._config.state_dim:
                raise ValueError(f"transition {index} has an invalid state dimension")
            if len(transition.action.values) != self._config.action_dim:
                raise ValueError(f"transition {index} has an invalid action dimension")
            states.append(list(transition.state.values))
            actions.append(list(transition.action.values))
            rewards.append([transition.reward])
            next_states.append(list(transition.next_state.values))
            dones.append([float(transition.terminated or transition.truncated)])
        batch = TensorBatch(states=torch.tensor(states, dtype=torch.float32), actions=torch.tensor(actions, dtype=torch.float32), rewards=torch.tensor(rewards, dtype=torch.float32), next_states=torch.tensor(next_states, dtype=torch.float32), dones=torch.tensor(dones, dtype=torch.float32))
        if not all(torch.isfinite(tensor).all() for tensor in (batch.states, batch.actions, batch.rewards, batch.next_states, batch.dones)):
            raise ValueError("transition batch contains non-finite values")
        return batch


def create_server(config: LearnerConfig | None = None, model_name: str | None = None) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    learner_pb2_grpc.add_LearnerServiceServicer_to_server(LearnerServicer(config, model_name), server)
    if server.add_insecure_port(ADDRESS) == 0:
        raise RuntimeError(f"could not bind learner server to {ADDRESS}")
    return server


def serve(model_name: str | None = None) -> None:
    config = LearnerConfig.from_environment()
    # Set these before any gRPC work is accepted. Limiting threads prevents a
    # small CPU actor/critic update from spawning enough native workers to make
    # the browser and desktop unresponsive.
    torch.set_num_threads(config.torch_num_threads)
    torch.set_num_interop_threads(config.torch_num_interop_threads)
    server = create_server(config, model_name)
    server.start()
    LOGGER.info("server_started", extra={"fields": {"address": ADDRESS}})

    def stop_server(*_args) -> None:
        LOGGER.info("server_stopping", extra={"fields": {"address": ADDRESS}})
        server.stop(grace=0)

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    server.wait_for_termination()
