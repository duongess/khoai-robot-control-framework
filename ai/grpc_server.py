"""Local gRPC server for the Soft Actor-Critic learner."""

import json
import logging
import signal
import sys
import threading
from concurrent import futures
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import grpc
import torch

# Generated modules import one another as ``learner.v1``.  Make their actual
# repository root available for both `python -m ai` and pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen" / "python"))

from ai.config import LearnerConfig, SACConfig
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
            graph_path=self._config.graph_path,
            propagation_steps=self._config.propagation_steps,
            train_edge_gains=self._config.train_edge_gains,
            freeze_topology=self._config.freeze_topology,
            activation=self._config.activation,
            action_dead_zone=self._config.action_dead_zone,
            min_log_std=self._config.min_log_std,
            max_horizontal_speed=self._config.max_horizontal_speed,
            max_vertical_speed=self._config.max_vertical_speed,
            max_gripper_command=self._config.max_gripper_command,
        ))
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
        if payload is not None:
            self._restore_checkpoint(payload, checkpoint)

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
                predicted_actions = self._agent.act_with_actor(
                    actor, states, deterministic=self._config.deterministic_inference
                ).clamp(-1, 1)
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        actions = [environment_pb2.Action(values=[float(value) for value in action]) for action in predicted_actions]
        self._predict_requests += 1
        if self._predict_requests % self._config.log_every_n_requests == 0:
            LOGGER.info("predict_batch", extra={"fields": {
                "states": len(request.states),
                "requested_policy_version": request.policy_version,
                "policy_version": served_version,
                "deterministic": self._config.deterministic_inference,
                "action_mean": [float(value) for value in predicted_actions.mean(dim=0)],
                "action_std": [float(value) for value in predicted_actions.std(dim=0, unbiased=False)],
                "actor_mean_pre_tanh": [float(value) for value in mean.detach().mean(dim=0)],
                "actor_log_std": [float(value) for value in log_std.detach().mean(dim=0)],
            }})
        return learner_pb2.PredictBatchResponse(actions=actions, policy_version=served_version)

    def TrainBatch(self, request, context):
        if not request.HasField("batch"):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "transition batch is required")
        try:
            batch = self._batch_to_tensors(request.batch.transitions)
            # The Go runtime already applies backpressure, and this lock also
            # makes the service safe if another client submits a train request.
            # Prediction continues against its prior immutable snapshot while
            # the live actor/critics perform a backward pass.
            with self._training_lock:
                metrics = self._agent.update(batch)
                snapshot = deepcopy(self._agent.actor).eval()
                with self._policy_lock:
                    self._policy_version += 1
                    self._actor_snapshots[self._policy_version] = snapshot
                    while len(self._actor_snapshots) > self._config.max_policy_snapshots:
                        del self._actor_snapshots[min(self._actor_snapshots)]
                # Keep counters in the same critical section as weights and
                # policy publication so a concurrent checkpoint is coherent.
                self._samples_seen += len(request.batch.transitions)
                self._training_step += 1
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
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
            requested_name = request.model_name.strip() or self._model_name
            model_name = validate_model_name(requested_name)
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
