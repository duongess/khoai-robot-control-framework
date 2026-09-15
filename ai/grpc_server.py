"""Local gRPC server for the Soft Actor-Critic learner."""

import json
import logging
import signal
import sys
import threading
from concurrent import futures
from copy import deepcopy
from pathlib import Path

import grpc
import torch

# Generated modules import one another as ``learner.v1``.  Make their actual
# repository root available for both `python -m ai` and pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen" / "python"))

from ai.config import LearnerConfig, SACConfig
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

    def __init__(self, config: LearnerConfig | None = None) -> None:
        self._config = config or LearnerConfig.from_environment()
        self._agent = SACAgent(SACConfig(
            state_dim=self._config.state_dim,
            action_dim=self._config.action_dim,
            target_entropy=-float(self._config.action_dim),
            controller_type=self._config.controller_type,
            graph_path=self._config.graph_path,
            propagation_steps=self._config.propagation_steps,
            train_edge_gains=self._config.train_edge_gains,
            freeze_topology=self._config.freeze_topology,
            activation=self._config.activation,
            action_dead_zone=self._config.action_dead_zone,
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
        self._training_step = 0

    def PredictBatch(self, request, context):
        try:
            states = self._states_to_tensor(request.states, "states")
            # SAC must sample actions while it is collecting replay data. A
            # deterministic, untrained actor repeats one arbitrary vector (for
            # example, simultaneous right/down motion) and never explores a
            # corrective action. Evaluation can opt in explicitly via
            # LEARNER_DETERMINISTIC_INFERENCE=true.
            with self._policy_lock:
                requested_version = request.policy_version
                served_version = self._policy_version if requested_version == 0 else requested_version
                actor = self._actor_snapshots.get(served_version)
                if actor is None:
                    context.abort(
                        grpc.StatusCode.FAILED_PRECONDITION,
                        f"policy snapshot {served_version} is unavailable; start a fresh episode",
                    )
                predicted_actions = self._agent.act_with_actor(
                    actor, states, deterministic=self._config.deterministic_inference
                ).clamp(-1, 1)
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        actions = [environment_pb2.Action(values=[float(value) for value in action]) for action in predicted_actions]
        LOGGER.info("predict_batch", extra={"fields": {"states": len(request.states), "requested_policy_version": request.policy_version, "policy_version": served_version, "deterministic": self._config.deterministic_inference}})
        return learner_pb2.PredictBatchResponse(actions=actions, policy_version=served_version)

    def TrainBatch(self, request, context):
        if not request.HasField("batch"):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "transition batch is required")
        try:
            batch = self._batch_to_tensors(request.batch.transitions)
            with self._policy_lock:
                metrics = self._agent.update(batch)
                self._policy_version += 1
                self._actor_snapshots[self._policy_version] = deepcopy(self._agent.actor).eval()
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        self._samples_seen += len(request.batch.transitions)
        self._training_step += 1
        LOGGER.info("train_batch", extra={"fields": {"received": len(request.batch.transitions), "samples_seen": self._samples_seen, **metrics}})
        return learner_pb2.TrainBatchResponse(accepted=True, samples_seen=self._samples_seen, policy_version=self._policy_version, actor_loss=metrics["actor_loss"], critic_loss=(metrics["critic_one_loss"] + metrics["critic_two_loss"]) / 2, alpha_loss=metrics["alpha_loss"], entropy=metrics["entropy"], training_step=self._training_step)

    def HealthCheck(self, request, context):
        return learner_pb2.HealthCheckResponse(ready=True, policy_version=self._policy_version, training_step=self._training_step, device=str(self._agent.device))

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


def create_server() -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    learner_pb2_grpc.add_LearnerServiceServicer_to_server(LearnerServicer(), server)
    if server.add_insecure_port(ADDRESS) == 0:
        raise RuntimeError(f"could not bind learner server to {ADDRESS}")
    return server


def serve() -> None:
    server = create_server()
    server.start()
    LOGGER.info("server_started", extra={"fields": {"address": ADDRESS}})

    def stop_server(*_args) -> None:
        LOGGER.info("server_stopping", extra={"fields": {"address": ADDRESS}})
        server.stop(grace=0)

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    server.wait_for_termination()
