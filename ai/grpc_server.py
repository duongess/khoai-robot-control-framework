"""Local gRPC server for the Soft Actor-Critic learner."""

import json
import logging
import signal
from concurrent import futures

import grpc
import torch

from ai.config import LearnerConfig, SACConfig
from proto.learner.v1 import environment_pb2, learner_pb2, learner_pb2_grpc
from ai.sac import SACAgent, TensorBatch


ADDRESS = "127.0.0.1:50051"
POLICY_VERSION = "sac-v1"


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
        self._agent = SACAgent(SACConfig(state_dim=self._config.state_dim))
        self._samples_seen = 0

    def PredictBatch(self, request, context):
        try:
            states = self._states_to_tensor(request.states, "states")
            predicted_actions = self._agent.act(states, deterministic=True)
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))

        actions = [environment_pb2.Action(values=[float(action[0])]) for action in predicted_actions]
        LOGGER.info("predict_batch", extra={"fields": {"states": len(request.states)}})
        return learner_pb2.PredictBatchResponse(actions=actions, policy_version=POLICY_VERSION)

    def TrainBatch(self, request, context):
        if not request.HasField("batch"):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "transition batch is required")

        try:
            batch = self._batch_to_tensors(request.batch.transitions)
            metrics = self._agent.update(batch)
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))

        received = len(request.batch.transitions)
        self._samples_seen += received
        samples_seen = self._samples_seen
        LOGGER.info(
            "train_batch",
            extra={"fields": {"received": received, "samples_seen": samples_seen, **metrics}},
        )
        return learner_pb2.TrainBatchResponse(
            accepted=True,
            samples_seen=samples_seen,
            policy_version=POLICY_VERSION,
        )

    def HealthCheck(self, request, context):
        LOGGER.info("health_check", extra={"fields": {"ready": True}})
        return learner_pb2.HealthCheckResponse(ready=True)

    def _states_to_tensor(self, states, field_name: str) -> torch.Tensor:
        values = [list(state.values) for state in states]
        if any(len(state) != self._config.state_dim for state in values):
            raise ValueError(f"{field_name} must contain states with dimension {self._config.state_dim}")
        if not values:
            return torch.empty((0, self._config.state_dim), dtype=torch.float32)
        return torch.tensor(values, dtype=torch.float32)

    def _batch_to_tensors(self, transitions) -> TensorBatch:
        if not transitions:
            raise ValueError("transition batch must not be empty")

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for index, transition in enumerate(transitions):
            if not transition.HasField("state"):
                raise ValueError(f"transition {index} is missing state")
            if not transition.HasField("action"):
                raise ValueError(f"transition {index} is missing action")
            if not transition.HasField("next_state"):
                raise ValueError(f"transition {index} is missing next state")
            if len(transition.state.values) != self._config.state_dim:
                raise ValueError(f"transition {index} has an invalid state dimension")
            if len(transition.next_state.values) != self._config.state_dim:
                raise ValueError(f"transition {index} has an invalid next state dimension")
            if len(transition.action.values) != 1:
                raise ValueError(f"transition {index} must contain one action value")

            states.append(list(transition.state.values))
            actions.append(list(transition.action.values))
            rewards.append([transition.reward])
            next_states.append(list(transition.next_state.values))
            dones.append([float(transition.terminated or transition.truncated)])

        return TensorBatch(
            states=torch.tensor(states, dtype=torch.float32),
            actions=torch.tensor(actions, dtype=torch.float32),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            next_states=torch.tensor(next_states, dtype=torch.float32),
            dones=torch.tensor(dones, dtype=torch.float32),
        )


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


if __name__ == "__main__":
    serve()
