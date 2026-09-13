"""Minimal local gRPC learner service."""

import json
import logging
import signal
import threading
from concurrent import futures

import grpc

from generated.learner.v1 import environment_pb2, learner_pb2, learner_pb2_grpc


ADDRESS = "127.0.0.1:50051"
POLICY_VERSION = "grip-0.5"


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
    """A deterministic learner implementation for local integration."""

    def __init__(self) -> None:
        self._samples_seen = 0
        self._lock = threading.Lock()

    def PredictBatch(self, request, context):
        actions = [environment_pb2.Action(values=[0.5]) for _ in request.states]
        LOGGER.info("predict_batch", extra={"fields": {"states": len(request.states)}})
        return learner_pb2.PredictBatchResponse(
            actions=actions,
            policy_version=POLICY_VERSION,
        )

    def TrainBatch(self, request, context):
        if not request.HasField("batch"):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "transition batch is required")

        for index, transition in enumerate(request.batch.transitions):
            if not transition.HasField("state"):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"transition {index} is missing state")
            if not transition.HasField("action"):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"transition {index} is missing action")
            if not transition.HasField("next_state"):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"transition {index} is missing next_state")

        received = len(request.batch.transitions)
        with self._lock:
            self._samples_seen += received
            samples_seen = self._samples_seen

        LOGGER.info(
            "train_batch",
            extra={"fields": {"received": received, "samples_seen": samples_seen}},
        )
        return learner_pb2.TrainBatchResponse(
            accepted=True,
            samples_seen=samples_seen,
            policy_version=POLICY_VERSION,
        )

    def HealthCheck(self, request, context):
        LOGGER.info("health_check", extra={"fields": {"ready": True}})
        return learner_pb2.HealthCheckResponse(ready=True)


def create_server() -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
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
