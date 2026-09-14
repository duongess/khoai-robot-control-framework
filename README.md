# Khoai robot control framework

The framework batches environment observations, requests continuous actions from the local SAC learner, steps registered tasks, stores replay transitions, and submits training batches. It does not own robot physics or browser-facing APIs.

## Local learner

Generate bindings after protocol changes:

```bash
bash scripts/generate-proto.sh
```

Start the private learner service:

```bash
poetry run python -m ai
```

The learner listens only on `127.0.0.1:50051`. Its gRPC protocol contains generic numeric states, actions, and transitions; it has no visualizer-specific messages.

## Tests

```bash
go test -race ./...
poetry run pytest
```

To run the complete vertical slice, start this learner first, then run `go run ./cmd/force-control-demo` from the sibling `khoai-robot-visualizer-web` repository and open `http://127.0.0.1:8080`.
