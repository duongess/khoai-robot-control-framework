#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROTO_FILES=(proto/learner/v1/environment.proto proto/learner/v1/transition.proto proto/learner/v1/learner.proto)
mkdir -p gen/go/learner/v1 ai/generated/learner/v1
touch ai/generated/__init__.py ai/generated/learner/__init__.py ai/generated/learner/v1/__init__.py

if command -v protoc >/dev/null 2>&1; then
  PROTOC=(protoc)
else
  PROTOC=(poetry run python -m grpc_tools.protoc)
fi

"${PROTOC[@]}" -I proto \
  --plugin="protoc-gen-go=$(command -v protoc-gen-go)" \
  --go_out=. \
  --go_opt=module=github.com/duongess/khoai-robot-control-framework \
  "${PROTO_FILES[@]}"
"${PROTOC[@]}" -I proto \
  --plugin="protoc-gen-go-grpc=$(command -v protoc-gen-go-grpc)" \
  --go-grpc_out=. \
  --go-grpc_opt=module=github.com/duongess/khoai-robot-control-framework \
  proto/learner/v1/learner.proto

poetry run python -m grpc_tools.protoc -I proto \
  --python_out=ai/generated --pyi_out=ai/generated \
  "${PROTO_FILES[@]}"
poetry run python -m grpc_tools.protoc -I proto \
  --grpc_python_out=ai/generated \
  proto/learner/v1/learner.proto
