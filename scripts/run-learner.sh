#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=ai/generated poetry run python -m ai.grpc_server
