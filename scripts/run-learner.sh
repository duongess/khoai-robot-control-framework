#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=.:gen/python poetry run python -m ai.grpc_server
