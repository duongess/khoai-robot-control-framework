"""Start a new local SAC learner, or resume one named local checkpoint."""

from __future__ import annotations

import argparse

from ai.grpc_server import serve


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m ai",
        description="Run a new SAC learner or resume a named local model checkpoint.",
    )
    parser.add_argument(
        "model_name",
        help="checkpoint name to load (or name a new run); e.g. grasp-v1",
    )
    arguments = parser.parse_args()
    serve(arguments.model_name)
