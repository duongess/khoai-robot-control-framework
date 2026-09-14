"""Cached neuPrint connectome artifacts and sparse controller components.

Nothing in this package queries neuPrint during training or inference.  Querying is
restricted to :mod:`ai.connectome.acquire` and produces a local graph artifact.
"""

from ai.connectome.graph import ConnectomeGraph, GraphArtifactError
from ai.connectome.policy import FlyConnectomePolicy, RandomGraphPolicy

__all__ = ["ConnectomeGraph", "FlyConnectomePolicy", "GraphArtifactError", "RandomGraphPolicy"]
