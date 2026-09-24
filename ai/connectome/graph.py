"""Portable, directed sparse connectome graph artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse
import torch


class GraphArtifactError(ValueError):
    """Raised when a connectome cache is absent, malformed, or incompatible."""


@dataclass(frozen=True)
class ConnectomeGraph:
    """A directed graph whose ``edge_index`` is ordered ``[source, target]``.

    ``weights`` are normalized by total incoming weight.  The raw synapse count is
    retained in ``edges.parquet``; this compact runtime representation deliberately
    avoids a dense N x N adjacency matrix.
    """

    body_ids: np.ndarray
    edge_index: np.ndarray
    weights: np.ndarray
    motor_groups: dict[str, tuple[int, ...]] = field(default_factory=dict)
    input_groups: dict[str, tuple[int, ...]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        body_ids = np.asarray(self.body_ids, dtype=np.int64)
        edge_index = np.asarray(self.edge_index, dtype=np.int64)
        weights = np.asarray(self.weights, dtype=np.float32)
        if body_ids.ndim != 1 or len(body_ids) == 0 or len(np.unique(body_ids)) != len(body_ids):
            raise GraphArtifactError("body_ids must be a non-empty vector of unique body IDs")
        if edge_index.shape != (2, len(weights)):
            raise GraphArtifactError("edge_index must have shape (2, edge_count)")
        if len(weights) == 0:
            raise GraphArtifactError("connectome graph must contain at least one edge")
        if (edge_index < 0).any() or (edge_index >= len(body_ids)).any():
            raise GraphArtifactError("edge index references a node outside the graph")
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise GraphArtifactError("edge weights must be finite and non-negative")
        object.__setattr__(self, "body_ids", body_ids)
        object.__setattr__(self, "edge_index", edge_index)
        object.__setattr__(self, "weights", weights)
        self._validate_groups(self.motor_groups, "motor")
        self._validate_groups(self.input_groups, "input")

    @property
    def node_count(self) -> int:
        return len(self.body_ids)

    @property
    def edge_count(self) -> int:
        return len(self.weights)

    @property
    def body_id_to_index(self) -> dict[int, int]:
        return {int(body_id): index for index, body_id in enumerate(self.body_ids)}

    def torch_adjacency(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return coalesced sparse COO adjacency in ``[target, source]`` layout."""
        indices = torch.tensor(self.edge_index[[1, 0]], dtype=torch.long, device=device)
        values = torch.tensor(self.weights, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, values, (self.node_count, self.node_count), device=device, check_invariants=False).coalesce()

    def save(self, directory: str | Path) -> None:
        """Save the sparse matrix plus auditable mapping and group metadata."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        matrix = scipy.sparse.coo_matrix(
            (self.weights, (self.edge_index[1], self.edge_index[0])), shape=(self.node_count, self.node_count)
        ).tocsr()
        scipy.sparse.save_npz(directory / "connectome_graph.npz", matrix)
        payload = {
            "format_version": 1,
            "body_ids": [int(value) for value in self.body_ids],
            "edge_orientation": "matrix[target, source]; source_to_target in edges.parquet",
            "weight_normalization": "incoming_sum",
            "motor_groups": {name: list(indices) for name, indices in self.motor_groups.items()},
            "input_groups": {name: list(indices) for name, indices in self.input_groups.items()},
            **self.metadata,
        }
        (directory / "connectome_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, graph_path: str | Path) -> "ConnectomeGraph":
        """Load a cache without importing or contacting neuprint-python."""
        graph_path = Path(graph_path)
        metadata_path = graph_path.with_name("connectome_metadata.json")
        if not graph_path.is_file() or not metadata_path.is_file():
            raise GraphArtifactError(f"connectome cache requires {graph_path} and {metadata_path}")
        try:
            matrix = scipy.sparse.load_npz(graph_path).tocoo()
            payload = json.loads(metadata_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise GraphArtifactError(f"could not load connectome cache: {error}") from error
        body_ids = np.asarray(payload.get("body_ids", []), dtype=np.int64)
        if matrix.shape != (len(body_ids), len(body_ids)):
            raise GraphArtifactError("cached matrix dimensions do not match body_ids")
        return cls(
            body_ids=body_ids,
            edge_index=np.vstack((matrix.col, matrix.row)),
            weights=matrix.data,
            motor_groups={name: tuple(int(i) for i in indices) for name, indices in payload.get("motor_groups", {}).items()},
            input_groups={name: tuple(int(i) for i in indices) for name, indices in payload.get("input_groups", {}).items()},
            metadata={key: value for key, value in payload.items() if key not in {"body_ids", "motor_groups", "input_groups"}},
        )

    def randomized(self, seed: int = 0) -> "ConnectomeGraph":
        """Configuration-model baseline with identical node/edge counts and degrees.

        Source and target stubs are independently permuted.  This preserves each
        node's in- and out-degree exactly (parallel edges are represented by their
        summed sparse value at runtime), while destroying biological pairings.
        """
        generator = np.random.default_rng(seed)
        sources = generator.permutation(self.edge_index[0])
        targets = generator.permutation(self.edge_index[1])
        return ConnectomeGraph(
            body_ids=self.body_ids.copy(),
            edge_index=np.vstack((sources, targets)),
            weights=self.weights.copy(),
            motor_groups=self.motor_groups,
            input_groups=self.input_groups,
            metadata={**self.metadata, "baseline": "degree_preserving_random", "random_seed": seed},
        )

    def _validate_groups(self, groups: dict[str, tuple[int, ...]], group_kind: str) -> None:
        for name, indices in groups.items():
            if not indices:
                raise GraphArtifactError(f"{group_kind} group {name!r} is empty")
            if min(indices) < 0 or max(indices) >= self.node_count:
                raise GraphArtifactError(f"{group_kind} group {name!r} contains an invalid node index")
