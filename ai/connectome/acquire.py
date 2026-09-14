"""Explicit neuPrint inspection, download, and preprocessing commands.

This is the only module allowed to contact neuprint.janelia.org.  It deliberately
requires a prior metadata inspection and an explicit selection manifest: VNC
annotations must be examined before deciding which neurons represent an
experimental leg-control group.  In particular, it never uses ``.*MN.*``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ai.connectome.graph import ConnectomeGraph, GraphArtifactError

SERVER = "neuprint.janelia.org"
# ``vnc:v1.0`` was the originally requested dataset name, but it is no longer
# advertised by neuprint.janelia.org.  MANC is the available adult male nerve
# cord dataset; the inspect command still verifies this server-side before any
# query or download is performed.
DEFAULT_DATASET = "manc:v1.2.3"
ANNOTATION_FIELDS = ("bodyId", "type", "instance", "status", "statusLabel", "somaSide", "rootSide", "somaNeuromere", "class", "subclass", "superclass", "hemilineage", "entryNerve", "exitNerve", "label")


def load_neuprint_token(environ: dict[str, str] | None = None) -> str:
    """Read a non-empty token only from ``NEUPRINT_TOKEN``; never print it."""
    # The checked-in example documents this variable in a local, ignored `.env`.
    # Explicit test/caller mappings remain isolated from the process environment.
    if environ is None:
        load_dotenv()
    source = os.environ if environ is None else environ
    token = source.get("NEUPRINT_TOKEN", "").strip()
    if not token or token == "your_token_here":
        raise RuntimeError("NEUPRINT_TOKEN is required; export it before running a neuPrint command")
    return token


def create_verified_client(dataset: str = DEFAULT_DATASET):
    """Create a client and prove the requested dataset is advertised by the server."""
    try:
        from neuprint import Client
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise RuntimeError("neuprint-python is required; install project dependencies") from error
    client = Client(SERVER, dataset=dataset, token=load_neuprint_token(), progress=False)
    datasets = client.fetch_datasets()
    if dataset not in datasets:
        available = ", ".join(sorted(datasets))
        raise RuntimeError(f"neuPrint dataset {dataset!r} is unavailable; server advertised: {available}")
    return client, datasets[dataset]


def inspect_dataset(output_dir: str | Path, dataset: str = DEFAULT_DATASET, sample_size: int = 1_000) -> dict[str, Any]:
    """Cache available schema keys and a metadata sample before selection."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    client, dataset_info = create_verified_client(dataset)
    keys = list(client.fetch_neuron_keys())
    fields = [field for field in ANNOTATION_FIELDS if field in keys]
    if "bodyId" not in fields:
        raise RuntimeError("dataset does not expose required Neuron bodyId metadata")
    returns = ", ".join(f"n.`{field}` AS `{field}`" for field in fields)
    sample = client.fetch_custom(f"MATCH (n:Neuron) RETURN {returns} ORDER BY n.bodyId LIMIT {int(sample_size)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(output_dir / "metadata_sample.parquet", index=False)
    summary = {
        "dataset": dataset,
        "server": SERVER,
        "dataset_info": dataset_info,
        "neuron_properties": keys,
        "returned_annotation_fields": fields,
        "sample_size": len(sample),
        "inspection_query": f"MATCH (n:Neuron) RETURN {returns} ORDER BY n.bodyId LIMIT {int(sample_size)}",
        "selection_rule": "Use an explicit manifest based on inspected exact annotations or body IDs; no MN regex is applied.",
    }
    (output_dir / "dataset_schema.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return summary


def preprocess_selected_network(
    nodes: pd.DataFrame,
    connections: pd.DataFrame,
    selection: dict[str, Any],
    output_dir: str | Path,
    max_neurons: int = 256,
    max_edges: int = 5_000,
    provenance: dict[str, Any] | None = None,
) -> ConnectomeGraph:
    """Normalize and cache an explicitly selected directed subgraph.

    The function is intentionally independent of neuPrint so it can be tested with
    a fixture and used to reproduce a previously downloaded query.
    """
    if max_neurons <= 0 or max_edges <= 0:
        raise ValueError("max_neurons and max_edges must be positive")
    required_nodes = {"bodyId"}
    required_edges = {"bodyId_pre", "bodyId_post", "weight"}
    if not required_nodes.issubset(nodes) or not required_edges.issubset(connections):
        raise GraphArtifactError("nodes require bodyId; connections require bodyId_pre, bodyId_post, weight")
    nodes = nodes.copy()
    connections = connections.copy()
    nodes["bodyId"] = pd.to_numeric(nodes["bodyId"], errors="raise").astype("int64")
    connections = connections.loc[:, ["bodyId_pre", "bodyId_post", "weight"]].copy()
    for column in ("bodyId_pre", "bodyId_post"):
        connections[column] = pd.to_numeric(connections[column], errors="raise").astype("int64")
    connections["weight"] = pd.to_numeric(connections["weight"], errors="raise")
    connections = connections.loc[np.isfinite(connections["weight"]) & (connections["weight"] > 0)]
    selected = _apply_selector(nodes, selection.get("nodes", {}))
    if selected.empty:
        raise GraphArtifactError("selection manifest matched no neurons")
    selected = selected.drop_duplicates("bodyId")
    # Keep the most connected selected bodies, with body ID as a reproducible tie-breaker.
    degree = pd.concat((connections.groupby("bodyId_pre")["weight"].sum(), connections.groupby("bodyId_post")["weight"].sum()), axis=1).fillna(0).sum(axis=1)
    selected["_degree"] = selected["bodyId"].map(degree).fillna(0)
    selected = selected.sort_values(["_degree", "bodyId"], ascending=[False, True]).head(max_neurons).drop(columns="_degree")
    body_set = set(selected["bodyId"])
    edges = connections[connections["bodyId_pre"].isin(body_set) & connections["bodyId_post"].isin(body_set)]
    edges = edges.groupby(["bodyId_pre", "bodyId_post"], as_index=False, sort=True)["weight"].sum()
    edges = edges.sort_values(["weight", "bodyId_pre", "bodyId_post"], ascending=[False, True, True]).head(max_edges)
    if edges.empty:
        raise GraphArtifactError("selected neurons have no positive internal connections")
    # Node order is a durable contiguous index mapping, independent of query order.
    selected = selected.sort_values("bodyId").reset_index(drop=True)
    mapping = {int(body_id): index for index, body_id in enumerate(selected["bodyId"])}
    edges = edges.sort_values(["bodyId_pre", "bodyId_post"]).reset_index(drop=True)
    incoming = edges.groupby("bodyId_post")["weight"].transform("sum")
    edges["normalized_weight"] = (edges["weight"] / incoming).astype("float32")
    graph = ConnectomeGraph(
        body_ids=selected["bodyId"].to_numpy(dtype=np.int64),
        edge_index=np.array([[mapping[int(body)] for body in edges["bodyId_pre"]], [mapping[int(body)] for body in edges["bodyId_post"]]], dtype=np.int64),
        weights=edges["normalized_weight"].to_numpy(dtype=np.float32),
        motor_groups=_resolve_groups(selected, selection.get("motor_groups", {}), mapping, "motor"),
        input_groups=_resolve_groups(selected, selection.get("input_groups", {}), mapping, "input"),
        metadata={"selection": selection, "provenance": provenance or {}, "node_limit": max_neurons, "edge_limit": max_edges},
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output_dir / "nodes.parquet", index=False)
    edges.to_parquet(output_dir / "edges.parquet", index=False)
    graph.save(output_dir)
    return graph


def download_and_preprocess(selection_path: str | Path, output_dir: str | Path, dataset: str = DEFAULT_DATASET, max_neurons: int = 256, max_edges: int = 5_000) -> ConnectomeGraph:
    """Download metadata and internal directed connections for a reviewed manifest."""
    output_dir = Path(output_dir)
    schema_path = output_dir / "dataset_schema.json"
    if not schema_path.is_file():
        raise RuntimeError("run `python -m ai.connectome inspect` first; selection must follow schema inspection")
    selection = json.loads(Path(selection_path).read_text())
    client, dataset_info = create_verified_client(dataset)
    from neuprint import NeuronCriteria, fetch_adjacencies, fetch_neurons

    node_selector = selection.get("nodes", {})
    explicit_body_ids = node_selector.get("body_ids")
    if explicit_body_ids:
        # A reviewed exact-ID manifest must not trigger a full-dataset metadata
        # download. This is the normal MVP path and keeps acquisition bounded.
        requested_metadata_ids = [int(body_id) for body_id in explicit_body_ids]
        nodes = fetch_neurons(
            NeuronCriteria(bodyId=requested_metadata_ids), omit_rois=True, client=client
        )
    else:
        # Annotation-based selectors remain supported for a reviewed schema, but
        # necessarily require metadata discovery before local filtering.
        nodes = fetch_neurons(client=client, omit_rois=True)
    reviewed = _apply_selector(nodes, selection.get("nodes", {}))
    if reviewed.empty:
        raise GraphArtifactError("selection manifest matched no metadata rows")
    requested_ids = reviewed["bodyId"].astype("int64").tolist()
    _, connections = fetch_adjacencies(requested_ids, requested_ids, omit_rois=True, weight_props=["weight"], client=client)
    # Keep the full metadata query: adjacency helpers only promise a small property
    # subset, whereas a reviewed group may depend on available side/leg annotations.
    return preprocess_selected_network(reviewed, connections, selection, output_dir, max_neurons, max_edges, {
        "server": SERVER,
        "dataset": dataset,
        "dataset_info": dataset_info,
        "selection_manifest": str(Path(selection_path)),
        "schema_file": str(schema_path),
        "connection_query": "fetch_adjacencies(selected_body_ids, selected_body_ids, omit_rois=True, weight_props=['weight'])",
    })


def _apply_selector(nodes: pd.DataFrame, selector: dict[str, Any]) -> pd.DataFrame:
    body_ids = selector.get("body_ids")
    contains = selector.get("field_contains", {})
    if body_ids is None and not contains:
        raise GraphArtifactError("selection requires body_ids or field_contains based on inspected metadata")
    result = nodes
    if body_ids is not None:
        ids = {int(value) for value in body_ids}
        result = result[result["bodyId"].isin(ids)]
    for field, terms in contains.items():
        if field not in result.columns:
            raise GraphArtifactError(f"selection refers to unavailable annotation field {field!r}")
        terms = [terms] if isinstance(terms, str) else terms
        values = result[field].fillna("").astype(str).str.casefold()
        for term in terms:
            result = result[values.str.contains(str(term).casefold(), regex=False)]
            values = result[field].fillna("").astype(str).str.casefold()
    return result.copy()


def _resolve_groups(nodes: pd.DataFrame, group_selectors: dict[str, dict[str, Any]], mapping: dict[int, int], kind: str) -> dict[str, tuple[int, ...]]:
    if not group_selectors:
        raise GraphArtifactError(f"selection manifest must define non-empty {kind}_groups")
    groups: dict[str, tuple[int, ...]] = {}
    for name, selector in group_selectors.items():
        rows = _apply_selector(nodes, selector)
        indices = tuple(mapping[int(body)] for body in rows["bodyId"] if int(body) in mapping)
        if not indices:
            raise GraphArtifactError(f"{kind} group {name!r} is empty after node limiting")
        groups[name] = indices
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and cache a small neuPrint VNC connectome subgraph.")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="verify dataset and cache schema/sample metadata")
    inspect.add_argument("--output", default="data/connectome")
    inspect.add_argument("--dataset", default=DEFAULT_DATASET)
    inspect.add_argument("--sample-size", type=int, default=1_000)
    download = commands.add_parser("download", help="download a reviewed manifest and produce cache artifacts")
    download.add_argument("--selection", required=True)
    download.add_argument("--output", default="data/connectome")
    download.add_argument("--dataset", default=DEFAULT_DATASET)
    download.add_argument("--max-neurons", type=int, default=256)
    download.add_argument("--max-edges", type=int, default=5_000)
    arguments = parser.parse_args()
    try:
        if arguments.command == "inspect":
            inspect_dataset(arguments.output, arguments.dataset, arguments.sample_size)
        else:
            download_and_preprocess(arguments.selection, arguments.output, arguments.dataset, arguments.max_neurons, arguments.max_edges)
    except (GraphArtifactError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
