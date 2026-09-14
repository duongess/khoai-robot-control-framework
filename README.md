# Khoai robot control framework

This repository provides a generic Go environment runtime and a localhost Python Soft Actor-Critic (SAC) learner. The runtime batches task observations, requests continuous actions over gRPC, stores replay transitions, and performs training. It does not provide a robot-arm simulator, robot driver, command methods, or a browser API; those are supplied by a registered task in a downstream project.

## Connectome-constrained controller

The learner can use `mlp` (the existing Gaussian actor), `random_graph` (a degree-preserving random topology baseline), or `fly_connectome` (a fruit-fly **connectome-inspired**, directed sparse graph actor).

The graph topology is a biological inductive bias; SAC still trains the sensory encoder, neuron biases/leaks, decoder gains, uncertainty head, and optionally one gain per existing edge. It never creates graph edges. A connectome is mainly a wiring diagram, not a functional copy of a living fly brain. The robot mapping is an engineered experimental decoder, not a claim that selected fly neurons naturally control this robot arm.

```text
registered task / simulator
  -> numeric observation -> sparse connectome policy -> decoded action
  -> TaskDescriptor bounds + downstream robot safety validator -> robot command
  -> task / simulator
```

The framework rejects non-finite or out-of-range actions against the task's declared `ActionMin`/`ActionMax` before calling `Task.Step`. A downstream robot task must still enforce joint/workspace limits, collision and force sensing, acceleration, emergency-stop, and command-timeout rules when it implements `move_horizontal`, `move_vertical`, `grip`, `release`, and `stop`.

## neuPrint data setup

Install dependencies, then export a personal credential. Never put a real token in source control or command output.

```bash
poetry install
# Either export the token or put it in the ignored local `.env` file.
export NEUPRINT_TOKEN='...'
poetry run python -m ai.connectome inspect --output data/connectome --dataset manc:v1.2.3
```

`vnc:v1.0` is no longer advertised by neuprint.janelia.org. The current default, `manc:v1.2.3`, is an available adult male nerve-cord dataset. `inspect` first verifies the chosen dataset against the server, then saves the available neuron-property schema and a metadata sample. Review those files before copying and editing `configs/connectome_selection.example.json`. Use exact body IDs or terms demonstrably present in reviewed metadata. The tool deliberately does not assume an `.*MN.*` convention or hard-code neuron IDs.

```bash
cp configs/connectome_selection.example.json data/connectome/selection.json
# Edit every empty group after reviewing data/connectome/dataset_schema.json
poetry run python -m ai.connectome download --selection data/connectome/selection.json \
  --output data/connectome --max-neurons 256 --max-edges 5000
```

The download produces `nodes.parquet`, `edges.parquet`, `connectome_metadata.json`, and `connectome_graph.npz`. Metadata records server, dataset, selection manifest, directed edge orientation, weight normalization, and the body-ID-to-contiguous-index mapping. Incoming-weight normalization preserves recurrent edges. Training and inference load only this cache and never contact neuPrint.

The checked-in `.env.example` contains only `NEUPRINT_TOKEN=your_token_here`. `.env`, neuPrint credential paths, and generated cache payloads are ignored.

## Controller configuration and commands

`configs/fly_connectome.yaml` documents controller fields. The gRPC service reads equivalent environment variables, preserving its wire protocol and public Go APIs:

```bash
# MLP baseline (default)
LEARNER_CONTROLLER=mlp poetry run python -m ai

# Biological topology baseline
LEARNER_CONTROLLER=fly_connectome \
LEARNER_GRAPH_PATH=data/connectome/connectome_graph.npz \
LEARNER_PROPAGATION_STEPS=4 LEARNER_TRAIN_EDGE_GAINS=true \
LEARNER_ACTION_DEAD_ZONE=0.1 LEARNER_MAX_HORIZONTAL_SPEED=0.2 \
LEARNER_MAX_VERTICAL_SPEED=0.2 poetry run python -m ai

# Same graph size, edge count, in-degree, and out-degree baseline
LEARNER_CONTROLLER=random_graph LEARNER_GRAPH_PATH=data/connectome/connectome_graph.npz poetry run python -m ai
```

Training remains driven by the existing Go runtime, which sends replay batches to the same gRPC service. Deterministic inference uses the existing `PredictBatch` RPC. There is no standalone robot inference binary in this repository.

The observation mapping is exact but intentionally generic: the task's ordered `State` vector of dimension `LEARNER_STATE_DIM` passes through a trainable `Linear -> tanh -> Linear` sensory encoder, whose outputs are injected only into explicit `input_groups.sensory` body IDs in the cached artifact. This repository has no named arm observation schema, so it cannot truthfully label state index 0 as target position or a joint angle; a downstream task must document its vector ordering.

Required motor groups are `front_left`, `front_right`, `middle_left`, `middle_right`, `hind_left`, and `hind_right`. Their mean graph activities map to continuous actions as:

```text
horizontal = front_right - front_left
vertical   = middle_right - middle_left
gripper    = hind_left - hind_right
```

The decoder uses tanh, a configurable dead zone, output normalization, and configured speed/command limits. In an arm task, the three values correspond to left/right, down/up, and close/open. A discrete task adapter can threshold them into `MOVE_LEFT`, `MOVE_RIGHT`, `MOVE_UP`, `MOVE_DOWN`, `GRIP`, `RELEASE`, or `NO_OP`; the present repository's interface is continuous only.

## Evaluation protocol

Evaluate all three controllers using the same downstream task, reward function, seeds, training budget, and episode count. Record episode reward, success rate, collision count (from the task safety layer), steps to a stated success threshold, inference latency, process memory, observation-noise robustness, and object-position-shift robustness. Do not claim a connectome advantage unless measurements demonstrate it. The framework has no arm environment, so it cannot fabricate these task-specific metrics locally.

## Tests and troubleshooting

```bash
bash scripts/generate-proto.sh
poetry run pytest -q
go test ./...
```

- The MVP default is limited to 256 nodes/5,000 edges; do not download an entire VNC graph.
- `NEUPRINT_TOKEN is required` means export the token for the inspection/download process. It is not read from an arbitrary credential file.
- A dataset-unavailable error is an explicit verification failure; choose only a dataset the client advertises, pass it with `--dataset`, and record that choice in the cached metadata.
- Empty motor/input groups are errors, not a silent fallback to arbitrary neurons.
- Sparse runtime grows with edge count, propagation steps, and batch size.
