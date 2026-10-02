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
LEARNER_BASE_POLICY=connectome \
LEARNER_PROPAGATION_STEPS=4 LEARNER_TRAIN_EDGE_GAINS=true \
LEARNER_PHASE_GATED_DECODER=true \
LEARNER_OBJECT_ATTACHED_OBSERVATION_INDEX=15 \
LEARNER_PHASE_OBSERVATION_INDEX=19 \
LEARNER_TRANSPORT_PHASE_THRESHOLD=0.0 \
LEARNER_ACTION_DEAD_ZONE=0.001 LEARNER_MAX_HORIZONTAL_SPEED=0.2 \
LEARNER_MAX_VERTICAL_SPEED=0.2 poetry run python -m ai

# Same graph size, edge count, in-degree, and out-degree baseline
LEARNER_CONTROLLER=random_graph LEARNER_GRAPH_PATH=data/connectome/connectome_graph.npz poetry run python -m ai
```

For a responsive local dashboard, the learner defaults to one PyTorch compute
thread, one inter-op thread, and retains at most 128 immutable episode-policy
snapshots. These limits prevent native worker-thread and snapshot-memory growth
from starving browser rendering. Override them only when the machine has spare
capacity:

```bash
LEARNER_TORCH_NUM_THREADS=1 \
LEARNER_TORCH_NUM_INTEROP_THREADS=1 \
LEARNER_GAMMA=0.999 \
LEARNER_MAX_POLICY_SNAPSHOTS=128 \
poetry run python -m ai
```

The force-control environment advances at 0.1 seconds per physics step and
uses a 30-second secure-hold curriculum criterion. `LEARNER_GAMMA=0.999`
keeps delayed object-break consequences meaningful across that interval;
`0.99` has an effective horizon of roughly ten seconds at this control rate.

Training remains driven by the existing Go runtime, which sends replay batches to the same gRPC service. Deterministic inference uses the existing `PredictBatch` RPC. There is no standalone robot inference binary in this repository.

## Save and resume a local model

The dashboard's **Save Model** control writes an atomic local checkpoint. It
contains the SAC actor, both critics and target critics, all four optimizer
states, entropy temperature, RNG state, and learner/policy counters. Saving
does not pause the simulation and overwrites only the named model's previous
checkpoint after the new file is complete.

```bash
# Start a brand-new, unnamed learner. Give it a name in the dashboard before
# its first Save Model action.
poetry run python -m ai

# If data/models/grasp-v1.pt exists, restore and continue it. Otherwise start
# a brand-new learner named grasp-v1; Save Model will create/overwrite it.
poetry run python -m ai grasp-v1
```

Names may contain letters, digits, `.`, `_`, and `-` only; they are never file
paths. Checkpoints default to the ignored `data/models/` directory, or set
`LEARNER_CHECKPOINT_DIR` to choose another local directory. Load only
checkpoints you created or trust. To resume a saved run, start the learner with
its name first, then start the Go simulator so every worker begins a fresh
episode on the restored policy. The Go replay buffer is deliberately not
checkpointed: it belongs to the short-lived simulator runtime and old
transitions can be incompatible after an environment or reward change.

The force-control task supplies a 30-value observation. It passes through the
connectome sensory encoder and sparse graph to one shared latent. A learned
connectome base head emits the nominal three-axis action; SAC receives the same
latent plus normalized slip severity (17), grip force (16), vertical
acceleration (18), and previous vertical action (26). The learner returns all
three vectors in one snapshot:

```text
a_final = clamp(a_fly_base + alpha * delta_a_sac, -1, 1)
```

The default normalized alpha is `[0.20, 0.15, 0.30]`. Set `base_policy: connectome` (the default for graph controllers) to make the sparse graph plus learned base head produce the nominal action; set `base_policy: closed_loop` only for the legacy geometry-reflex ablation.
The grip value is kept
normalized because the Go task owns unit conversion and the material safety
limit; with its default 12 N/s actuator, `0.30` is approximately `3.6 N/s` of
residual authority. For the first `LEARNER_BASE_WARMUP_STEPS` gradient updates
(128 by default), the residual mean head and log standard deviation are frozen
while its fixed Gaussian noise supplies exploration; SAC therefore teaches the
fly-base path first. Afterwards the residual head trains at full rate and the
base/backbone optimizer group drops to `0.1x`. This is a decoupled warmup and
fine-tuning schedule without a teacher or runtime FSM controller.

This actor uses checkpoint format 3 and force-control schema 33. Start a new
model name and a fresh Go process/replay buffer; pre-dual-loop checkpoints are
rejected instead of being partially restored.

Required graph readout groups are `front_left`, `front_right`, `middle_left`,
`middle_right`, `hind_left`, and `hind_right`. Their six mean activities are
projected into the shared latent; the learned base/residual heads—not a
hard-coded antagonist or FSM mapping—produce the production action. The old
left/right difference decoder remains only as a diagnostic helper in tests. The
geometry reflex is not part of the default connectome path.

Both heads use tanh and preserve the existing continuous three-value action
contract: left/right, down/up, and signed grip-force rate. The Go environment
does not add a scripted FSM command; it only applies workspace, slew-rate, and
`F_break - 0.5 N` safety bounds.

For the 30-dimensional force-control task, `LEARNER_PHASE_GATED_DECODER=true`
keeps one sparse graph core but adds two learned motor readouts: an
acquisition readout until an object is securely attached, and a transport
readout once the attached-object phase reaches `MoveToTarget`.  Their outputs
are selected by state; they are never averaged. This configuration changes the
actor checkpoint schema, so begin a new named model and fresh Go replay buffer
instead of resuming an older checkpoint.

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
