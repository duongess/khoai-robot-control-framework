# Learner Protocol

The local `LearnerService` exposes batched prediction/training and a local
checkpoint operation:

- `PredictBatch` returns actions from an immutable episode-policy snapshot.
- `TrainBatch` updates the live SAC actor and critics, then publishes the next
  snapshot version.
- `HealthCheck` reports learner readiness and its active named model, if any.
- `SaveCheckpoint` atomically writes the full SAC state. A non-empty
  `model_name` creates or overwrites that identifier; an empty name overwrites
  the active model only when the learner was launched with one.

Model names are identifiers, not file paths. The local implementation accepts
letters, digits, `.`, `_`, and `-`, writes beneath `LEARNER_CHECKPOINT_DIR`
(`data/models` by default), and rejects traversal attempts. A checkpoint
contains actor/critic/target networks, all optimizer state, entropy
temperature, RNG state, and policy/training counters. The Go replay buffer is
owned by the simulator runtime and is intentionally not part of this RPC.
