"""Run a paired Parametric SAC ablation on the real force-control task.

From this directory: python run_comparison_demo.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from time import perf_counter_ns

_project_python = Path(__file__).resolve().parent / ".venv/bin/python"
if _project_python.is_file() and Path(sys.prefix).resolve() != _project_python.parent.parent.resolve():
    os.execv(str(_project_python), [str(_project_python), *sys.argv])

import numpy as np
import torch

from ai.config import SACConfig
from ai.connectome.dynamic_policy import DEFAULT_GANTRY_REFLEX_CONFIG
from ai.connectome.graph import ConnectomeGraph
from ai.sac import SACAgent, TensorBatch


HERE = Path(__file__).resolve().parent
VISUALIZER = HERE.parent / "khoai-robot-visualizer-web"
GRAPH = HERE / "data/connectome/connectome_graph.npz"
FEATURE_EXTRACTORS = {
    "dense_mlp": "parametric_mlp",
    "sparse_connectome": "fly_connectome",
}


@dataclass(frozen=True)
class Scenario:
    seed: int
    friction: float
    position_offset: float


def scenario(seed: int) -> Scenario:
    rng = random.Random(seed)
    return Scenario(seed, 0.35 + rng.uniform(-0.06, 0.06), rng.uniform(-0.35, 0.35))


class ForceControlBridge:
    def __init__(self, executable: Path) -> None:
        self.process = subprocess.Popen(
            [str(executable)], cwd=VISUALIZER, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )

    def call(self, payload: dict) -> dict:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"force-control bridge exited with code {self.process.poll()}")
        result = json.loads(line)
        if result.get("error"):
            raise RuntimeError(f"force-control bridge: {result['error']}")
        return result

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
        self.process.communicate(timeout=10)

    def __enter__(self) -> ForceControlBridge:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def build_bridge(destination: Path) -> Path:
    if not (VISUALIZER / "cmd/benchmark-bridge/main.go").is_file():
        raise FileNotFoundError(f"force-control simulator bridge is missing in {VISUALIZER}")
    executable = destination / "force-control-benchmark"
    environment = os.environ.copy()
    environment["GOCACHE"] = str(destination / "go-cache")
    subprocess.run(["go", "build", "-o", str(executable), "./cmd/benchmark-bridge"],
                   cwd=VISUALIZER, env=environment, check=True)
    return executable


def make_agent(feature_extractor_type: str, seed: int) -> SACAgent:
    if feature_extractor_type not in FEATURE_EXTRACTORS:
        raise ValueError(f"unknown feature_extractor_type: {feature_extractor_type}")
    config = SACConfig(
        state_dim=30, action_dim=3, hidden_dim=128, seed=seed,
        controller_type=FEATURE_EXTRACTORS[feature_extractor_type],
        graph_path=str(GRAPH) if feature_extractor_type == "sparse_connectome" else None,
        full_actor_unlock_step=0, base_warmup_steps=0,
        phase_gated_decoder=False,
        slip_severity_observation_index=17, grip_force_observation_index=16,
        vertical_acceleration_observation_index=18, previous_vertical_action_observation_index=26,
    )
    agent = SACAgent(config)
    agent.register_reflex_law(DEFAULT_GANTRY_REFLEX_CONFIG)
    return agent


def batch_from_replay(replay: deque, indices: np.ndarray) -> TensorBatch:
    items = [replay[int(index)] for index in indices]
    return TensorBatch(
        states=torch.tensor(np.stack([item[0] for item in items]), dtype=torch.float32),
        actions=torch.tensor(np.stack([item[1] for item in items]), dtype=torch.float32),
        rewards=torch.tensor([[item[2]] for item in items], dtype=torch.float32),
        next_states=torch.tensor(np.stack([item[3] for item in items]), dtype=torch.float32),
        dones=torch.tensor([[item[4]] for item in items], dtype=torch.float32),
    )


def run_episode(
    bridge: ForceControlBridge, agent: SACAgent, spec: Scenario, *,
    train: bool, replay: list | None = None, sampler: np.random.Generator | None = None,
    batch_size: int = 64, update_every: int = 4,
) -> dict:
    response = bridge.call({"command": "reset", **asdict(spec)})
    state = np.asarray(response["state"], dtype=np.float32)
    time_step = float(response["time_step"])
    max_grip_force = float(response["max_grip_force"])
    reward_total = 0.0
    peak_force = 0.0
    slip_steps = 0
    latencies_ms: list[float] = []
    steps = 0
    while True:
        started = perf_counter_ns()
        action = agent.act(torch.from_numpy(state).unsqueeze(0), deterministic=not train)[0].numpy()
        latencies_ms.append((perf_counter_ns() - started) / 1e6)
        response = bridge.call({"command": "step", "action": action.tolist()})
        next_state = np.asarray(response["state"], dtype=np.float32)
        info = response["info"]
        reward = float(response.get("reward", 0.0))
        done = bool(response.get("done", False))
        steps += 1
        reward_total += reward
        # Observation 16 is normalize(grip force, 0, MaxGripForce).
        peak_force = max(peak_force, (float(next_state[16]) + 1.0) * max_grip_force / 2.0)
        slip_steps += int(info["slipping"] > 0.5)
        if train:
            assert replay is not None and sampler is not None
            if len(replay) == 100_000:
                replay.pop(0)
            replay.append((state, action, reward, next_state, float(done)))
            if len(replay) >= batch_size and steps % update_every == 0:
                sample_indices = sampler.integers(0, len(replay), size=batch_size)
                agent.update(batch_from_replay(replay, sample_indices))
        state = next_state
        if done:
            break
    return {
        "seed": spec.seed, "friction": spec.friction, "position_offset": spec.position_offset,
        "reward": reward_total, "success": bool(response.get("success", False)),
        "peak_force_n": peak_force, "slip_steps": slip_steps, "steps": steps,
        "cycle_time_s": steps * time_step,
        "inference_latency_ms": float(np.mean(latencies_ms)),
    }


def write_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")


def summarize(rows: list[dict]) -> dict:
    return {
        "success_rate_pct": 100 * sum(row["success"] for row in rows) / len(rows),
        "peak_force_n": max(row["peak_force_n"] for row in rows),
        "slip_rate_pct": 100 * sum(row["slip_steps"] for row in rows) / sum(row["steps"] for row in rows),
        "average_cycle_time_s": float(np.mean([row["cycle_time_s"] for row in rows])),
        "inference_latency_ms": sum(row["inference_latency_ms"] * row["steps"] for row in rows) / sum(row["steps"] for row in rows),
    }


def plot_learning_curves(training: dict[str, list[dict]], path: Path) -> None:
    config_dir = path.parent / ".matplotlib"
    config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, rows in training.items():
        episodes = np.arange(1, len(rows) + 1)
        rewards = np.array([row["reward"] for row in rows])
        successes = np.array([row["success"] for row in rows], dtype=float)
        window = min(20, len(rows))
        reward_curve = [rewards[max(0, i-window+1):i+1].mean() for i in range(len(rows))]
        success_curve = [100*successes[max(0, i-window+1):i+1].mean() for i in range(len(rows))]
        axes[0].plot(episodes, reward_curve, label=label)
        axes[1].plot(episodes, success_curve, label=label)
    axes[0].set(xlabel="Training episode", ylabel="Reward", title="Episode reward")
    axes[1].set(xlabel="Training episode", ylabel="Success rate (%)", title="Rolling success rate", ylim=(0, 100))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data/comparison_demo")
    args = parser.parse_args()
    if min(args.train_episodes, args.eval_episodes, args.batch_size, args.update_every) < 1:
        parser.error("episode counts, batch size, and update interval must be positive")
    if importlib.util.find_spec("matplotlib") is None:
        parser.error("Matplotlib is required; install the project dependencies with poetry install")
    ConnectomeGraph.load(GRAPH)
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_specs = [scenario(args.seed + i) for i in range(args.train_episodes)]
    test_specs = [scenario(args.seed + 10_000 + i) for i in range(args.eval_episodes)]
    (output / "protocol.json").write_text(json.dumps({
        "training_scenarios": [asdict(item) for item in train_specs],
        "test_scenarios": [asdict(item) for item in test_specs],
        "reflex_law": DEFAULT_GANTRY_REFLEX_CONFIG,
        "feature_extractors": FEATURE_EXTRACTORS,
    }, indent=2) + "\n")
    training: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="force-control-benchmark-") as temp:
        executable = build_bridge(Path(temp))
        with ForceControlBridge(executable) as bridge:
            for feature_extractor_type in FEATURE_EXTRACTORS:
                label = feature_extractor_type
                print(f"Training {label} ({args.train_episodes} episodes)", flush=True)
                arm_dir = output / label
                arm_dir.mkdir(parents=True, exist_ok=True)
                train_log = arm_dir / "train.jsonl"
                test_log = arm_dir / "test.jsonl"
                train_log.write_text("")
                test_log.write_text("")
                agent = make_agent(label, args.seed)
                replay: list = []
                sampler = np.random.default_rng(args.seed)
                training[label] = []
                for number, spec in enumerate(train_specs, start=1):
                    row = run_episode(bridge, agent, spec, train=True, replay=replay,
                                      sampler=sampler, batch_size=args.batch_size,
                                      update_every=args.update_every)
                    training[label].append(row)
                    write_jsonl(train_log, {"episode": number, **row})
                    if number % 10 == 0 or number == args.train_episodes:
                        print(f"  {label}: episode {number}/{args.train_episodes}", flush=True)
                torch.save({"agent": agent.checkpoint_state(), "reflex_law": DEFAULT_GANTRY_REFLEX_CONFIG}, arm_dir / "checkpoint.pt")
                print(f"Evaluating {label} ({args.eval_episodes} paired episodes)", flush=True)
                test_rows = []
                for number, spec in enumerate(test_specs, start=1):
                    row = run_episode(bridge, agent, spec, train=False)
                    test_rows.append(row)
                    write_jsonl(test_log, {"episode": number, **row})
                summaries[label] = summarize(test_rows)
                (arm_dir / "summary.json").write_text(json.dumps(summaries[label], indent=2) + "\n")
    plot_learning_curves(training, output / "comparison_plot.png")
    print("\n| Pipeline | Success Rate (%) | Peak Force (N) | Slip Rate (%) | Average Cycle Time (s) | Inference Latency (ms) |")
    print("|---|---:|---:|---:|---:|---:|")
    for label, metric in summaries.items():
        print(f"| {label} | {metric['success_rate_pct']:.1f} | {metric['peak_force_n']:.2f} | "
              f"{metric['slip_rate_pct']:.1f} | {metric['average_cycle_time_s']:.2f} | "
              f"{metric['inference_latency_ms']:.3f} |")
    print(f"\nArtifacts: {output}")


if __name__ == "__main__":
    main()
