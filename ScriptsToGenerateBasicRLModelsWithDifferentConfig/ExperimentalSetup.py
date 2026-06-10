from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.pipeline import ExperimentConfig, load_config, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a reproducible CompilerGym RL experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON configuration file. Command-line defaults are used when omitted.",
    )
    parser.add_argument("--name", default="ppo_bzip2")
    parser.add_argument("--algorithm", choices=("PPO", "A2C", "DQN"), default="PPO")
    parser.add_argument("--train-benchmark", default="cbench-v1/bzip2")
    parser.add_argument(
        "--evaluation-benchmark",
        action="append",
        dest="evaluation_benchmarks",
        help="Repeat for each validation/test benchmark.",
    )
    parser.add_argument("--observation-space", default="Autophase")
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--evaluation-interval", type=int, default=1_000)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=40)
    parser.add_argument("--layer-nodes", type=int, default=128)
    parser.add_argument("--layer-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save a model ZIP after every evaluation checkpoint.",
    )
    parser.add_argument(
        "--stochastic-evaluation",
        action="store_true",
        help="Sample policy actions instead of using deterministic predictions.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.config:
        return load_config(args.config)
    benchmarks = tuple(args.evaluation_benchmarks or [args.train_benchmark])
    return ExperimentConfig(
        name=args.name,
        algorithm=args.algorithm,
        train_benchmark=args.train_benchmark,
        evaluation_benchmarks=benchmarks,
        observation_space=args.observation_space,
        total_timesteps=args.total_timesteps,
        evaluation_interval=args.evaluation_interval,
        evaluation_episodes=args.evaluation_episodes,
        max_episode_steps=args.max_episode_steps,
        layer_nodes=args.layer_nodes,
        layer_count=args.layer_count,
        seed=args.seed,
        deterministic_evaluation=not args.stochastic_evaluation,
        save_checkpoints=args.save_checkpoints,
        output_dir=args.output_dir,
    )


def main() -> None:
    run_experiment(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
