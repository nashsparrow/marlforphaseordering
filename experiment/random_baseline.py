from __future__ import annotations

import argparse
from pathlib import Path

from compiler_gym.errors import ServiceError

from experiment.pipeline import (
    append_dataclasses,
    evaluate_model,
    make_compiler_env,
    seed_everything,
)


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, deterministic=True):
        del observation, deterministic
        return self.action_space.sample(), None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate seeded random phase sequences under the same budget."
    )
    parser.add_argument(
        "--benchmark", action="append", required=True, dest="benchmarks"
    )
    parser.add_argument("--observation-space", default="Autophase")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/random"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    for index, benchmark in enumerate(args.benchmarks):
        env = make_compiler_env(
            benchmark,
            args.observation_space,
            args.max_episode_steps,
            args.seed + index,
        )
        try:
            records, summary = evaluate_model(
                RandomPolicy(env.action_space),
                env,
                experiment="random",
                seed=args.seed,
                benchmark=benchmark,
                timesteps=0,
                episodes=args.episodes,
                deterministic=False,
                service_error_types=(ServiceError,),
            )
            append_dataclasses(args.output_dir / "evaluation_episodes.csv", records)
            append_dataclasses(args.output_dir / "evaluation_summary.csv", [summary])
        finally:
            env.close()


if __name__ == "__main__":
    main()
