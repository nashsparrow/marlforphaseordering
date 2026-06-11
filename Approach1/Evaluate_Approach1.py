from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Approach1.approach1_pipeline import (
    CandidateSelectorEnv,
    load_trained_approach1,
    make_objective_env,
)
from experiment.pipeline import append_dataclasses, evaluate_model, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reload and evaluate a completed Approach 1 run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark", action="append", dest="benchmarks")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, candidates, main_model, timesteps = load_trained_approach1(args.run_dir)
    benchmarks = tuple(args.benchmarks or config.evaluation_benchmarks)
    episodes = args.episodes or config.main.evaluation_episodes
    seed = config.seed if args.seed is None else args.seed
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or args.run_dir / "manual_evaluations" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(args.run_dir.resolve()),
                "benchmarks": benchmarks,
                "episodes": episodes,
                "seed": seed,
                "deterministic": not args.stochastic,
                "model_timesteps": timesteps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for index, benchmark in enumerate(benchmarks):
        env = CandidateSelectorEnv(
            make_objective_env(
                benchmark=benchmark,
                observation_space=config.observation_space,
                max_episode_steps=config.main.max_episode_steps,
                seed=seed + index,
                objective=config.main.objective,
                runtime_reward=config.runtime_reward,
            ),
            candidates,
            config.main.deterministic_candidates,
        )
        try:
            from compiler_gym.errors import ServiceError

            records, summary = evaluate_model(
                main_model,
                env,
                experiment=f"{config.name}_loaded",
                seed=seed,
                benchmark=benchmark,
                timesteps=timesteps,
                episodes=episodes,
                deterministic=not args.stochastic,
                service_error_types=(ServiceError,),
            )
            append_dataclasses(output_dir / "evaluation_episodes.csv", records)
            append_dataclasses(output_dir / "evaluation_summary.csv", [summary])
        finally:
            env.close()

    print(f"Evaluation written to {output_dir}")


if __name__ == "__main__":
    main()
