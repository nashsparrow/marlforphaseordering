from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import optuna

from experiment.pipeline import (
    ExperimentConfig,
    build_model,
    evaluate_model,
    load_config,
    make_compiler_env,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune PPO on validation benchmarks with reproducible Optuna trials."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--trial-timesteps", type=int, default=10_000)
    parser.add_argument("--evaluation-episodes", type=int, default=30)
    parser.add_argument(
        "--trial-seed",
        action="append",
        type=int,
        dest="trial_seeds",
        help="Repeat to evaluate every trial across multiple training seeds.",
    )
    parser.add_argument("--study-name", default="ppo_phase_ordering")
    parser.add_argument("--storage", default="sqlite:///optuna-study.db")
    parser.add_argument("--output", type=Path, default=Path("best_hyperparameters.json"))
    parser.add_argument("--jobs", type=int, default=1)
    return parser.parse_args()


def suggest_hyperparameters(trial: optuna.Trial) -> dict[str, float | int]:
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-2, log=True
        ),
        "n_steps": trial.suggest_int("n_steps", 128, 2048, step=128),
        "gamma": trial.suggest_float("gamma", 0.8, 0.99),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.5, 1.0),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
    }


def make_objective(
    base_config: ExperimentConfig,
    *,
    trial_timesteps: int,
    evaluation_episodes: int,
    trial_seeds: tuple[int, ...],
):
    if base_config.algorithm.upper() != "PPO":
        raise ValueError("This tuning script currently supports PPO only")

    def objective(trial: optuna.Trial) -> float:
        hyperparameters = suggest_hyperparameters(trial)
        scores: list[float] = []

        for seed_index, seed in enumerate(trial_seeds):
            seed_everything(seed)
            config = replace(
                base_config,
                seed=seed,
                total_timesteps=trial_timesteps,
                evaluation_episodes=evaluation_episodes,
                hyperparameters=hyperparameters,
            )
            training_env = make_compiler_env(
                config.train_benchmark,
                config.observation_space,
                config.max_episode_steps,
                seed,
            )
            validation_envs = [
                (
                    benchmark,
                    make_compiler_env(
                        benchmark,
                        config.observation_space,
                        config.max_episode_steps,
                        seed + benchmark_index + 1,
                    ),
                )
                for benchmark_index, benchmark in enumerate(
                    config.evaluation_benchmarks
                )
            ]

            try:
                model = build_model(config, training_env)
                model.learn(total_timesteps=trial_timesteps)
                actual_timesteps = model.num_timesteps
                for benchmark, validation_env in validation_envs:
                    from compiler_gym.errors import ServiceError

                    _, summary = evaluate_model(
                        model,
                        validation_env,
                        experiment=f"trial_{trial.number}",
                        seed=seed,
                        benchmark=benchmark,
                        timesteps=actual_timesteps,
                        episodes=evaluation_episodes,
                        deterministic=True,
                        service_error_types=(ServiceError,),
                    )
                    if summary.completed_episodes == 0:
                        raise optuna.TrialPruned(
                            f"No successful evaluations for {benchmark}"
                        )
                    scores.append(summary.reward_mean)
            finally:
                training_env.close()
                for _, validation_env in validation_envs:
                    validation_env.close()

            trial.report(
                sum(scores) / len(scores),
                step=seed_index,
            )
            if trial.should_prune():
                raise optuna.TrialPruned()

        score = sum(scores) / len(scores)
        if not math.isfinite(score):
            raise optuna.TrialPruned("Non-finite validation reward")
        return score

    return objective


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trial_seeds = tuple(args.trial_seeds or [config.seed])
    sampler = optuna.samplers.TPESampler(seed=config.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        make_objective(
            config,
            trial_timesteps=args.trial_timesteps,
            evaluation_episodes=args.evaluation_episodes,
            trial_seeds=trial_seeds,
        ),
        n_trials=args.trials,
        n_jobs=args.jobs,
    )

    result = {
        "study_name": study.study_name,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "trial_seeds": trial_seeds,
        "validation_benchmarks": config.evaluation_benchmarks,
        "hyperparameters": study.best_params,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
