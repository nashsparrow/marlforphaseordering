from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from experiment.pipeline import (
    ExperimentConfig,
    append_dataclasses,
    evaluate_model,
    load_config,
    load_model,
    make_compiler_env,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a saved Stable-Baselines3 model and evaluate it on demand."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Experiment JSON. If omitted, metadata.json beside the model is used.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        dest="benchmarks",
        help="Repeat to override the evaluation benchmarks from the saved config.",
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--name")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using deterministic predictions.",
    )
    return parser.parse_args()


def config_from_metadata(path: Path) -> ExperimentConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    config_data = data.get("config")
    if not isinstance(config_data, dict):
        raise ValueError(f"{path} does not contain an experiment config")
    if "evaluation_benchmarks" in config_data:
        config_data["evaluation_benchmarks"] = tuple(
            config_data["evaluation_benchmarks"]
        )
    config = ExperimentConfig(**config_data)
    config.validate()
    return config


def resolve_config(model_path: Path, config_path: Path | None) -> ExperimentConfig:
    if config_path is not None:
        return load_config(config_path)
    metadata_path = model_path.parent / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "No --config was supplied and metadata.json was not found beside "
            f"the model: {metadata_path}"
        )
    return config_from_metadata(metadata_path)


def default_output_dir(model_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return model_path.parent / "manual_evaluations" / timestamp


def run_saved_evaluation(
    *,
    model_path: Path,
    config: ExperimentConfig,
    benchmarks: tuple[str, ...],
    episodes: int,
    seed: int,
    deterministic: bool,
    output_dir: Path,
    evaluation_name: str,
    env_factory: Callable[[str, str, int, int], Any] = make_compiler_env,
) -> None:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not benchmarks:
        raise ValueError("At least one benchmark is required")
    seed_everything(seed)
    model = load_model(model_path, config.algorithm)
    timesteps = int(getattr(model, "num_timesteps", 0))
    output_dir.mkdir(parents=True, exist_ok=True)
    for generated_csv in ("evaluation_episodes.csv", "evaluation_summary.csv"):
        path = output_dir / generated_csv
        if path.exists():
            path.unlink()

    evaluation_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path.resolve()),
        "algorithm": config.algorithm,
        "observation_space": config.observation_space,
        "max_episode_steps": config.max_episode_steps,
        "benchmarks": benchmarks,
        "episodes": episodes,
        "seed": seed,
        "deterministic": deterministic,
        "model_timesteps": timesteps,
    }
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(evaluation_metadata, indent=2),
        encoding="utf-8",
    )

    for index, benchmark in enumerate(benchmarks):
        env = env_factory(
            benchmark,
            config.observation_space,
            config.max_episode_steps,
            seed + index,
        )
        try:
            from compiler_gym.errors import ServiceError

            records, summary = evaluate_model(
                model,
                env,
                experiment=evaluation_name,
                seed=seed,
                benchmark=benchmark,
                timesteps=timesteps,
                episodes=episodes,
                deterministic=deterministic,
                service_error_types=(ServiceError,),
            )
            append_dataclasses(output_dir / "evaluation_episodes.csv", records)
            append_dataclasses(output_dir / "evaluation_summary.csv", [summary])
        finally:
            env.close()


def main() -> None:
    args = parse_args()
    model_path = args.model
    if not model_path.exists() and model_path.suffix != ".zip":
        zipped_path = model_path.with_suffix(".zip")
        if zipped_path.exists():
            model_path = zipped_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    config = resolve_config(model_path, args.config)
    benchmarks = tuple(args.benchmarks or config.evaluation_benchmarks)
    seed = config.seed if args.seed is None else args.seed
    episodes = config.evaluation_episodes if args.episodes is None else args.episodes
    deterministic = (
        config.deterministic_evaluation if not args.stochastic else False
    )
    evaluation_name = args.name or f"{config.name}_loaded"
    output_dir = args.output_dir or default_output_dir(model_path)
    run_saved_evaluation(
        model_path=model_path,
        config=replace(config, seed=seed),
        benchmarks=benchmarks,
        episodes=episodes,
        seed=seed,
        deterministic=deterministic,
        output_dir=output_dir,
        evaluation_name=evaluation_name,
    )
    print(f"Evaluation written to {output_dir}")


if __name__ == "__main__":
    main()
