from __future__ import annotations

import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import gym
import numpy as np
from gym import spaces

from Approach1.approach1_pipeline import CandidateSelectorEnv
from experiment.pipeline import (
    ExperimentConfig,
    append_dataclasses,
    build_model,
    evaluate_model,
    load_model,
    make_compiler_env,
    package_versions,
    seed_everything,
)


@dataclass(frozen=True)
class CandidateTrainingConfig:
    algorithm: str = "PPO"
    total_timesteps: int = 20_000
    max_episode_steps: int = 40
    layer_nodes: int = 128
    layer_count: int = 4
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("Candidate total_timesteps must be positive")
        if self.max_episode_steps <= 0:
            raise ValueError("Candidate max_episode_steps must be positive")
        if self.layer_nodes < 0 or self.layer_count <= 0:
            raise ValueError("Candidate layer configuration is invalid")


@dataclass(frozen=True)
class MainTrainingConfig:
    algorithm: str = "PPO"
    total_timesteps: int = 20_000
    evaluation_interval: int = 1_000
    evaluation_episodes: int = 100
    max_episode_steps: int = 40
    layer_nodes: int = 128
    layer_count: int = 4
    deterministic_candidates: bool = True
    deterministic_evaluation: bool = True
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for name in (
            "total_timesteps",
            "evaluation_interval",
            "evaluation_episodes",
            "max_episode_steps",
            "layer_count",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"Main agent {name} must be positive")
        if self.layer_nodes < 0:
            raise ValueError("Main agent layer_nodes cannot be negative")


@dataclass(frozen=True)
class Approach2Config:
    name: str
    train_benchmark: str
    evaluation_benchmarks: tuple[str, ...]
    observation_space: str
    subset_count: int
    seed: int
    subset_seed: int
    output_dir: str
    candidate: CandidateTrainingConfig
    main: MainTrainingConfig

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Experiment name cannot be empty")
        if self.subset_count <= 0:
            raise ValueError("subset_count must be positive")
        if not self.evaluation_benchmarks:
            raise ValueError("At least one evaluation benchmark is required")
        self.candidate.validate()
        self.main.validate()


def approach2_config_from_dict(data: dict[str, Any]) -> Approach2Config:
    data = dict(data)
    data["evaluation_benchmarks"] = tuple(data["evaluation_benchmarks"])
    data["candidate"] = CandidateTrainingConfig(**data["candidate"])
    data["main"] = MainTrainingConfig(**data["main"])
    config = Approach2Config(**data)
    config.validate()
    return config


def load_approach2_config(path: str | os.PathLike[str]) -> Approach2Config:
    return approach2_config_from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def generate_action_subsets(
    action_count: int,
    subset_count: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    if action_count <= 0:
        raise ValueError("action_count must be positive")
    if subset_count <= 0:
        raise ValueError("subset_count must be positive")
    if subset_count > action_count:
        raise ValueError("subset_count cannot exceed action_count")

    actions = list(range(action_count))
    random.Random(seed).shuffle(actions)
    subsets = [[] for _ in range(subset_count)]
    for index, action in enumerate(actions):
        subsets[index % subset_count].append(action)
    return tuple(tuple(subset) for subset in subsets)


class SubsetActionEnv(gym.Wrapper):
    """Expose a local discrete action space mapped to LLVM action IDs."""

    def __init__(self, env: gym.Env, llvm_actions: tuple[int, ...]):
        super().__init__(env)
        if not llvm_actions:
            raise ValueError("An action subset cannot be empty")
        self.llvm_actions = llvm_actions
        self.action_space = spaces.Discrete(len(llvm_actions))
        self.observation_space = env.observation_space

    def seed(self, seed: int | None = None) -> list[int]:
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
        if seed is not None:
            self.action_space.seed(seed)
        return [0 if seed is None else seed]

    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        local_action = int(np.asarray(action).reshape(-1)[0])
        llvm_action = self.llvm_actions[local_action]
        observation, reward, done, info = self.env.step(llvm_action)
        info = dict(info)
        info.update(
            {
                "subset_action": local_action,
                "llvm_action": llvm_action,
            }
        )
        return observation, reward, done, info


class SubsetPolicyAdapter:
    """Convert a child model's local prediction into an LLVM action."""

    def __init__(self, model: Any, llvm_actions: tuple[int, ...]):
        self.model = model
        self.llvm_actions = llvm_actions

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[int, Any]:
        local_action, state = self.model.predict(
            observation,
            deterministic=deterministic,
        )
        local_action = int(np.asarray(local_action).reshape(-1)[0])
        return self.llvm_actions[local_action], state


def _model_config(
    *,
    config: Approach2Config,
    name: str,
    algorithm: str,
    total_timesteps: int,
    max_episode_steps: int,
    layer_nodes: int,
    layer_count: int,
    hyperparameters: Mapping[str, Any],
    seed: int,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        algorithm=algorithm,
        train_benchmark=config.train_benchmark,
        evaluation_benchmarks=config.evaluation_benchmarks,
        observation_space=config.observation_space,
        total_timesteps=total_timesteps,
        evaluation_interval=config.main.evaluation_interval,
        evaluation_episodes=config.main.evaluation_episodes,
        max_episode_steps=max_episode_steps,
        layer_nodes=layer_nodes,
        layer_count=layer_count,
        seed=seed,
        hyperparameters=hyperparameters,
    )


def _actual_timesteps(model: Any) -> int:
    return int(getattr(model, "num_timesteps", 0))


def _metadata(
    config: Approach2Config,
    subsets: tuple[tuple[int, ...], ...],
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "approach": "approach2",
        "config": asdict(config),
        "action_subsets": subsets,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": package_versions(
            [
                "compiler-gym",
                "gym",
                "gymnasium",
                "stable-baselines3",
                "shimmy",
                "torch",
            ]
        ),
    }


def run_approach2(config: Approach2Config) -> None:
    config.validate()
    seed_everything(config.seed)
    output_dir = Path(config.output_dir) / config.name / f"seed_{config.seed}"
    children_dir = output_dir / "children"
    children_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("evaluation_episodes.csv", "evaluation_summary.csv"):
        path = output_dir / filename
        if path.exists():
            path.unlink()

    probe_env = make_compiler_env(
        config.train_benchmark,
        config.observation_space,
        config.candidate.max_episode_steps,
        config.seed,
    )
    try:
        action_count = int(probe_env.action_space.n)
    finally:
        probe_env.close()
    subsets = generate_action_subsets(
        action_count,
        config.subset_count,
        config.subset_seed,
    )
    (output_dir / "action_subsets.json").write_text(
        json.dumps(
            {
                "action_count": action_count,
                "subset_seed": config.subset_seed,
                "subsets": subsets,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(_metadata(config, subsets), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    candidate_models: list[tuple[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    for index, subset in enumerate(subsets):
        name = f"subset_{index}"
        candidate_seed = config.seed + index + 1
        env = SubsetActionEnv(
            make_compiler_env(
                config.train_benchmark,
                config.observation_space,
                config.candidate.max_episode_steps,
                candidate_seed,
            ),
            subset,
        )
        try:
            model_config = _model_config(
                config=config,
                name=name,
                algorithm=config.candidate.algorithm,
                total_timesteps=config.candidate.total_timesteps,
                max_episode_steps=config.candidate.max_episode_steps,
                layer_nodes=config.candidate.layer_nodes,
                layer_count=config.candidate.layer_count,
                hyperparameters=config.candidate.hyperparameters,
                seed=candidate_seed,
            )
            print(
                f"Training {name}: {len(subset)} LLVM actions, "
                f"seed={candidate_seed}"
            )
            model = build_model(model_config, env)
            model.learn(total_timesteps=config.candidate.total_timesteps)
            model_path = children_dir / (
                f"{name}_{_actual_timesteps(model)}_steps"
            )
            model.save(model_path)
            candidate_models.append(
                (name, SubsetPolicyAdapter(model, subset))
            )
            candidate_summaries.append(
                {
                    "name": name,
                    "subset": subset,
                    "requested_timesteps": config.candidate.total_timesteps,
                    "actual_timesteps": _actual_timesteps(model),
                    "model": str(model_path.with_suffix(".zip").resolve()),
                }
            )
        finally:
            env.close()

    main_config = _model_config(
        config=config,
        name=f"{config.name}_main",
        algorithm=config.main.algorithm,
        total_timesteps=config.main.total_timesteps,
        max_episode_steps=config.main.max_episode_steps,
        layer_nodes=config.main.layer_nodes,
        layer_count=config.main.layer_count,
        hyperparameters=config.main.hyperparameters,
        seed=config.seed,
    )
    main_env = CandidateSelectorEnv(
        make_compiler_env(
            config.train_benchmark,
            config.observation_space,
            config.main.max_episode_steps,
            config.seed,
        ),
        candidate_models,
        config.main.deterministic_candidates,
    )
    evaluation_envs = {
        benchmark: CandidateSelectorEnv(
            make_compiler_env(
                benchmark,
                config.observation_space,
                config.main.max_episode_steps,
                config.seed + 100 + benchmark_index,
            ),
            candidate_models,
            config.main.deterministic_candidates,
        )
        for benchmark_index, benchmark in enumerate(config.evaluation_benchmarks)
    }

    try:
        print(f"Training main agent with {len(candidate_models)} subset choices")
        main_model = build_model(main_config, main_env)
        while _actual_timesteps(main_model) < config.main.total_timesteps:
            chunk = min(
                config.main.evaluation_interval,
                config.main.total_timesteps - _actual_timesteps(main_model),
            )
            main_model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=_actual_timesteps(main_model) == 0,
            )
            checkpoint = _actual_timesteps(main_model)
            for benchmark, evaluation_env in evaluation_envs.items():
                from compiler_gym.errors import ServiceError

                records, summary = evaluate_model(
                    main_model,
                    evaluation_env,
                    experiment=config.name,
                    seed=config.seed,
                    benchmark=benchmark,
                    timesteps=checkpoint,
                    episodes=config.main.evaluation_episodes,
                    deterministic=config.main.deterministic_evaluation,
                    service_error_types=(ServiceError,),
                )
                append_dataclasses(output_dir / "evaluation_episodes.csv", records)
                append_dataclasses(output_dir / "evaluation_summary.csv", [summary])

        main_model_path = output_dir / (
            f"{config.name}_main_{_actual_timesteps(main_model)}_steps"
        )
        main_model.save(main_model_path)
        (output_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "action_count": action_count,
                    "subset_seed": config.subset_seed,
                    "candidate_models": candidate_summaries,
                    "main_model": {
                        "requested_timesteps": config.main.total_timesteps,
                        "actual_timesteps": _actual_timesteps(main_model),
                        "model": str(main_model_path.with_suffix(".zip").resolve()),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        main_env.close()
        for evaluation_env in evaluation_envs.values():
            evaluation_env.close()


def load_trained_approach2(
    run_dir: str | os.PathLike[str],
) -> tuple[Approach2Config, list[tuple[str, Any]], Any, int]:
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    config = approach2_config_from_dict(metadata["config"])

    candidates: list[tuple[str, Any]] = []
    for saved_candidate in summary["candidate_models"]:
        model_path = Path(saved_candidate["model"])
        if not model_path.is_absolute():
            model_path = run_dir / "children" / model_path.name
        model = load_model(model_path, config.candidate.algorithm)
        subset = tuple(saved_candidate["subset"])
        candidates.append(
            (saved_candidate["name"], SubsetPolicyAdapter(model, subset))
        )

    main_model_path = Path(summary["main_model"]["model"])
    if not main_model_path.is_absolute():
        main_model_path = run_dir / main_model_path.name
    main_model = load_model(main_model_path, config.main.algorithm)
    return (
        config,
        candidates,
        main_model,
        int(summary["main_model"]["actual_timesteps"]),
    )

