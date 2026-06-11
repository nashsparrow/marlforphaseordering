from __future__ import annotations

import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import gym
import numpy as np
from gym import spaces

from experiment.pipeline import (
    ExperimentConfig,
    append_dataclasses,
    build_model,
    evaluate_model,
    make_compiler_env,
    package_versions,
    seed_everything,
    load_model,
)


SUPPORTED_OBJECTIVES = {"codesize", "runtime"}


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    objective: str
    max_episode_steps: int
    total_timesteps: int
    algorithm: str = "PPO"
    layer_nodes: int = 128
    layer_count: int = 4
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Candidate name cannot be empty")
        if self.objective not in SUPPORTED_OBJECTIVES:
            raise ValueError(
                f"Unsupported candidate objective {self.objective!r}; "
                f"expected one of {sorted(SUPPORTED_OBJECTIVES)}"
            )
        if self.max_episode_steps <= 0:
            raise ValueError("Candidate max_episode_steps must be positive")
        if self.total_timesteps <= 0:
            raise ValueError("Candidate total_timesteps must be positive")


@dataclass(frozen=True)
class MainAgentConfig:
    objective: str = "codesize"
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
        if self.objective not in SUPPORTED_OBJECTIVES:
            raise ValueError(
                f"Unsupported main objective {self.objective!r}; "
                f"expected one of {sorted(SUPPORTED_OBJECTIVES)}"
            )
        for name in (
            "total_timesteps",
            "evaluation_interval",
            "evaluation_episodes",
            "max_episode_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"Main agent {name} must be positive")


@dataclass(frozen=True)
class Approach1Config:
    name: str
    train_benchmark: str
    evaluation_benchmarks: tuple[str, ...]
    observation_space: str
    seed: int
    output_dir: str
    runtime_reward: str
    candidates: tuple[CandidateConfig, ...]
    main: MainAgentConfig

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Experiment name cannot be empty")
        if not self.evaluation_benchmarks:
            raise ValueError("At least one evaluation benchmark is required")
        if not self.candidates:
            raise ValueError("Approach 1 requires at least one candidate")
        if self.runtime_reward not in {"negative", "delta"}:
            raise ValueError("runtime_reward must be 'negative' or 'delta'")
        names = [candidate.name for candidate in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("Candidate names must be unique")
        for candidate in self.candidates:
            candidate.validate()
        self.main.validate()


def approach1_config_from_dict(data: dict[str, Any]) -> Approach1Config:
    data = dict(data)
    data["evaluation_benchmarks"] = tuple(data["evaluation_benchmarks"])
    data["candidates"] = tuple(
        CandidateConfig(**candidate) for candidate in data["candidates"]
    )
    data["main"] = MainAgentConfig(**data["main"])
    config = Approach1Config(**data)
    config.validate()
    return config


def load_approach1_config(path: str | os.PathLike[str]) -> Approach1Config:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return approach1_config_from_dict(data)


def _observation_value(env: Any, name: str) -> Any:
    current = env
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        observation = getattr(current, "observation", None)
        if observation is not None:
            return observation[name]
        if not hasattr(current, "env"):
            break
        current = current.env
    raise AttributeError(f"Environment does not expose observation {name!r}")


def _runtime(env: Any) -> float:
    value = np.asarray(_observation_value(env, "Runtime"), dtype=float).reshape(-1)
    return float(value[0]) if value.size else math.nan


class ObjectiveRewardWrapper(gym.Wrapper):
    """Replace CompilerGym's reward when runtime is the selected objective."""

    def __init__(self, env: gym.Env, objective: str, runtime_reward: str):
        super().__init__(env)
        self.objective = objective
        self.runtime_reward = runtime_reward
        self.previous_runtime = math.nan

    def reset(self, **kwargs: Any) -> Any:
        observation = self.env.reset(**kwargs)
        if self.objective == "runtime":
            self.previous_runtime = _runtime(self.env)
        return observation

    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        observation, reward, done, info = self.env.step(action)
        if self.objective == "runtime":
            current_runtime = _runtime(self.env)
            if self.runtime_reward == "delta":
                reward = self.previous_runtime - current_runtime
            else:
                reward = -current_runtime
            self.previous_runtime = current_runtime
        return observation, float(reward), done, info


class CandidateSelectorEnv(gym.Wrapper):
    """Main-agent environment whose actions select candidate predictions."""

    def __init__(
        self,
        env: gym.Env,
        candidates: list[tuple[str, Any]],
        deterministic_candidates: bool,
    ):
        super().__init__(env)
        if not candidates:
            raise ValueError("At least one candidate model is required")
        self.candidates = candidates
        self.deterministic_candidates = deterministic_candidates
        self.action_space = spaces.Discrete(len(candidates))
        self.observation_space = env.observation_space
        self.current_observation: Any | None = None

    def seed(self, seed: int | None = None) -> list[int]:
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
        if seed is not None:
            self.action_space.seed(seed)
        return [0 if seed is None else seed]

    def reset(self, **kwargs: Any) -> Any:
        self.current_observation = self.env.reset(**kwargs)
        return self.current_observation

    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        if self.current_observation is None:
            raise RuntimeError("reset() must be called before step()")
        candidate_index = int(np.asarray(action).reshape(-1)[0])
        candidate_name, candidate_model = self.candidates[candidate_index]
        llvm_action, _ = candidate_model.predict(
            self.current_observation,
            deterministic=self.deterministic_candidates,
        )
        llvm_action = int(np.asarray(llvm_action).reshape(-1)[0])
        observation, reward, done, info = self.env.step(llvm_action)
        self.current_observation = observation
        info = dict(info)
        info.update(
            {
                "candidate_index": candidate_index,
                "candidate_name": candidate_name,
                "llvm_action": llvm_action,
            }
        )
        return observation, reward, done, info


def _model_config(
    *,
    approach: Approach1Config,
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
        train_benchmark=approach.train_benchmark,
        evaluation_benchmarks=approach.evaluation_benchmarks,
        observation_space=approach.observation_space,
        total_timesteps=total_timesteps,
        evaluation_interval=approach.main.evaluation_interval,
        evaluation_episodes=approach.main.evaluation_episodes,
        max_episode_steps=max_episode_steps,
        layer_nodes=layer_nodes,
        layer_count=layer_count,
        seed=seed,
        hyperparameters=hyperparameters,
    )


def make_objective_env(
    *,
    benchmark: str,
    observation_space: str,
    max_episode_steps: int,
    seed: int,
    objective: str,
    runtime_reward: str,
) -> gym.Env:
    env = make_compiler_env(
        benchmark,
        observation_space,
        max_episode_steps,
        seed,
    )
    return ObjectiveRewardWrapper(env, objective, runtime_reward)


def _actual_timesteps(model: Any) -> int:
    return int(getattr(model, "num_timesteps", 0))


def _metadata(config: Approach1Config) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "approach": "approach1",
        "config": asdict(config),
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


def run_approach1(config: Approach1Config) -> None:
    config.validate()
    seed_everything(config.seed)
    output_dir = Path(config.output_dir) / config.name / f"seed_{config.seed}"
    children_dir = output_dir / "children"
    children_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("evaluation_episodes.csv", "evaluation_summary.csv"):
        path = output_dir / filename
        if path.exists():
            path.unlink()
    (output_dir / "metadata.json").write_text(
        json.dumps(_metadata(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    candidate_models: list[tuple[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []

    for index, candidate in enumerate(config.candidates):
        candidate_seed = config.seed + index + 1
        env = make_objective_env(
            benchmark=config.train_benchmark,
            observation_space=config.observation_space,
            max_episode_steps=candidate.max_episode_steps,
            seed=candidate_seed,
            objective=candidate.objective,
            runtime_reward=config.runtime_reward,
        )
        try:
            model_config = _model_config(
                approach=config,
                name=candidate.name,
                algorithm=candidate.algorithm,
                total_timesteps=candidate.total_timesteps,
                max_episode_steps=candidate.max_episode_steps,
                layer_nodes=candidate.layer_nodes,
                layer_count=candidate.layer_count,
                hyperparameters=candidate.hyperparameters,
                seed=candidate_seed,
            )
            print(
                f"Training candidate {candidate.name}: "
                f"objective={candidate.objective}, "
                f"max_steps={candidate.max_episode_steps}"
            )
            model = build_model(model_config, env)
            model.learn(total_timesteps=candidate.total_timesteps)
            model_path = children_dir / (
                f"{candidate.name}_{_actual_timesteps(model)}_steps"
            )
            model.save(model_path)
            candidate_models.append((candidate.name, model))
            candidate_summaries.append(
                {
                    "name": candidate.name,
                    "objective": candidate.objective,
                    "max_episode_steps": candidate.max_episode_steps,
                    "requested_timesteps": candidate.total_timesteps,
                    "actual_timesteps": _actual_timesteps(model),
                    "model": str(model_path.with_suffix(".zip").resolve()),
                }
            )
        finally:
            env.close()

    main_config = _model_config(
        approach=config,
        name=f"{config.name}_main",
        algorithm=config.main.algorithm,
        total_timesteps=config.main.total_timesteps,
        max_episode_steps=config.main.max_episode_steps,
        layer_nodes=config.main.layer_nodes,
        layer_count=config.main.layer_count,
        hyperparameters=config.main.hyperparameters,
        seed=config.seed,
    )
    main_base_env = make_objective_env(
        benchmark=config.train_benchmark,
        observation_space=config.observation_space,
        max_episode_steps=config.main.max_episode_steps,
        seed=config.seed,
        objective=config.main.objective,
        runtime_reward=config.runtime_reward,
    )
    main_env = CandidateSelectorEnv(
        main_base_env,
        candidate_models,
        config.main.deterministic_candidates,
    )
    evaluation_envs = {
        benchmark: CandidateSelectorEnv(
            make_objective_env(
                benchmark=benchmark,
                observation_space=config.observation_space,
                max_episode_steps=config.main.max_episode_steps,
                seed=config.seed + 100 + benchmark_index,
                objective=config.main.objective,
                runtime_reward=config.runtime_reward,
            ),
            candidate_models,
            config.main.deterministic_candidates,
        )
        for benchmark_index, benchmark in enumerate(config.evaluation_benchmarks)
    }

    try:
        print(
            f"Training main agent with {len(candidate_models)} candidate choices"
        )
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
                    "candidate_models": candidate_summaries,
                    "main_model": {
                        "objective": config.main.objective,
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


def load_trained_approach1(
    run_dir: str | os.PathLike[str],
) -> tuple[Approach1Config, list[tuple[str, Any]], Any, int]:
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    config = approach1_config_from_dict(metadata["config"])
    candidate_by_name = {candidate.name: candidate for candidate in config.candidates}

    candidates: list[tuple[str, Any]] = []
    for saved_candidate in summary["candidate_models"]:
        name = saved_candidate["name"]
        candidate_config = candidate_by_name[name]
        model_path = Path(saved_candidate["model"])
        if not model_path.is_absolute():
            model_path = run_dir / "children" / model_path.name
        candidates.append(
            (name, load_model(model_path, candidate_config.algorithm))
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
