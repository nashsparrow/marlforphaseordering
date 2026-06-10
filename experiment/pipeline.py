from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "ppo_bzip2"
    algorithm: str = "PPO"
    train_benchmark: str = "cbench-v1/bzip2"
    evaluation_benchmarks: tuple[str, ...] = ("cbench-v1/bzip2",)
    observation_space: str = "Autophase"
    total_timesteps: int = 20_000
    evaluation_interval: int = 1_000
    evaluation_episodes: int = 100
    max_episode_steps: int = 40
    layer_nodes: int = 128
    layer_count: int = 4
    seed: int = 0
    deterministic_evaluation: bool = True
    save_checkpoints: bool = False
    output_dir: str = "results"
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.algorithm.upper() not in {"PPO", "A2C", "DQN"}:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        positive_values = {
            "total_timesteps": self.total_timesteps,
            "evaluation_interval": self.evaluation_interval,
            "evaluation_episodes": self.evaluation_episodes,
            "max_episode_steps": self.max_episode_steps,
            "layer_count": self.layer_count,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.layer_nodes < 0:
            raise ValueError("layer_nodes cannot be negative")
        if not self.evaluation_benchmarks:
            raise ValueError("At least one evaluation benchmark is required")


@dataclass(frozen=True)
class EvaluationRecord:
    experiment: str
    seed: int
    benchmark: str
    timesteps: int
    episode: int
    reward: float
    instruction_count: float
    instruction_reduction_ratio: float
    runtime: float
    episode_steps: int


@dataclass(frozen=True)
class EvaluationSummary:
    experiment: str
    seed: int
    benchmark: str
    timesteps: int
    training_episodes: int
    requested_episodes: int
    completed_episodes: int
    failed_episodes: int
    reward_mean: float
    reward_median: float
    reward_std: float
    reward_ci95: float
    reward_min: float
    reward_max: float
    instruction_count_mean: float
    instruction_count_median: float
    instruction_count_std: float
    instruction_count_ci95: float
    instruction_count_min: float
    instruction_count_max: float
    runtime_mean: float
    runtime_median: float
    runtime_std: float
    runtime_ci95: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _seed_env(env: Any, seed: int) -> None:
    if hasattr(env, "seed"):
        env.seed(seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


def _reset_env(env: Any, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
    if seed is not None:
        _seed_env(env, seed)
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, {}


def _step_env(env: Any, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated), bool(truncated), info
    observation, reward, done, info = result
    truncated = bool(info.get("TimeLimit.truncated", False))
    terminated = bool(done and not truncated)
    return observation, float(reward), terminated, truncated, info


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


def _numeric_scalar(value: Any) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        return math.nan
    return float(array[0])


def _summary(values: Sequence[float]) -> tuple[float, float, float, float, float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return (math.nan,) * 6
    mean = statistics.fmean(finite)
    median = statistics.median(finite)
    standard_deviation = statistics.stdev(finite) if len(finite) > 1 else 0.0
    ci95 = 1.96 * standard_deviation / math.sqrt(len(finite))
    return mean, median, standard_deviation, ci95, min(finite), max(finite)


def summarize_records(
    records: Sequence[EvaluationRecord],
    *,
    experiment: str,
    seed: int,
    benchmark: str,
    timesteps: int,
    training_episodes: int = 0,
    requested_episodes: int,
    failed_episodes: int,
) -> EvaluationSummary:
    reward = _summary([record.reward for record in records])
    instruction_count = _summary([record.instruction_count for record in records])
    runtime = _summary([record.runtime for record in records])
    return EvaluationSummary(
        experiment=experiment,
        seed=seed,
        benchmark=benchmark,
        timesteps=timesteps,
        training_episodes=training_episodes,
        requested_episodes=requested_episodes,
        completed_episodes=len(records),
        failed_episodes=failed_episodes,
        reward_mean=reward[0],
        reward_median=reward[1],
        reward_std=reward[2],
        reward_ci95=reward[3],
        reward_min=reward[4],
        reward_max=reward[5],
        instruction_count_mean=instruction_count[0],
        instruction_count_median=instruction_count[1],
        instruction_count_std=instruction_count[2],
        instruction_count_ci95=instruction_count[3],
        instruction_count_min=instruction_count[4],
        instruction_count_max=instruction_count[5],
        runtime_mean=runtime[0],
        runtime_median=runtime[1],
        runtime_std=runtime[2],
        runtime_ci95=runtime[3],
    )


def evaluate_model(
    model: Any,
    env: Any,
    *,
    experiment: str,
    seed: int,
    benchmark: str,
    timesteps: int,
    training_episodes: int = 0,
    episodes: int,
    deterministic: bool = True,
    service_error_types: tuple[type[BaseException], ...] = (),
) -> tuple[list[EvaluationRecord], EvaluationSummary]:
    records: list[EvaluationRecord] = []
    failures = 0

    for episode in range(episodes):
        try:
            observation, _ = _reset_env(env, seed + episode)
            original_instruction_count = _numeric_scalar(
                _observation_value(env, "IrInstructionCount")
            )
            done = False
            score = 0.0
            episode_steps = 0
            while not done:
                action, _ = model.predict(
                    observation, deterministic=deterministic
                )
                observation, reward, terminated, truncated, _ = _step_env(env, action)
                score += reward
                episode_steps += 1
                done = terminated or truncated

            instruction_count = _numeric_scalar(
                _observation_value(env, "IrInstructionCount")
            )
            runtime = _numeric_scalar(_observation_value(env, "Runtime"))
            ratio = (
                original_instruction_count / instruction_count
                if instruction_count
                else math.nan
            )
            records.append(
                EvaluationRecord(
                    experiment=experiment,
                    seed=seed,
                    benchmark=benchmark,
                    timesteps=timesteps,
                    episode=episode,
                    reward=score,
                    instruction_count=instruction_count,
                    instruction_reduction_ratio=ratio,
                    runtime=runtime,
                    episode_steps=episode_steps,
                )
            )
        except service_error_types:
            failures += 1

    summary = summarize_records(
        records,
        experiment=experiment,
        seed=seed,
        benchmark=benchmark,
        timesteps=timesteps,
        training_episodes=training_episodes,
        requested_episodes=episodes,
        failed_episodes=failures,
    )
    return records, summary


def append_dataclasses(path: Path, rows: Iterable[Any]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def package_versions(packages: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def collect_metadata(config: ExperimentConfig) -> dict[str, Any]:
    try:
        llvm_version = subprocess.run(
            ["llvm-config", "--version"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError:
        llvm_version = "unavailable"
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "llvm": llvm_version or "unavailable",
        "packages": package_versions(
            ["compiler-gym", "gym", "gymnasium", "stable-baselines3", "shimmy", "torch"]
        ),
    }


def save_metadata(path: Path, config: ExperimentConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(collect_metadata(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_run_summary(path: Path, *, requested_timesteps: int, actual_timesteps: int) -> None:
    path.write_text(
        json.dumps(
            {
                "requested_timesteps": requested_timesteps,
                "actual_timesteps": actual_timesteps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_config(path: str | os.PathLike[str]) -> ExperimentConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "evaluation_benchmarks" in data:
        data["evaluation_benchmarks"] = tuple(data["evaluation_benchmarks"])
    config = ExperimentConfig(**data)
    config.validate()
    return config


def make_compiler_env(
    benchmark: str,
    observation_space: str,
    max_episode_steps: int,
    seed: int,
) -> Any:
    import compiler_gym  # noqa: F401
    import gym
    from compiler_gym.envs import CompilerEnv

    class CompilerGymTimeLimit(gym.Wrapper):
        def __init__(self, env: CompilerEnv):
            super().__init__(env)
            self.elapsed_steps = 0
            self._seed = seed

        def seed(self, value: int | None = None) -> list[int]:
            if value is not None:
                self._seed = value
                if hasattr(self.action_space, "seed"):
                    self.action_space.seed(value)
                if hasattr(self.observation_space, "seed"):
                    self.observation_space.seed(value)
            return [self._seed]

        def reset(self, **kwargs: Any) -> Any:
            self.elapsed_steps = 0
            return self.env.reset(**kwargs)

        def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
            observation, reward, done, info = self.env.step(int(action))
            self.elapsed_steps += 1
            if self.elapsed_steps >= max_episode_steps and not done:
                done = True
                info = dict(info)
                info["TimeLimit.truncated"] = True
            return observation, reward, done, info

    env = gym.make(
        "llvm-autophase-ic-v0",
        benchmark=benchmark,
        observation_space=observation_space,
    )
    wrapped = CompilerGymTimeLimit(env)
    _seed_env(wrapped, seed)
    return wrapped


def build_model(config: ExperimentConfig, env: Any) -> Any:
    from stable_baselines3 import A2C, DQN, PPO

    algorithms = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
    algorithm = config.algorithm.upper()
    layers = [config.layer_nodes] * config.layer_count
    kwargs = dict(config.hyperparameters)
    if config.layer_nodes:
        if algorithm == "DQN":
            kwargs.setdefault("policy_kwargs", {"net_arch": layers})
        else:
            kwargs.setdefault(
                "policy_kwargs",
                {"net_arch": {"pi": layers, "vf": layers}},
            )
    return algorithms[algorithm](
        "MlpPolicy",
        env,
        seed=config.seed,
        verbose=1,
        **kwargs,
    )


def load_model(
    path: str | os.PathLike[str],
    algorithm: str,
    env: Any | None = None,
) -> Any:
    from stable_baselines3 import A2C, DQN, PPO

    algorithms = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
    normalized_algorithm = algorithm.upper()
    if normalized_algorithm not in algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return algorithms[normalized_algorithm].load(path, env=env)


def run_experiment(
    config: ExperimentConfig,
    env_factory: Callable[[str, str, int, int], Any] = make_compiler_env,
) -> None:
    config.validate()
    seed_everything(config.seed)
    output_dir = Path(config.output_dir) / config.name / f"seed_{config.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for generated_csv in ("evaluation_episodes.csv", "evaluation_summary.csv"):
        path = output_dir / generated_csv
        if path.exists():
            path.unlink()
    for generated_model in output_dir.glob(f"{config.name}_*_steps.zip"):
        generated_model.unlink()
    save_metadata(output_dir / "metadata.json", config)

    training_env = env_factory(
        config.train_benchmark,
        config.observation_space,
        config.max_episode_steps,
        config.seed,
    )
    evaluation_envs = {
        benchmark: env_factory(
            benchmark,
            config.observation_space,
            config.max_episode_steps,
            config.seed + index + 1,
        )
        for index, benchmark in enumerate(config.evaluation_benchmarks)
    }

    try:
        from stable_baselines3.common.callbacks import BaseCallback

        class EpisodeCounterCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__()
                self.completed_episodes = 0

            def _on_step(self) -> bool:
                self.completed_episodes += sum(
                    bool(done) for done in self.locals.get("dones", ())
                )
                return True

        model = build_model(config, training_env)
        episode_counter = EpisodeCounterCallback()
        while model.num_timesteps < config.total_timesteps:
            chunk = min(
                config.evaluation_interval,
                config.total_timesteps - model.num_timesteps,
            )
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=model.num_timesteps == 0,
                callback=episode_counter,
            )
            actual_timesteps = model.num_timesteps

            for benchmark, evaluation_env in evaluation_envs.items():
                from compiler_gym.errors import ServiceError

                records, summary = evaluate_model(
                    model,
                    evaluation_env,
                    experiment=config.name,
                    seed=config.seed,
                    benchmark=benchmark,
                    timesteps=actual_timesteps,
                    training_episodes=episode_counter.completed_episodes,
                    episodes=config.evaluation_episodes,
                    deterministic=config.deterministic_evaluation,
                    service_error_types=(ServiceError,),
                )
                append_dataclasses(output_dir / "evaluation_episodes.csv", records)
                append_dataclasses(output_dir / "evaluation_summary.csv", [summary])

            if config.save_checkpoints:
                model.save(output_dir / f"{config.name}_{actual_timesteps}_steps")

        model.save(output_dir / f"{config.name}_{model.num_timesteps}_steps")
        save_run_summary(
            output_dir / "run_summary.json",
            requested_timesteps=config.total_timesteps,
            actual_timesteps=model.num_timesteps,
        )
    finally:
        training_env.close()
        for evaluation_env in evaluation_envs.values():
            evaluation_env.close()
