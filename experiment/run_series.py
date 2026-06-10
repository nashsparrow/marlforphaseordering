from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from experiment.pipeline import load_config, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experiment configurations sequentially."
    )
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/series_summary.json"),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def write_summary(path: Path, runs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "runs": runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    runs: list[dict] = []

    for config_path in args.config:
        base_config = load_config(config_path)
        for seed in args.seed or [base_config.seed]:
            config = replace(base_config, seed=seed)
            result = {
                "config_file": str(config_path),
                "config": asdict(config),
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            try:
                print(f"Running {config.name} with seed {seed}")
                run_experiment(config)
                result["status"] = "completed"
            except Exception as error:
                result["status"] = "failed"
                result["error"] = f"{type(error).__name__}: {error}"
                result["traceback"] = traceback.format_exc()
                runs.append(result)
                write_summary(args.summary, runs)
                if not args.continue_on_error:
                    raise
            else:
                runs.append(result)
                write_summary(args.summary, runs)

    completed = sum(run["status"] == "completed" for run in runs)
    failed = sum(run["status"] == "failed" for run in runs)
    print(f"Series finished: {completed} completed, {failed} failed")
    print(f"Summary written to {args.summary}")


if __name__ == "__main__":
    main()
