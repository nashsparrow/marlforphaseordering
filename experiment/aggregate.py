from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate final-checkpoint evaluation summaries across seeds."
    )
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("aggregate_seeds.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for path in args.result_directory.glob("seed_*/evaluation_summary.csv"):
        with path.open(newline="", encoding="utf-8") as input_file:
            rows = list(csv.DictReader(input_file))
        latest: dict[tuple[str, str], dict[str, str]] = {}
        for row in rows:
            key = (row["experiment"], row["benchmark"])
            if key not in latest or int(row["timesteps"]) > int(latest[key]["timesteps"]):
                latest[key] = row
        for key, row in latest.items():
            grouped[key].append(row)

    output_rows = []
    for (experiment, benchmark), rows in sorted(grouped.items()):
        for metric in ("reward_mean", "instruction_count_mean", "runtime_median"):
            values = [float(row[metric]) for row in rows]
            mean = statistics.fmean(values)
            standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
            output_rows.append(
                {
                    "experiment": experiment,
                    "benchmark": benchmark,
                    "metric": metric,
                    "seeds": len(values),
                    "mean_across_seeds": mean,
                    "std_across_seeds": standard_deviation,
                    "ci95_across_seeds": (
                        1.96 * standard_deviation / math.sqrt(len(values))
                    ),
                    "min_across_seeds": min(values),
                    "max_across_seeds": max(values),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "benchmark",
        "metric",
        "seeds",
        "mean_across_seeds",
        "std_across_seeds",
        "ci95_across_seeds",
        "min_across_seeds",
        "max_across_seeds",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
