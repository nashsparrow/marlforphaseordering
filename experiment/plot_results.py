from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MEASUREMENTS = {
    "codesize": {
        "column": "instruction_count_mean",
        "ci_column": "instruction_count_ci95",
        "ylabel": "Mean Instruction Count",
        "title": "Code Size",
    },
    "runtime": {
        "column": "runtime_mean",
        "ci_column": "runtime_ci95",
        "ylabel": "Mean Runtime (seconds)",
        "title": "Runtime",
    },
}
MEASUREMENT_CHOICES = ("codesize", "runtime", "combined")


@dataclass(frozen=True)
class PlotPoint:
    timesteps: int
    value: float
    ci95: float
    samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot runtime and/or code size against training timesteps."
    )
    parser.add_argument(
        "--result",
        action="append",
        type=Path,
        required=True,
        dest="results",
        help=(
            "Result folder or evaluation_summary.csv. Repeat for every "
            "experiment to compare."
        ),
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help="Optional display label for each --result, in the same order.",
    )
    parser.add_argument(
        "--measurement",
        action="append",
        choices=MEASUREMENT_CHOICES,
        dest="measurements",
        help=(
            "Select runtime, codesize, or combined. Combined plots instruction "
            "count and scaled runtime on the same Y-axis."
        ),
    )
    parser.add_argument(
        "--runtime-scale",
        type=float,
        default=1_000_000.0,
        help="Multiplier applied to runtime in combined charts.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        dest="benchmarks",
        help="Only plot selected benchmarks. Repeat for multiple benchmarks.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("charts"))
    parser.add_argument("--title-prefix", default="")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--no-ci",
        action="store_true",
        help="Do not draw 95 percent confidence bands.",
    )
    return parser.parse_args()


def find_summary_files(result: Path) -> list[Path]:
    if result.is_file():
        if result.name != "evaluation_summary.csv":
            raise ValueError(f"Expected evaluation_summary.csv, got: {result}")
        return [result]
    if not result.is_dir():
        raise FileNotFoundError(f"Result path not found: {result}")

    direct = result / "evaluation_summary.csv"
    if direct.exists():
        return [direct]

    seed_files = sorted(result.glob("seed_*/evaluation_summary.csv"))
    if seed_files:
        return seed_files

    recursive = sorted(result.rglob("evaluation_summary.csv"))
    if recursive:
        return recursive
    raise FileNotFoundError(f"No evaluation_summary.csv found under: {result}")


def load_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as input_file:
            rows.extend(csv.DictReader(input_file))
    return rows


def aggregate_points(
    rows: Iterable[dict[str, str]],
    measurement: str,
) -> dict[str, list[PlotPoint]]:
    settings = MEASUREMENTS[measurement]
    grouped: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)

    for row in rows:
        benchmark = row["benchmark"]
        timesteps = int(row["timesteps"])
        value = float(row[settings["column"]])
        ci95 = float(row[settings["ci_column"]])
        if math.isfinite(value):
            grouped[(benchmark, timesteps)].append((value, ci95))

    result: dict[str, list[PlotPoint]] = defaultdict(list)
    for (benchmark, timesteps), values_and_ci in grouped.items():
        values = [item[0] for item in values_and_ci]
        mean = statistics.fmean(values)
        if len(values) > 1:
            standard_deviation = statistics.stdev(values)
            ci95 = 1.96 * standard_deviation / math.sqrt(len(values))
        else:
            ci95 = values_and_ci[0][1]
        result[benchmark].append(
            PlotPoint(
                timesteps=timesteps,
                value=mean,
                ci95=ci95,
                samples=len(values),
            )
        )

    for points in result.values():
        points.sort(key=lambda point: point.timesteps)
    return dict(result)


def default_label(result: Path) -> str:
    if result.is_file():
        return result.parent.name
    return result.name


def safe_filename(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    ).strip("-")


def plot_measurement(
    *,
    series: list[tuple[str, dict[str, list[PlotPoint]]]],
    measurement: str,
    benchmarks: Iterable[str],
    output_dir: Path,
    title_prefix: str,
    file_format: str,
    dpi: int,
    show_ci: bool,
) -> list[Path]:
    settings = MEASUREMENTS[measurement]
    output_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for benchmark in benchmarks:
        figure, axis = plt.subplots(figsize=(10, 6))
        plotted = False
        for label, benchmark_data in series:
            points = benchmark_data.get(benchmark, [])
            if not points:
                continue
            timesteps = [point.timesteps for point in points]
            values = [point.value for point in points]
            line = axis.plot(
                timesteps,
                values,
                marker="o",
                linewidth=2,
                label=label,
            )[0]
            if show_ci:
                lower = [
                    point.value - point.ci95
                    for point in points
                ]
                upper = [
                    point.value + point.ci95
                    for point in points
                ]
                axis.fill_between(
                    timesteps,
                    lower,
                    upper,
                    color=line.get_color(),
                    alpha=0.18,
                )
            plotted = True

        if not plotted:
            plt.close(figure)
            continue

        title_parts = [
            part
            for part in (title_prefix, settings["title"], benchmark)
            if part
        ]
        axis.set_title(" - ".join(title_parts))
        axis.set_xlabel("Training Timesteps")
        axis.set_ylabel(settings["ylabel"])
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend()
        figure.tight_layout()

        filename = (
            f"{safe_filename(benchmark)}_{measurement}_vs_timesteps."
            f"{file_format}"
        )
        output_path = output_dir / filename
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)

    return output_paths


def plot_combined(
    *,
    series: list[
        tuple[
            str,
            dict[str, list[PlotPoint]],
            dict[str, list[PlotPoint]],
        ]
    ],
    benchmarks: Iterable[str],
    output_dir: Path,
    title_prefix: str,
    file_format: str,
    dpi: int,
    show_ci: bool,
    runtime_scale: float,
) -> list[Path]:
    output_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for benchmark in benchmarks:
        figure, axis = plt.subplots(figsize=(10, 6))
        plotted = False
        for label, codesize_data, runtime_data in series:
            codesize_points = codesize_data.get(benchmark, [])
            runtime_points = runtime_data.get(benchmark, [])

            if codesize_points:
                timesteps = [point.timesteps for point in codesize_points]
                values = [point.value for point in codesize_points]
                line = axis.plot(
                    timesteps,
                    values,
                    marker="o",
                    linewidth=2,
                    linestyle="-",
                    label=f"{label} Code Size",
                )[0]
                if show_ci:
                    axis.fill_between(
                        timesteps,
                        [point.value - point.ci95 for point in codesize_points],
                        [point.value + point.ci95 for point in codesize_points],
                        color=line.get_color(),
                        alpha=0.14,
                    )
                plotted = True

            if runtime_points:
                timesteps = [point.timesteps for point in runtime_points]
                values = [point.value * runtime_scale for point in runtime_points]
                line = axis.plot(
                    timesteps,
                    values,
                    marker="s",
                    linewidth=2,
                    linestyle="--",
                    label=f"{label} Runtime x {runtime_scale:g}",
                )[0]
                if show_ci:
                    axis.fill_between(
                        timesteps,
                        [
                            (point.value - point.ci95) * runtime_scale
                            for point in runtime_points
                        ],
                        [
                            (point.value + point.ci95) * runtime_scale
                            for point in runtime_points
                        ],
                        color=line.get_color(),
                        alpha=0.14,
                    )
                plotted = True

        if not plotted:
            plt.close(figure)
            continue

        title_parts = [
            part for part in (title_prefix, "Code Size and Scaled Runtime", benchmark)
            if part
        ]
        axis.set_title(" - ".join(title_parts))
        axis.set_xlabel("Training Timesteps")
        axis.set_ylabel(
            f"Instruction Count / Runtime x {runtime_scale:g}"
        )
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend()
        figure.tight_layout()

        filename = f"{safe_filename(benchmark)}_combined_vs_timesteps.{file_format}"
        output_path = output_dir / filename
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.results):
        raise ValueError("Provide exactly one --label for every --result")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    if args.runtime_scale <= 0:
        raise ValueError("runtime-scale must be positive")

    labels = args.labels or [default_label(path) for path in args.results]
    measurements = tuple(dict.fromkeys(args.measurements or ("codesize", "runtime")))
    all_rows = [
        load_rows(find_summary_files(result))
        for result in args.results
    ]
    available_benchmarks = sorted(
        {
            row["benchmark"]
            for rows in all_rows
            for row in rows
        }
    )
    benchmarks = args.benchmarks or available_benchmarks

    generated: list[Path] = []
    for measurement in measurements:
        if measurement == "combined":
            combined_series = [
                (
                    label,
                    aggregate_points(rows, "codesize"),
                    aggregate_points(rows, "runtime"),
                )
                for label, rows in zip(labels, all_rows)
            ]
            generated.extend(
                plot_combined(
                    series=combined_series,
                    benchmarks=benchmarks,
                    output_dir=args.output_dir,
                    title_prefix=args.title_prefix,
                    file_format=args.format,
                    dpi=args.dpi,
                    show_ci=not args.no_ci,
                    runtime_scale=args.runtime_scale,
                )
            )
            continue

        series = [
            (label, aggregate_points(rows, measurement))
            for label, rows in zip(labels, all_rows)
        ]
        generated.extend(
            plot_measurement(
                series=series,
                measurement=measurement,
                benchmarks=benchmarks,
                output_dir=args.output_dir,
                title_prefix=args.title_prefix,
                file_format=args.format,
                dpi=args.dpi,
                show_ci=not args.no_ci,
            )
        )

    if not generated:
        raise ValueError("No charts were generated for the selected benchmarks")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
