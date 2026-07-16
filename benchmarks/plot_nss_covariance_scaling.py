"""Plot nested-sampler covariance-factor benchmark data."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "nss_covariance_scaling.csv"
OUTPUT_PATH = HERE / "nss_covariance_scaling.png"

BLUE = "#4472C4"
ORANGE = "#ED7D31"
GRID = "#D9D9D9"


def format_runtime(milliseconds: float) -> str:
    """Choose readable units for a runtime annotation."""
    if milliseconds >= 1000:
        return f"{milliseconds / 1000:.1f} s"  # noqa: E231
    return f"{milliseconds:.2f} ms"  # noqa: E231


def load_data(
    data_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load benchmark columns from the checked-in CSV file."""
    with data_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"benchmark data is empty: {data_path}")

    def values(name: str, dtype: type = float) -> np.ndarray:
        return np.asarray([dtype(row[name]) for row in rows])

    return (
        {
            "dimension": values("dimension", int),
            "num_live": values("num_live", int),
            "num_inner_steps": values("num_inner_steps", int),
            "repeats": values("repeats", int),
            "old_ms": values("old_ms"),
            "old_mad_ms": values("old_mad_ms"),
            "factored_ms": values("factored_ms"),
            "factored_mad_ms": values("factored_mad_ms"),
            "speedup": values("speedup"),
            "matched_paths": np.asarray(
                [row["matched_paths"].lower() == "true" for row in rows]
            ),
        },
        {
            "device": rows[0].get("device", "unknown device"),
            "jax_version": rows[0].get("jax_version", "unknown"),
            "dtype": rows[0].get("dtype", "unknown"),
        },
    )


def plot(data_path: Path = DATA_PATH, output_path: Path = OUTPUT_PATH) -> Path:
    """Generate the two-panel runtime and speedup figure."""
    data, metadata = load_data(data_path)
    dimension = data["dimension"]
    repeats = data["repeats"]
    old_ms = data["old_ms"]
    old_mad_ms = data["old_mad_ms"]
    factored_ms = data["factored_ms"]
    factored_mad_ms = data["factored_mad_ms"]
    speedup = data["speedup"]
    matched = data["matched_paths"]
    diverged = ~matched

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )
    figure, (runtime_axis, speedup_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 2)},
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")
    figure.suptitle("Nested sampler covariance factorization benchmark", fontsize=15)

    old_line = runtime_axis.plot(
        dimension,
        old_ms,
        color=BLUE,
        linewidth=2.2,
        label="Covariance + inverse",
    )[0]
    factored_line = runtime_axis.plot(
        dimension,
        factored_ms,
        color=ORANGE,
        linewidth=2.2,
        label="Factored once",
    )[0]
    runtime_axis.fill_between(
        dimension,
        np.maximum(old_ms - old_mad_ms, np.finfo(float).tiny),
        old_ms + old_mad_ms,
        color=BLUE,
        alpha=0.14,
        linewidth=0,
    )
    runtime_axis.fill_between(
        dimension,
        np.maximum(factored_ms - factored_mad_ms, np.finfo(float).tiny),
        factored_ms + factored_mad_ms,
        color=ORANGE,
        alpha=0.14,
        linewidth=0,
    )
    runtime_axis.scatter(
        dimension[matched], old_ms[matched], color=BLUE, marker="o", zorder=3
    )
    runtime_axis.scatter(
        dimension[matched], factored_ms[matched], color=ORANGE, marker="s", zorder=3
    )
    runtime_axis.scatter(
        dimension[diverged],
        old_ms[diverged],
        facecolors="white",
        edgecolors=BLUE,
        linewidths=1.8,
        marker="o",
        zorder=4,
    )
    runtime_axis.scatter(
        dimension[diverged],
        factored_ms[diverged],
        facecolors="white",
        edgecolors=ORANGE,
        linewidths=1.8,
        marker="s",
        zorder=4,
    )
    runtime_axis.set_xscale("log", base=2)
    runtime_axis.set_yscale("log")
    runtime_axis.set_ylabel("Jitted step time (ms)")
    runtime_axis.set_title("Full nested-sampler step; compilation excluded", loc="left")
    runtime_axis.grid(True, which="major", color=GRID, linewidth=0.8)
    runtime_axis.annotate(
        format_runtime(old_ms[-1]),
        (dimension[-1], old_ms[-1]),
        xytext=(-7, 8),
        textcoords="offset points",
        ha="right",
        fontweight="bold",
    )
    runtime_axis.annotate(
        format_runtime(factored_ms[-1]),
        (dimension[-1], factored_ms[-1]),
        xytext=(-7, -14),
        textcoords="offset points",
        ha="right",
        fontweight="bold",
    )
    diverged_handle = Line2D(
        [],
        [],
        color="0.4",
        marker="s",
        markerfacecolor="white",
        linestyle="none",
        label="Fixed-seed paths diverged",
    )
    runtime_axis.legend(
        handles=(old_line, factored_line, diverged_handle),
        loc="upper left",
        frameon=False,
    )

    speedup_axis.plot(dimension, speedup, color=ORANGE, linewidth=2.2)
    speedup_axis.fill_between(
        dimension,
        1.0,
        speedup,
        where=speedup >= 1.0,
        color=ORANGE,
        alpha=0.14,
        interpolate=True,
    )
    speedup_axis.scatter(
        dimension[matched], speedup[matched], color=ORANGE, marker="s", zorder=3
    )
    speedup_axis.scatter(
        dimension[diverged],
        speedup[diverged],
        facecolors="white",
        edgecolors=ORANGE,
        linewidths=1.8,
        marker="s",
        zorder=4,
    )
    speedup_axis.axhline(1.0, color="0.45", linewidth=1.0, linestyle="--")
    speedup_axis.set_ylabel("Speedup (old / factored)")
    speedup_axis.set_xlabel("Dimension")
    speedup_axis.set_ylim(
        min(0.85, float(np.min(speedup)) * 0.95),
        max(1.2, float(np.max(speedup)) * 1.15),
    )
    speedup_axis.grid(True, which="major", color=GRID, linewidth=0.8)

    peak_index = int(np.argmax(speedup))
    peak_alignment = "center"
    if peak_index == 0:
        peak_alignment = "left"
    elif peak_index == len(speedup) - 1:
        peak_alignment = "right"
    speedup_axis.annotate(
        f"{speedup[peak_index]:.2f}× peak",  # noqa: E231
        (dimension[peak_index], speedup[peak_index]),
        xytext=(0, 10),
        textcoords="offset points",
        ha=peak_alignment,
        fontweight="bold",
    )
    endpoint_label = f"{speedup[-1]:.2f}×"  # noqa: E231
    if repeats[-1] == 1:
        endpoint_label += "\nsingle timed step"
    speedup_axis.annotate(
        endpoint_label,
        (dimension[-1], speedup[-1]),
        xytext=(-7, -2),
        textcoords="offset points",
        ha="right",
        va="center",
    )

    if len(dimension) <= 10:
        x_ticks = dimension
    else:
        x_ticks = np.asarray([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    speedup_axis.set_xticks(x_ticks, labels=[str(value) for value in x_ticks])
    num_live_factor = data["num_live"][0] // dimension[0]
    inner_steps_factor = data["num_inner_steps"][0] // dimension[0]
    footer = (
        f"{metadata['device']} · JAX {metadata['jax_version']} · "
        f"{metadata['dtype']} · N={num_live_factor}d · M={inner_steps_factor}d · "
        "bands show median ± MAD"
    )
    if repeats[-1] == 1:
        footer += f" · {dimension[-1]}D has one timed repetition"
    figure.text(
        0.5,
        -0.015,
        footer,
        ha="center",
        color="0.35",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse optional input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(plot(arguments.data, arguments.output))
