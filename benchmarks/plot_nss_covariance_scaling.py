"""Plot the nested-sampler covariance-factor benchmark."""

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


def load_data() -> dict[str, np.ndarray]:
    """Load benchmark columns from the checked-in CSV file."""
    with DATA_PATH.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    def values(name: str, dtype: type = float) -> np.ndarray:
        return np.asarray([dtype(row[name]) for row in rows])

    return {
        "dimension": values("dimension", int),
        "old_ms": values("old_ms"),
        "old_mad_ms": values("old_mad_ms"),
        "factored_ms": values("factored_ms"),
        "factored_mad_ms": values("factored_mad_ms"),
        "speedup": values("speedup"),
        "matched_paths": np.asarray(
            [row["matched_paths"].lower() == "true" for row in rows]
        ),
    }


def plot() -> Path:
    """Generate the two-panel runtime and speedup figure."""
    data = load_data()
    dimension = data["dimension"]
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
        f"{old_ms[-1] / 1000:.1f} s",  # noqa: E231
        (dimension[-1], old_ms[-1]),
        xytext=(-7, 8),
        textcoords="offset points",
        ha="right",
        fontweight="bold",
    )
    runtime_axis.annotate(
        f"{factored_ms[-1] / 1000:.1f} s",  # noqa: E231
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
    speedup_axis.set_ylim(0.85, 3.7)
    speedup_axis.grid(True, which="major", color=GRID, linewidth=0.8)

    peak_index = int(np.argmax(speedup))
    speedup_axis.annotate(
        f"{speedup[peak_index]:.2f}× peak",  # noqa: E231
        (dimension[peak_index], speedup[peak_index]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontweight="bold",
    )
    speedup_axis.annotate(
        f"{speedup[-1]:.2f}×\nsingle timed step",  # noqa: E231
        (dimension[-1], speedup[-1]),
        xytext=(-7, -2),
        textcoords="offset points",
        ha="right",
        va="center",
    )

    x_ticks = np.asarray([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    speedup_axis.set_xticks(x_ticks, labels=[str(value) for value in x_ticks])
    figure.text(
        0.5,
        -0.015,
        "Apple M3 CPU · JAX 0.10.0 · float32 · N=4d · M=2d · bands show median ± MAD · 4096D has one timed repetition",
        ha="center",
        color="0.35",
        fontsize=9,
    )
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(plot())
