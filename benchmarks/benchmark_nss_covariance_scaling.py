"""Generate timing data for the NSS covariance-factor benchmark.

The legacy implementation is kept locally in this benchmark so production code
does not retain the inverse-based path. Each measurement times one full jitted
nested-sampler step after compilation.
"""

import argparse
import csv
import gc
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

from blackjax.ns import nss
from blackjax.smc.tuning.from_particles import particles_covariance_matrix
from blackjax.types import Array, ArrayTree, PRNGKey

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = HERE / "nss_covariance_scaling.csv"
DEFAULT_DIMENSIONS = (
    16,
    32,
    64,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
)
DEFAULT_REPEATS = {
    16: 21,
    32: 21,
    64: 21,
    128: 21,
    192: 9,
    256: 11,
    384: 9,
    512: 7,
    768: 9,
    1024: 5,
    1536: 5,
    2048: 5,
    3072: 3,
    4096: 1,
}
CSV_COLUMNS = (
    "dimension",
    "num_live",
    "num_inner_steps",
    "num_delete",
    "max_steps",
    "max_shrinkage",
    "repeats",
    "old_ms",
    "old_mad_ms",
    "factored_ms",
    "factored_mad_ms",
    "speedup",
    "matched_paths",
    "max_position_difference",
    "sample_kind",
    "seed",
    "backend",
    "device",
    "jax_version",
    "dtype",
)


@dataclass(frozen=True)
class BenchmarkResult:
    """One dimension's full-step timing summary."""

    dimension: int
    num_live: int
    num_inner_steps: int
    num_delete: int
    max_steps: int
    max_shrinkage: int
    repeats: int
    old_ms: float
    old_mad_ms: float
    factored_ms: float
    factored_mad_ms: float
    speedup: float
    matched_paths: bool
    max_position_difference: float
    sample_kind: str
    seed: int
    backend: str
    device: str
    jax_version: str
    dtype: str


def legacy_sample_direction_from_covariance(
    rng_key: PRNGKey, position: ArrayTree, cov: Array
) -> ArrayTree:
    """Reproduce the pre-refactor covariance draw and inverse normalization."""
    _, unravel_fn = jax.flatten_util.ravel_pytree(position)
    direction = jax.random.multivariate_normal(
        rng_key,
        jnp.zeros(cov.shape[0], dtype=cov.dtype),
        cov,
    )
    direction = 2.0 * direction / jnp.sqrt(direction @ jnp.linalg.inv(cov) @ direction)
    return unravel_fn(direction)


def legacy_covariance_proposal(
    init_state_fn: Callable, loglikelihood_0: Array, cov: Array
) -> Callable:
    """Build the pre-refactor NSS proposal for the benchmark comparison."""

    def proposal_generator(rng_key, position, logdensity_fn):
        del logdensity_fn
        direction = legacy_sample_direction_from_covariance(rng_key, position, cov)

        def slice_fn(t):
            candidate = jax.tree.map(
                lambda value, delta: value + t * delta,
                position,
                direction,
            )
            new_state = init_state_fn(
                candidate,
                loglikelihood_birth=loglikelihood_0,
            )
            return new_state, new_state.loglikelihood > loglikelihood_0

        return slice_fn

    return proposal_generator


def legacy_live_covariance(
    rng_key: PRNGKey,
    state,
    info,
    params: dict[str, ArrayTree] | None = None,
) -> dict[str, ArrayTree]:
    """Return the unfactored live covariance used before the refactor."""
    del rng_key, info, params
    cov = jnp.atleast_2d(particles_covariance_matrix(state.particles.position))
    return {"cov": cov}


def gaussian_logprior(position: Array) -> Array:
    """Unnormalized standard-normal log density for a live point."""
    return -0.5 * jnp.vdot(position, position)


def gaussian_loglikelihood(position: Array) -> Array:
    """Unnormalized shifted-normal log likelihood for a live point."""
    residual = position - 1.0
    return -0.5 * jnp.vdot(residual, residual)


def block_until_ready(tree):
    """Block until every JAX array leaf in a result PyTree is ready."""
    return jax.tree.map(lambda leaf: leaf.block_until_ready(), tree)


def timed_step(step: Callable, rng_key: PRNGKey, state) -> float:
    """Run and synchronize one already-compiled sampler step."""
    start = time.perf_counter_ns()
    result = step(rng_key, state)
    block_until_ready(result)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    return elapsed_ms


def median_and_mad(samples: list[float]) -> tuple[float, float]:
    """Return the median and median absolute deviation in milliseconds."""
    values = np.asarray(samples)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad


def fixed_six(value: float) -> str:
    """Serialize a timing or ratio with stable six-place CSV formatting."""
    return np.format_float_positional(
        value,
        precision=6,
        unique=False,
        fractional=True,
        trim="k",
    )


def automatic_repeats(dimension: int) -> int:
    """Choose a bounded timing count for a custom dimension."""
    if dimension in DEFAULT_REPEATS:
        return DEFAULT_REPEATS[dimension]
    if dimension <= 128:
        return 21
    if dimension <= 256:
        return 11
    if dimension <= 768:
        return 9
    if dimension <= 2048:
        return 5
    if dimension <= 3072:
        return 3
    return 1


def device_label(device) -> str:
    """Build a useful hardware label without platform-specific dependencies."""
    kind = getattr(device, "device_kind", device.platform)
    machine = platform.machine()
    return f"{kind} ({machine})" if machine else str(kind)


def benchmark_dimension(
    dimension: int,
    repeats: int,
    num_live_factor: int,
    inner_steps_factor: int,
    num_delete: int,
    max_steps: int,
    max_shrinkage: int,
    seed: int,
) -> BenchmarkResult:
    """Compile and benchmark both full NSS steps at one dimension."""
    num_live = num_live_factor * dimension
    num_inner_steps = inner_steps_factor * dimension
    root_key = jax.random.fold_in(jax.random.key(seed), dimension)
    positions_key, validation_key, timing_key = jax.random.split(root_key, 3)
    positions = jax.random.normal(
        positions_key,
        shape=(num_live, dimension),
        dtype=jnp.float32,
    )

    legacy_algorithm = nss.as_top_level_api(
        gaussian_logprior,
        gaussian_loglikelihood,
        num_inner_steps=num_inner_steps,
        num_delete=num_delete,
        max_steps=max_steps,
        max_shrinkage=max_shrinkage,
        proposal=legacy_covariance_proposal,
        inner_kernel_params=legacy_live_covariance,
    )
    factored_algorithm = nss.as_top_level_api(
        gaussian_logprior,
        gaussian_loglikelihood,
        num_inner_steps=num_inner_steps,
        num_delete=num_delete,
        max_steps=max_steps,
        max_shrinkage=max_shrinkage,
    )
    legacy_state = legacy_algorithm.init(positions, root_key)
    factored_state = factored_algorithm.init(positions, root_key)
    block_until_ready((legacy_state, factored_state))

    legacy_step = jax.jit(legacy_algorithm.step)
    factored_step = jax.jit(factored_algorithm.step)

    # These first calls compile and provide the fixed-key equivalence check. Their
    # execution time is deliberately excluded from the timing samples.
    legacy_validation = legacy_step(validation_key, legacy_state)
    factored_validation = factored_step(validation_key, factored_state)
    block_until_ready((legacy_validation, factored_validation))
    legacy_position = legacy_validation[0].particles.position
    factored_position = factored_validation[0].particles.position
    max_position_difference = float(
        jnp.max(jnp.abs(legacy_position - factored_position))
    )
    matched_paths = bool(
        jnp.allclose(legacy_position, factored_position, rtol=1e-5, atol=1e-5)
    )
    del legacy_validation, factored_validation, legacy_position, factored_position

    legacy_times = []
    factored_times = []
    for repetition in range(repeats):
        repeat_key = jax.random.fold_in(timing_key, repetition)
        if repetition % 2 == 0:
            legacy_ms = timed_step(legacy_step, repeat_key, legacy_state)
            factored_ms = timed_step(factored_step, repeat_key, factored_state)
        else:
            factored_ms = timed_step(factored_step, repeat_key, factored_state)
            legacy_ms = timed_step(legacy_step, repeat_key, legacy_state)
        legacy_times.append(legacy_ms)
        factored_times.append(factored_ms)

    legacy_median, legacy_mad = median_and_mad(legacy_times)
    factored_median, factored_mad = median_and_mad(factored_times)
    device = jax.devices()[0]
    return BenchmarkResult(
        dimension=dimension,
        num_live=num_live,
        num_inner_steps=num_inner_steps,
        num_delete=num_delete,
        max_steps=max_steps,
        max_shrinkage=max_shrinkage,
        repeats=repeats,
        old_ms=legacy_median,
        old_mad_ms=legacy_mad,
        factored_ms=factored_median,
        factored_mad_ms=factored_mad,
        speedup=legacy_median / factored_median,
        matched_paths=matched_paths,
        max_position_difference=max_position_difference,
        sample_kind="single" if repeats == 1 else "median",
        seed=seed,
        backend=device.platform,
        device=device_label(device),
        jax_version=jax.__version__,
        dtype=str(positions.dtype),
    )


def write_results(output_path: Path, results: list[BenchmarkResult]) -> None:
    """Write all completed dimensions to the benchmark CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=CSV_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["old_ms"] = fixed_six(result.old_ms)
            row["old_mad_ms"] = fixed_six(result.old_mad_ms)
            row["factored_ms"] = fixed_six(result.factored_ms)
            row["factored_mad_ms"] = fixed_six(result.factored_mad_ms)
            row["speedup"] = fixed_six(result.speedup)
            row["matched_paths"] = str(result.matched_paths).lower()
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    """Parse benchmark dimensions and workload controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help="Dimensions to benchmark (default: the checked-in extended sweep).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        help="Use this timing count at every dimension instead of the default schedule.",
    )
    parser.add_argument("--num-live-factor", type=int, default=4)
    parser.add_argument("--inner-steps-factor", type=int, default=2)
    parser.add_argument("--num-delete", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-shrinkage", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    positive_values = (
        *args.dimensions,
        args.num_live_factor,
        args.inner_steps_factor,
        args.num_delete,
        args.max_steps,
        args.max_shrinkage,
    )
    if any(value <= 0 for value in positive_values):
        parser.error("dimensions and workload controls must be positive")
    if args.repeats is not None and args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.num_live_factor <= 1:
        parser.error("each run needs more live points than its dimension")
    return args


def main() -> None:
    """Run each requested dimension and persist progress after every result."""
    args = parse_args()
    results = []
    for dimension in args.dimensions:
        repeats = args.repeats or automatic_repeats(dimension)
        print(
            f"benchmarking d={dimension}, N={args.num_live_factor * dimension}, "
            f"M={args.inner_steps_factor * dimension}, repeats={repeats}",
            flush=True,
        )
        jax.clear_caches()
        gc.collect()
        result = benchmark_dimension(
            dimension=dimension,
            repeats=repeats,
            num_live_factor=args.num_live_factor,
            inner_steps_factor=args.inner_steps_factor,
            num_delete=args.num_delete,
            max_steps=args.max_steps,
            max_shrinkage=args.max_shrinkage,
            seed=args.seed,
        )
        results.append(result)
        write_results(args.output, results)
        print(
            f"  old={round(result.old_ms, 3)} ms, "
            f"factored={round(result.factored_ms, 3)} ms, "
            f"speedup={round(result.speedup, 3)}x, "
            f"matched_paths={result.matched_paths}",
            flush=True,
        )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
