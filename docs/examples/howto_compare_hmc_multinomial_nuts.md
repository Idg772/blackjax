---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Comparing HMC, Multinomial HMC, and NUTS

This notebook compares three Hamiltonian Monte Carlo variants available in BlackJAX:

- **HMC**: proposes only the trajectory endpoint after a fixed number of leapfrog steps.
- **Multinomial HMC**: samples from the *full* leapfrog trajectory using progressive reservoir sampling weighted by exp(-H(z_i)), instead of only proposing the endpoint. Drop-in replacement for HMC with the same parameters.
- **NUTS**: dynamically determines the trajectory length at runtime using the No-U-Turn criterion.

We measure **ESS per gradient evaluation** on a 10-dimensional standard normal to show how each sampler trades off exploration efficiency against computational cost.

```{code-cell}
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import blackjax
from blackjax.diagnostics import effective_sample_size
```

```{code-cell}
jax.config.update("jax_platform_name", "cpu")

from datetime import date

rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
```

## Setup

We use a 10-dimensional standard normal as the target distribution, with 4 chains and 2,000 post-warmup samples per chain.

```{code-cell}
ndim = 10


def logdensity_fn(x):
    return -0.5 * jnp.sum(x**2)


num_chains = 4
num_warmup = 500
num_samples = 2000
```

## Shared Inference Loop

We define a `lax.scan`-based inference loop shared across all samplers.

```{code-cell}
def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states
```

## Run Samplers

We compare HMC and Multinomial HMC across trajectory lengths L=10, 30, 50, and include NUTS (which sets its own trajectory length dynamically). Each sampler uses `window_adaptation` for warmup.

```{code-cell}
results = {}

# Fixed-trajectory samplers: HMC and Multinomial HMC
for algo_idx, (name, algorithm) in enumerate(
    [
        ("hmc", blackjax.hmc),
        ("multinomial_hmc", blackjax.multinomial_hmc),
    ]
):
    for num_steps in [10, 30, 50]:
        chain_samples = []
        for chain_idx in range(num_chains):
            seed = algo_idx * 1000 + num_steps * 10 + chain_idx
            key = jax.random.fold_in(rng_key, seed)
            warmup_key, sample_key = jax.random.split(key)

            warmup = blackjax.window_adaptation(
                algorithm=algorithm,
                logdensity_fn=logdensity_fn,
                num_integration_steps=num_steps,
            )
            (state, parameters), _ = warmup.run(
                warmup_key, jnp.zeros(ndim), num_steps=num_warmup
            )

            kernel = jax.jit(algorithm(logdensity_fn, **parameters).step)
            states = inference_loop(sample_key, kernel, state, num_samples)
            chain_samples.append(states.position)

        all_samples = jnp.stack(chain_samples)  # (num_chains, num_samples, ndim)
        ess = effective_sample_size(all_samples)  # (ndim,)
        mean_ess = float(jnp.mean(ess))
        total_grads = num_samples * num_steps
        ess_per_grad = mean_ess / total_grads

        results[(name, num_steps)] = {
            "mean_ess": mean_ess,
            "total_grads": total_grads,
            "ess_per_grad": ess_per_grad,
        }
        print(
            f"{name:20s} L={num_steps:3d}  ESS={mean_ess:8.1f}  "
            f"grads={total_grads:7d}  ESS/grad={ess_per_grad:.6f}"
        )
```

```{code-cell}
# NUTS (dynamic trajectory length)
nuts_chain_samples = []
for chain_idx in range(num_chains):
    key = jax.random.fold_in(rng_key, 2000 + chain_idx)
    warmup_key, sample_key = jax.random.split(key)

    warmup = blackjax.window_adaptation(
        algorithm=blackjax.nuts,
        logdensity_fn=logdensity_fn,
    )
    (state, parameters), _ = warmup.run(
        warmup_key, jnp.zeros(ndim), num_steps=num_warmup
    )

    kernel = jax.jit(blackjax.nuts(logdensity_fn, **parameters).step)
    states = inference_loop(sample_key, kernel, state, num_samples)
    nuts_chain_samples.append(states.position)

all_samples = jnp.stack(nuts_chain_samples)
ess = effective_sample_size(all_samples)
mean_ess = float(jnp.mean(ess))
results[("nuts", "dynamic")] = {
    "mean_ess": mean_ess,
}
print(f"{'nuts':20s} L=dynamic  ESS={mean_ess:8.1f}")
```

## Results

### ESS per Gradient: HMC vs Multinomial HMC

```{code-cell}
print(f"{'Sampler':<22s} {'L':>5s} {'ESS':>10s} {'Grads':>10s} {'ESS/grad':>12s}")
print("-" * 62)
for (name, num_steps), res in sorted(results.items(), key=lambda x: str(x[0])):
    if name == "nuts":
        continue
    print(
        f"{name:<22s} {num_steps:5d} {res['mean_ess']:10.1f} "
        f"{res['total_grads']:10d} {res['ess_per_grad']:12.6f}"
    )

print(f"\n{'nuts':<22s} {'dyn':>5s} {results[('nuts', 'dynamic')]['mean_ess']:10.1f}")
```

```{code-cell}
print("\n--- Ratio (multinomial_hmc / hmc) ---")
for num_steps in [10, 30, 50]:
    hmc_r = results[("hmc", num_steps)]
    mhmc_r = results[("multinomial_hmc", num_steps)]
    ratio = mhmc_r["ess_per_grad"] / hmc_r["ess_per_grad"]
    print(f"L={num_steps:3d}  ESS/grad ratio = {ratio:.3f}")
```

### Visualization

```{code-cell}
trajectory_lengths = [10, 30, 50]

hmc_ess = [results[("hmc", L)]["ess_per_grad"] for L in trajectory_lengths]
mhmc_ess = [results[("multinomial_hmc", L)]["ess_per_grad"] for L in trajectory_lengths]

import numpy as np

x = np.arange(len(trajectory_lengths))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width / 2, hmc_ess, width, label="HMC")
ax.bar(x + width / 2, mhmc_ess, width, label="Multinomial HMC")

ax.set_xlabel("Trajectory length (L)")
ax.set_ylabel("ESS / gradient evaluation")
ax.set_title("ESS per Gradient: HMC vs Multinomial HMC")
ax.set_xticks(x)
ax.set_xticklabels([str(L) for L in trajectory_lengths])
ax.legend()
fig.tight_layout()
plt.show()
```

## Trace Plots (L=30)

We visualize the first two dimensions of the samples from each sampler (using L=30 for HMC and Multinomial HMC).

```{code-cell}
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

# Collect single-chain samples for trace plots
trace_data = {}
for seed_offset, (name, algorithm) in enumerate(
    [
        ("HMC", blackjax.hmc),
        ("Multinomial HMC", blackjax.multinomial_hmc),
    ]
):
    key = jax.random.fold_in(rng_key, 3000 + seed_offset)
    warmup_key, sample_key = jax.random.split(key)

    warmup = blackjax.window_adaptation(
        algorithm=algorithm,
        logdensity_fn=logdensity_fn,
        num_integration_steps=30,
    )
    (state, parameters), _ = warmup.run(
        warmup_key, jnp.zeros(ndim), num_steps=num_warmup
    )
    kernel = jax.jit(algorithm(logdensity_fn, **parameters).step)
    states = inference_loop(sample_key, kernel, state, num_samples)
    trace_data[name] = states.position

# NUTS trace
key = jax.random.fold_in(rng_key, 3002)
warmup_key, sample_key = jax.random.split(key)
warmup = blackjax.window_adaptation(
    algorithm=blackjax.nuts,
    logdensity_fn=logdensity_fn,
)
(state, parameters), _ = warmup.run(warmup_key, jnp.zeros(ndim), num_steps=num_warmup)
kernel = jax.jit(blackjax.nuts(logdensity_fn, **parameters).step)
states = inference_loop(sample_key, kernel, state, num_samples)
trace_data["NUTS"] = states.position

for row, (name, samples) in enumerate(trace_data.items()):
    axes[row, 0].plot(samples[:, 0], linewidth=0.5)
    axes[row, 0].set_ylabel(f"{name}\ndim 0")

    axes[row, 1].plot(samples[:, 1], linewidth=0.5)
    axes[row, 1].set_ylabel(f"{name}\ndim 1")

axes[-1, 0].set_xlabel("Sample index")
axes[-1, 1].set_xlabel("Sample index")
fig.suptitle("Trace plots (L=30 for HMC/Multinomial HMC, dynamic for NUTS)", y=1.01)
fig.tight_layout()
plt.show()
```

## Summary

- **Multinomial HMC** achieves substantially higher ESS per gradient evaluation than standard HMC, especially at shorter trajectory lengths where HMC often overshoots and wastes the entire trajectory.
- **NUTS** dynamically adapts its trajectory length and generally provides excellent ESS without requiring the user to tune `num_integration_steps`.
- Multinomial HMC is a drop-in replacement for HMC: same parameters, same state type (`HMCState`), and compatible with `window_adaptation`.
