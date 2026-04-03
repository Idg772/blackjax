# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public API for the Multinomial HMC Kernel.

Multinomial HMC :cite:p:`betancourt2018conceptual` (Section 4.3) samples the
next state from the **entire leapfrog trajectory** using a categorical
distribution weighted by the canonical density ``exp(-H(z_i))``, rather than
proposing only the trajectory endpoint as standard HMC does.

This uses the same within-subtree proposal mechanism as NUTS
(``progressive_uniform_sampling``), but applied to a fixed-length trajectory
rather than a dynamically-expanding tree.  Because every leapfrog state is a
candidate, multinomial HMC yields higher effective-sample-size per gradient
evaluation than endpoint HMC, especially at shorter trajectory lengths.

The implementation reuses BlackJAX's existing progressive sampling machinery
(``proposal.progressive_uniform_sampling``) so that only O(1) memory is needed
regardless of trajectory length.
"""

from typing import Callable, Union

import jax
import jax.numpy as jnp

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.proposal import (
    Proposal,
    progressive_uniform_sampling,
    proposal_generator,
)
from blackjax.mcmc.trajectory import hmc_energy
from blackjax.types import ArrayLikeTree, PRNGKey

__all__ = [
    "init",
    "build_kernel",
    "as_top_level_api",
]

# Reuse HMCState, HMCInfo, and init directly from hmc.py
init = hmc.init


def build_kernel(
    integrator: Callable = integrators.velocity_verlet,
    divergence_threshold: float = 1000,
):
    """Build a Multinomial HMC kernel.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the Hamiltonian dynamics.
    divergence_threshold
        Value of the difference in energy above which we consider that the
        transition is divergent.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def kernel(
        rng_key: PRNGKey,
        state: hmc.HMCState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: metrics.MetricTypes,
        num_integration_steps: int,
    ) -> tuple[hmc.HMCState, hmc.HMCInfo]:
        """Generate a new sample with the Multinomial HMC kernel."""

        metric = metrics.default_metric(inverse_mass_matrix)
        symplectic_integrator = integrator(logdensity_fn, metric.kinetic_energy)
        proposal_gen = multinomial_hmc_proposal(
            symplectic_integrator,
            metric.kinetic_energy,
            step_size,
            num_integration_steps,
            divergence_threshold,
        )

        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        position, logdensity, logdensity_grad = state
        momentum = metric.sample_momentum(key_momentum, position)

        integrator_state = integrators.IntegratorState(
            position, momentum, logdensity, logdensity_grad
        )
        proposal, info = proposal_gen(key_integrator, integrator_state)
        proposal = hmc.HMCState(
            proposal.position, proposal.logdensity, proposal.logdensity_grad
        )

        return proposal, info

    return kernel


def multinomial_hmc_proposal(
    integrator: Callable,
    kinetic_energy: metrics.KineticEnergy,
    step_size: Union[float, ArrayLikeTree],
    num_integration_steps: int = 1,
    divergence_threshold: float = 1000,
) -> Callable:
    """Multinomial HMC proposal.

    Instead of proposing only the trajectory endpoint (as standard HMC does),
    this builds the full leapfrog trajectory of ``num_integration_steps`` and
    samples one state proportional to ``exp(-H(z_i))`` using progressive
    reservoir sampling (``progressive_uniform_sampling``).

    Parameters
    ----------
    integrator
        Symplectic integrator used to build the trajectory step by step.
    kinetic_energy
        Function that computes the kinetic energy.
    step_size
        Size of the integration step.
    num_integration_steps
        Number of times we run the symplectic integrator to build the trajectory.
    divergence_threshold
        Threshold above which we say that there is a divergence.

    Returns
    -------
    A function that generates a new chain state and information about the transition.

    """
    hmc_energy_fn = hmc_energy(kinetic_energy)
    _, generate_proposal = proposal_generator(hmc_energy_fn)

    def generate(
        rng_key, state: integrators.IntegratorState
    ) -> tuple[integrators.IntegratorState, hmc.HMCInfo]:
        """Generate a new chain state."""
        initial_energy = hmc_energy_fn(state)
        init_proposal = Proposal(state, initial_energy, 0.0, -jnp.inf)

        def one_step(i, carry):
            current_state, current_proposal, any_divergent = carry
            step_key = jax.random.fold_in(rng_key, i)

            new_state = integrator(current_state, step_size)
            new_proposal = generate_proposal(initial_energy, new_state)
            is_diverging = -new_proposal.weight > divergence_threshold
            any_divergent = any_divergent | is_diverging
            sampled_proposal = progressive_uniform_sampling(
                step_key, current_proposal, new_proposal
            )

            return (new_state, sampled_proposal, any_divergent)

        (_, sampled_proposal, is_divergent) = jax.lax.fori_loop(
            0, num_integration_steps, one_step, (state, init_proposal, False)
        )

        acceptance_rate = (
            jnp.exp(sampled_proposal.sum_log_p_accept) / num_integration_steps
        )

        info = hmc.HMCInfo(
            momentum=state.momentum,
            acceptance_rate=acceptance_rate,
            is_accepted=True,
            is_divergent=is_divergent,
            energy=sampled_proposal.energy,
            proposal=sampled_proposal.state,
            num_integration_steps=num_integration_steps,
        )
        return sampled_proposal.state, info

    return generate


def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    inverse_mass_matrix: metrics.MetricTypes,
    num_integration_steps: int,
    *,
    divergence_threshold: int = 1000,
    integrator: Callable = integrators.velocity_verlet,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the Multinomial HMC kernel.

    Multinomial HMC is a drop-in replacement for standard HMC that samples the
    next state from the entire trajectory rather than only the endpoint, yielding
    better effective-sample-size per gradient evaluation.

    Examples
    --------

    A new Multinomial HMC kernel can be initialized and used with the following
    code:

    .. code::

        multinomial_hmc = blackjax.multinomial_hmc(
            logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps
        )
        state = multinomial_hmc.init(position)
        new_state, info = multinomial_hmc.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(multinomial_hmc.step)
       new_state, info = step(rng_key, state)

    Should you need to you can always use the base kernel directly:

    .. code::

       import blackjax.mcmc.integrators as integrators

       kernel = blackjax.multinomial_hmc.build_kernel(integrators.mclachlan)
       state = blackjax.multinomial_hmc.init(position, logdensity_fn)
       state, info = kernel(
           rng_key,
           state,
           logdensity_fn,
           step_size,
           inverse_mass_matrix,
           num_integration_steps,
       )

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.
    inverse_mass_matrix
        The value to use for the inverse mass matrix when drawing a value for
        the momentum and computing the kinetic energy. This argument will be
        passed to the ``metrics.default_metric`` function so it supports the
        full interface presented there.
    num_integration_steps
        The number of steps we take with the symplectic integrator at each
        sample step before returning a sample.
    divergence_threshold
        The absolute value of the difference in energy between two states above
        which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    integrator
        (algorithm parameter) The symplectic integrator to use to integrate the
        trajectory.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    kernel = build_kernel(integrator, divergence_threshold)
    metric = metrics.default_metric(inverse_mass_matrix)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            metric,
            num_integration_steps,
        )

    return SamplingAlgorithm(init_fn, step_fn)
