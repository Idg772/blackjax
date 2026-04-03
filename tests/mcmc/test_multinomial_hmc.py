"""Tests for the Multinomial HMC kernel."""

import jax
import jax.numpy as jnp
from absl.testing import absltest

import blackjax
from blackjax.mcmc.hmc import HMCInfo, HMCState
from tests.fixtures import BlackJAXTest, std_normal_logdensity


class MultinomialHMCTest(BlackJAXTest):
    """Unit tests for the multinomial HMC sampler."""

    def test_sampling_algorithm_interface(self):
        """The high-level API returns a SamplingAlgorithm with init/step."""
        sampler = blackjax.multinomial_hmc(
            std_normal_logdensity,
            step_size=0.1,
            inverse_mass_matrix=jnp.array([1.0]),
            num_integration_steps=10,
        )
        state = sampler.init(jnp.array(0.5))
        self.assertIsInstance(state, HMCState)

        new_state, info = jax.jit(sampler.step)(self.next_key(), state)
        self.assertIsInstance(new_state, HMCState)
        self.assertIsInstance(info, HMCInfo)

    def test_correct_sampling(self):
        """On a standard normal, the sampler should produce reasonable samples."""
        sampler = blackjax.multinomial_hmc(
            std_normal_logdensity,
            step_size=0.5,
            inverse_mass_matrix=jnp.array([1.0]),
            num_integration_steps=20,
        )
        state = sampler.init(jnp.array(0.0))
        step = jax.jit(sampler.step)

        states = []
        for i in range(500):
            key = jax.random.fold_in(self.next_key(), i)
            state, info = step(key, state)
            states.append(state.position)

        samples = jnp.stack(states)
        # Mean should be close to 0, std close to 1 for a standard normal
        self.assertAlmostEqual(float(jnp.mean(samples)), 0.0, delta=0.3)
        self.assertAlmostEqual(float(jnp.std(samples)), 1.0, delta=0.3)

    def test_divergence_detection(self):
        """With a huge step size the sampler should flag divergences."""
        sampler = blackjax.multinomial_hmc(
            std_normal_logdensity,
            step_size=1000.0,
            inverse_mass_matrix=jnp.array([1.0]),
            num_integration_steps=100,
            divergence_threshold=100,
        )
        state = sampler.init(jnp.array(0.0))
        _, info = jax.jit(sampler.step)(self.next_key(), state)
        self.assertTrue(info.is_divergent)

    def test_acceptance_rate(self):
        """With a well-tuned step size the acceptance rate should be high."""
        sampler = blackjax.multinomial_hmc(
            std_normal_logdensity,
            step_size=0.1,
            inverse_mass_matrix=jnp.array([1.0]),
            num_integration_steps=10,
        )
        state = sampler.init(jnp.array(0.0))
        _, info = jax.jit(sampler.step)(self.next_key(), state)
        self.assertGreater(float(info.acceptance_rate), 0.5)

    def test_is_always_accepted(self):
        """Multinomial HMC always accepts (no MH reject step)."""
        sampler = blackjax.multinomial_hmc(
            std_normal_logdensity,
            step_size=0.1,
            inverse_mass_matrix=jnp.array([1.0]),
            num_integration_steps=10,
        )
        state = sampler.init(jnp.array(0.0))
        _, info = jax.jit(sampler.step)(self.next_key(), state)
        self.assertTrue(info.is_accepted)

    def test_pytree_position(self):
        """The sampler should handle dict-structured positions."""
        logdensity_fn = std_normal_logdensity

        sampler = blackjax.multinomial_hmc(
            logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.array([1.0, 1.0]),
            num_integration_steps=10,
        )
        state = sampler.init({"a": jnp.array(0.0), "b": jnp.array(1.0)})
        new_state, info = jax.jit(sampler.step)(self.next_key(), state)
        self.assertIn("a", new_state.position)
        self.assertIn("b", new_state.position)

    def test_build_kernel_api(self):
        """The low-level build_kernel API should work."""
        kernel = blackjax.multinomial_hmc.build_kernel()
        state = blackjax.multinomial_hmc.init(jnp.array(0.0), std_normal_logdensity)

        new_state, info = jax.jit(kernel, static_argnums=(2,))(
            self.next_key(),
            state,
            std_normal_logdensity,
            0.1,
            jnp.array([1.0]),
            10,
        )
        self.assertIsInstance(new_state, HMCState)
        self.assertIsInstance(info, HMCInfo)


if __name__ == "__main__":
    absltest.main()
