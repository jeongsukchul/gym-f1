import functools
from typing import NamedTuple, Callable, Tuple
import chex
import jax.numpy as jnp

from gymkhana.jax_sampler_ppo.gmmvi.sample_db import SampleDB, SampleDBState
from gymkhana.jax_sampler_ppo.gmmvi.gmm_setup import GMMWrapperState, GMMWrapper
import jax
import time




class SampleSelector(NamedTuple):
    select_samples: Callable
    save_samples_and_select: Callable
    save_samples: Callable
    select_train_datas: Callable

def setup_fixed_sample_selector(sample_db: SampleDB, gmm_wrapper: GMMWrapper,
                                SIMUL_SAMPLES, BATCH_SAMPLES):

    # @functools.partial(jax.jit, static_argnames=['num_components'])
    def _sample_desired_samples(gmm_wrapper_state: GMMWrapperState,
                                seed: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        seed ,mapping_seed = jax.random.split(seed)
        mapping = jax.random.randint(
            mapping_seed,
            shape=(SIMUL_SAMPLES,),
            minval=0,
            maxval=gmm_wrapper_state.gmm_state.num_components,   # can be a tracer
            dtype=jnp.int32,
        )
        # mapping = jax.random.categorical(
        #     mapping_seed,
        #     gmm_wrapper_state.gmm_state.component_mask, gmm_wrapper_state.gmm_state.log_weights,
        #     shape=(SIMUL_SAMPLES,),
        # )
        mapping = jnp.sort(mapping)
        new_samples, _ = gmm_wrapper.sample_from_components_shuffle(gmm_wrapper_state.gmm_state,
                                                                             mapping,
                                                                             seed)
        return new_samples, mapping
    def select_samples(gmm_wrapper_state: GMMWrapperState, seed: chex.PRNGKey):
        new_samples, mapping = _sample_desired_samples(gmm_wrapper_state, seed)
        return new_samples, mapping
    def save_samples(gmm_wrapper_state: GMMWrapperState, sampledb_state: SampleDBState, 
                     new_samples, new_target_lnpdfs, new_target_grads, mapping):
        return sample_db.add_samples(sampledb_state, new_samples, gmm_wrapper_state.gmm_state.means,
                    gmm_wrapper_state.gmm_state.chol_covs, new_target_lnpdfs,
                    new_target_grads, mapping)
    def select_train_datas(sampledb_state):
        old_samples_pdf, samples, mapping, target_lnpdfs, target_grads = sample_db.get_newest_samples(
            sampledb_state, BATCH_SAMPLES)
        return samples, mapping, old_samples_pdf, target_lnpdfs, target_grads
    def save_samples_and_select(gmm_wrapper_state: GMMWrapperState, sampledb_state: SampleDBState, \
                        new_samples, new_target_lnpdfs, new_target_grads,
                        mapping) :
        # num_new_samples = SIMUL_SAMPLES*gmm_wrapper_state.gmm_state.num_components
        sampledb_state = sample_db.add_samples(sampledb_state, new_samples, gmm_wrapper_state.gmm_state.means,
                                               gmm_wrapper_state.gmm_state.chol_covs, new_target_lnpdfs,
                                               new_target_grads, mapping)
        old_samples_pdf, samples, mapping, target_lnpdfs, target_grads = sample_db.get_newest_samples(
            sampledb_state, BATCH_SAMPLES)

        return sampledb_state, samples, mapping, old_samples_pdf, target_lnpdfs, target_grads

    return SampleSelector(select_samples=select_samples, save_samples=save_samples, select_train_datas=select_train_datas, save_samples_and_select=save_samples_and_select)


# def setup_vips_sample_selector(sample_db: SampleDB, gmm_wrapper: GMMWrapper,
#                                SIMUL_SAMPLES, BATCH_SAMPLES):

#     @jax.jit
#     def _get_effective_weights(model_densities: chex.Array, old_samples_pdf: chex.Array) -> chex.Array:
#         log_weight = model_densities - jnp.expand_dims(old_samples_pdf, axis=0)
#         log_weight = log_weight - jax.nn.logsumexp(log_weight, axis=1, keepdims=True)
#         weights = jnp.exp(log_weight)
#         print('weights', weights)
#         # num_effective_samples = 1. / jnp.sum(weights * weights, axis=1)
#         return weights

#     def _sample_where_needed(gmm_wrapper_state: GMMWrapperState,
#                              samples: chex.Array, seed: chex.Array, old_samples_pdf: chex.Array,
#                              ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:

#         seed ,mapping_seed = jax.random.split(seed)
#         if jnp.shape(samples)[0] == 0:
#             num_effective_samples = jnp.zeros(gmm_wrapper_state.gmm_state.num_components, dtype=jnp.int32)
#         else:
#             model_logpdfs = jax.vmap(gmm_wrapper.component_log_densities, in_axes=(None, 0))(gmm_wrapper_state.gmm_state, samples)
#             model_logpdfs = jnp.transpose(model_logpdfs)
#             weights = jnp.array(jnp.floor(_get_effective_weights(model_logpdfs, old_samples_pdf)),
#                                               dtype=jnp.int32)
#         mapping = jax.random.randint(
#             mapping_seed,
#             shape=(SIMUL_SAMPLES,),
#             minval=0,
#             maxval=gmm_wrapper_state.gmm_state.num_components,   # can be a tracer
#             dtype=jnp.int32,
#         )
#         new_samples, mapping = gmm_wrapper.sample_from_components_shuffle(gmm_wrapper_state.gmm_state,
#                                                                              mapping,
#                                                                              seed)
#         # new_target_grads, new_target_lnpdfs = get_target_grads(new_samples)
#         return new_samples, mapping

#     def select_samples(gmm_wrapper_state: GMMWrapperState, sampledb_state: SampleDBState, seed: chex.PRNGKey):
#         # Get old samples from the database
#         old_samples_pdf, samples, _, _, _ = sample_db.get_newest_samples(sampledb_state, BATCH_SAMPLES)
#         new_samples, mapping = _sample_where_needed(gmm_wrapper_state, samples, seed, old_samples_pdf)

#         return new_samples, mapping
#     def save_samples(gmm_wrapper_state: GMMWrapperState, sampledb_state: SampleDBState, 
#                      new_samples, new_target_lnpdfs, new_target_grads, mapping):
#         return sample_db.add_samples(sampledb_state, new_samples, gmm_wrapper_state.gmm_state.means,
#                     gmm_wrapper_state.gmm_state.chol_covs, new_target_lnpdfs,
#                     new_target_grads, mapping)
#     def select_train_datas(sampledb_state):
#         old_samples_pdf, samples, mapping, target_lnpdfs, target_grads = sample_db.get_newest_samples(
#             sampledb_state, BATCH_SAMPLES)
#         return samples, mapping, old_samples_pdf, target_lnpdfs, target_grads
#     def save_samples_and_select(gmm_wrapper_state: GMMWrapperState, sampledb_state: SampleDBState, \
#                      new_samples, new_target_lnpdfs, new_target_grads, \
#                      mapping, num_reused_samples) :

#         sampledb_state = sample_db.add_samples(sampledb_state, new_samples, gmm_wrapper_state.gmm_state.means,
#                                                gmm_wrapper_state.gmm_state.chol_covs, new_target_lnpdfs,
#                                                new_target_grads, mapping)
#         num_new_samples = jnp.shape(new_samples)[0]

#         # We call get_newest_samples again in order to recompute the background densities
#         old_samples_pdf, samples, mapping, target_lnpdfs, target_grads = sample_db.get_newest_samples(
#             sampledb_state, int(num_reused_samples + num_new_samples))
#         return sampledb_state, samples, mapping, old_samples_pdf, target_lnpdfs, target_grads

#     return SampleSelector(select_samples=select_samples, save_samples=save_samples, select_train_datas=select_train_datas, save_samples_and_select=save_samples_and_select)


# def setup_lin_sample_selector(sample_db: SampleDB, gmm_wrapper: GMMWrapper, target_log_prob_fn: Callable,
#                               DESIRED_SAMPLES_PER_COMPONENT, RATIO_REUSED_SAMPLES_TO_DESIRED):


#     def _get_effective_samples(model_densities: chex.Array, old_samples_pdf: chex.Array) -> chex.Array:

#         log_weight = model_densities - jnp.expand_dims(old_samples_pdf, axis=0)
#         log_weight = log_weight - jax.nn.logsumexp(log_weight, axis=1, keepdims=True)
#         weights = jnp.exp(log_weight)
#         num_effective_samples = 1. / jnp.sum(weights * weights, axis=1)
#         return num_effective_samples

#     def _sample_where_needed(gmm_wrapper_state: GMMWrapperState,
#                              sampledb_state: SampleDBState, seed) -> Tuple[chex.Array, chex.Array, int]:

#         # Get old samples from the database
#         num_samples_to_reuse = (jnp.int32(jnp.floor(RATIO_REUSED_SAMPLES_TO_DESIRED * DESIRED_SAMPLES_PER_COMPONENT)) *
#                                 gmm_wrapper_state.gmm_state.num_components)
#         old_samples_pdf, old_samples, _, _, _ = sample_db.get_newest_samples(sampledb_state, num_samples_to_reuse)
#         num_reused_samples = jnp.shape(old_samples)[0]

#         # Get additional samples to ensure a desired effective sample size for every component
#         if jnp.shape(old_samples)[0] == 0:
#             num_effective_samples = jnp.zeros(1, dtype=jnp.int32)
#         else:
#             model_logpdfs = gmm_wrapper.log_density(old_samples)
#             num_effective_samples = jnp.floor(_get_effective_samples(model_logpdfs, old_samples_pdf))
#         num_additional_samples = jnp.maximum(1, DESIRED_SAMPLES_PER_COMPONENT -
#                                              num_effective_samples)
#         new_samples, mapping = gmm_wrapper.sample(gmm_wrapper_state.gmm_state, seed, jnp.squeeze(num_additional_samples))

#         return new_samples, mapping, num_reused_samples

#     def select_samples(gmm_wrapper_state: GMMWrapperState, sampledb_state: SampleDBState, seed: chex.PRNGKey) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
#         # Get additional samples to ensure a desired effective sample size for every component
#         new_samples, mapping, num_reused_samples = _sample_where_needed(gmm_wrapper_state, sampledb_state, seed)

#         new_target_grads, new_target_lnpdfs = get_target_grads(new_samples)

#         sampledb_state = sample_db.add_samples(sampledb_state, new_samples, gmm_wrapper_state.gmm_state.means,
#                                                gmm_wrapper_state.gmm_state.chol_covs,
#                                                new_target_lnpdfs, new_target_grads, mapping)

#         # We call get_newest_samples again in order to recompute the background densities
#         samples_this_iter = num_reused_samples + jnp.shape(new_samples)[0]
#         old_samples_pdf, samples, mapping, target_lnpdfs, target_grads = sample_db.get_newest_samples(sampledb_state,
#                                                                                                       samples_this_iter)
#         return sampledb_state, samples, mapping, old_samples_pdf, target_lnpdfs, target_grads

#     # def target_uld(samples: chex.Array) -> chex.Array:
#     #     return jax.vmap(target_log_prob_fn)(samples)

#     # def get_target_grads(samples: chex.Array) -> Tuple[chex.Array, chex.Array]:
#     #     target, gradient = jax.vmap(jax.value_and_grad(target_log_prob_fn))(samples)
#     #     return gradient, target

#     return SampleSelector(
#                           select_samples=select_samples)
