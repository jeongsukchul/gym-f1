from functools import partial
from typing import NamedTuple, Callable, Optional
import chex
import jax.numpy as jnp
import jax
from gymkhana.jax_sampler_ppo.gmmvi.gmm_setup import GMMWrapperState


#-------------------------------------------------------------------------------------
#Component Stepsize Adaptation
#-------------------------------------------------------------------------------------
def update_component_stepsize_decaying(gmm_wrapper_state: GMMWrapperState, INITIAL_STEPSIZE, ANNEALING_EXPONENT) -> chex.Array:

    new_stepsizes = jnp.empty((gmm_wrapper_state.gmm_state.num_components,), dtype=jnp.float32)
    for i in range(jnp.shape(gmm_wrapper_state.stepsizes)[0]):
        new_stepsize = INITIAL_STEPSIZE / (1 + jax.lax.pow(float(gmm_wrapper_state.num_received_updates[i]), ANNEALING_EXPONENT))
        new_stepsizes = new_stepsizes.at[i].set(new_stepsize)

    return jnp.stack(new_stepsizes)


@partial(jax.jit, static_argnames=['MIN_STEPSIZE', 'MAX_STEPSIZE', 'STEPSIZE_INC_FACTOR', 'STEPSIZE_DEC_FACTOR'])
def update_component_stepsize_adaptive(gmm_wrapper_state: GMMWrapperState, STEPSIZE_DEC_FACTOR, STEPSIZE_INC_FACTOR, MIN_STEPSIZE, MAX_STEPSIZE) -> chex.Array:

    def update_fn(reward_history, current_stepsize):
        return jax.lax.cond(reward_history[-2] >= reward_history[-1],
                            lambda current_stepsize: jnp.maximum(STEPSIZE_DEC_FACTOR * current_stepsize, MIN_STEPSIZE),
                            lambda current_stepsize: jnp.minimum(STEPSIZE_INC_FACTOR * current_stepsize, MAX_STEPSIZE),
                            current_stepsize)

    return jax.vmap(update_fn)(gmm_wrapper_state.reward_history, gmm_wrapper_state.stepsizes)

#-------------------------------------------------------------------------------------
#Weight Stepsize Adaptation
#-------------------------------------------------------------------------------------
class WeightStepsizeAdaptationState(NamedTuple):
    stepsize: chex.Array
    DECAYING_num_weight_updates: Optional[chex.Array] = None
    IMPROVEMENT_elbo_history: Optional[chex.Array] = None

def setup_weight_stepsize_fn(initial_stepsize, ANNEALING_EXPONENT, \
                             STEPSIZE_DEC_FACTOR, STEPSIZE_INC_FACTOR, MIN_STEPSIZE, MAX_STEPSIZE):
    # if ANNEALING_EXPONENT is not None:
    #     return Partial(init_improvement_based_weight_stepsize_adaptation_state, initial_stepsize=initial_stepsize),
    def init():
            return WeightStepsizeAdaptationState(stepsize=initial_stepsize,
                                                DECAYING_num_weight_updates=jnp.array(0, dtype=jnp.float32),
                                                IMPROVEMENT_elbo_history=jnp.array([jnp.finfo(jnp.float32).min], dtype=jnp.float32))

    @jax.jit
    def update_weight_stepsize_decaying(weight_stepsize_adaption_state: WeightStepsizeAdaptationState):
        return WeightStepsizeAdaptationState(stepsize=initial_stepsize / (1. + jax.lax.pow(weight_stepsize_adaption_state.DECAYING_num_weight_updates, ANNEALING_EXPONENT)),
                                             DECAYING_num_weight_updates=weight_stepsize_adaption_state.DECAYING_num_weight_updates + 1)
    
    @jax.jit
    def update_weight_stepsize_adaptive(weight_stepsize_adaption_state: WeightStepsizeAdaptationState, gmm_wrapper_state: GMMWrapperState):
        elbo = jnp.sum(jnp.exp(gmm_wrapper_state.gmm_state.log_weights) * gmm_wrapper_state.reward_history[:, -1]) - jnp.sum(
            jnp.exp(gmm_wrapper_state.gmm_state.log_weights) * gmm_wrapper_state.gmm_state.log_weights)

        elbo_history = jnp.concatenate((weight_stepsize_adaption_state.IMPROVEMENT_elbo_history, jnp.expand_dims(elbo, 0)), axis=0)

        stepsize = jax.lax.cond(elbo_history[-1] > elbo_history[-2],
                            lambda stepsize: jnp.minimum(STEPSIZE_INC_FACTOR * stepsize, MAX_STEPSIZE),
                            lambda stepsize: jnp.maximum(STEPSIZE_DEC_FACTOR * stepsize, MIN_STEPSIZE), weight_stepsize_adaption_state.stepsize)
        return WeightStepsizeAdaptationState(stepsize=stepsize, IMPROVEMENT_elbo_history=elbo_history)
    
    return init(), update_weight_stepsize_decaying, update_weight_stepsize_adaptive