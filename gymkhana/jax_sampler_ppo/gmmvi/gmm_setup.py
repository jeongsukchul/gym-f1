import functools
from typing import NamedTuple, Callable, Tuple
import chex
import jax.numpy as jnp
import jax



class GMM(NamedTuple):
    init_gmm_state: Callable
    sample: Callable
    sample_from_components_no_shuffle: Callable
    sample_from_components_shuffle: Callable
    add_component: Callable
    remove_component: Callable
    replace_components: Callable
    average_entropy: Callable
    replace_weights: Callable
    component_log_densities: Callable
    log_densities_also_individual: Callable
    log_density: Callable
    log_density_and_grad: Callable
    bijector: Callable
    inv_bijector: Callable
    bijector_log_prob : Callable
class GMMState(NamedTuple):
    log_weights: chex.Array
    means: chex.Array
    chol_covs: chex.Array
    num_components: int
    component_mask : chex.Array

class GMMWrapper(NamedTuple):
    init_gmm_wrapper_state: Callable
    add_component: Callable
    remove_component: Callable
    replace_components: Callable
    store_rewards: Callable
    update_stepsizes: Callable
    replace_weights: Callable
    log_density: Callable
    average_entropy: Callable
    log_densities_also_individual: Callable
    component_log_densities: Callable
    sample_from_components_no_shuffle: Callable
    sample_from_components_shuffle: Callable
    log_density_and_grad: Callable
    sample: Callable
    bijector: Callable
    inv_bijector:Callable
    bijector_log_prob : Callable
class GMMWrapperState(NamedTuple):
    gmm_state: GMMState
    l2_regularizers: chex.Array
    last_log_etas: chex.Array
    num_received_updates: chex.Array
    stepsizes: chex.ArrayTree
    reward_history: chex.Array
    weight_history: chex.Array
    max_component_id: chex.Array
    adding_thresholds: chex.Array



def setup_gmm_wrapper(gmm: GMM, MAX_COMPONENTS, INITIAL_STEPSIZE, INITIAL_REGULARIZER, MAX_REWARD_HISTORY_LENGTH, INITIAL_LAST_ETA=-1):
    def init_gmm_wrapper_state(gmm_state: GMMState):
        return GMMWrapperState(gmm_state=gmm_state,
                               l2_regularizers=INITIAL_REGULARIZER * jnp.ones(MAX_COMPONENTS),
                               last_log_etas=INITIAL_LAST_ETA * jnp.ones(MAX_COMPONENTS),
                               num_received_updates=jnp.zeros(MAX_COMPONENTS),
                               stepsizes=INITIAL_STEPSIZE * jnp.ones(MAX_COMPONENTS),
                               reward_history=-jnp.inf * jnp.ones(
                                   (MAX_COMPONENTS, MAX_REWARD_HISTORY_LENGTH)),
                               weight_history=-jnp.inf * jnp.ones(
                                   (MAX_COMPONENTS, MAX_REWARD_HISTORY_LENGTH)),
                               max_component_id=jnp.max(jnp.arange(MAX_COMPONENTS)),
                               adding_thresholds=-jnp.ones(MAX_COMPONENTS))

    def add_component(gmm_wrapper_state: GMMWrapperState, initial_weight: jnp.float32, initial_mean: chex.Array,
                      initial_cov: chex.Array, adding_threshold: chex.Array):
        mask = gmm_wrapper_state.gmm_state.component_mask
        idx = jnp.nonzero(mask == 0, size=1, fill_value=MAX_COMPONENTS)[0][0]
        return GMMWrapperState(gmm_state=gmm.add_component(gmm_wrapper_state.gmm_state, idx, initial_weight, initial_mean, initial_cov),
                               l2_regularizers=gmm_wrapper_state.l2_regularizers.at[idx].set(INITIAL_REGULARIZER),
                               last_log_etas=gmm_wrapper_state.last_log_etas.at[idx].set(INITIAL_LAST_ETA),
                               num_received_updates=gmm_wrapper_state.num_received_updates.at[idx].set(0),
                               stepsizes=gmm_wrapper_state.stepsizes.at[idx].set(INITIAL_STEPSIZE),
                               reward_history=gmm_wrapper_state.reward_history.at[idx].set(jnp.ones(MAX_REWARD_HISTORY_LENGTH)),
                               weight_history=gmm_wrapper_state.weight_history.at[idx].set(jnp.ones(MAX_REWARD_HISTORY_LENGTH)*initial_weight),
                               max_component_id=idx,
                               adding_thresholds=gmm_wrapper_state.adding_thresholds.at[idx].set(adding_threshold))

    def remove_component(gmm_wrapper_state: GMMWrapperState, bad_mask: jnp.ndarray):
        return gmm_wrapper_state._replace(
                gmm_state=gmm.remove_component(gmm_wrapper_state.gmm_state, bad_mask),
            )

    def update_weights(gmm_wrapper_state: GMMWrapperState, new_log_weights: chex.Array):
        return gmm_wrapper_state._replace(gmm_state=gmm.replace_weights(gmm_wrapper_state.gmm_state, new_log_weights),
                               weight_history=jnp.concatenate((gmm_wrapper_state.weight_history[:, 1:],
                                                               jnp.expand_dims(jnp.exp(gmm_wrapper_state.gmm_state.log_weights), 1)), axis=1),)

    def update_rewards(gmm_wrapper_state: GMMWrapperState, rewards: chex.Array):
        return gmm_wrapper_state._replace(
                               reward_history=jnp.concatenate((gmm_wrapper_state.reward_history[:, 1:],
                                                               jnp.expand_dims(rewards, 1)), axis=1),
                               )

    def update_stepsizes(gmm_wrapper_state: GMMWrapperState, new_stepsizes: chex.Array):
        return gmm_wrapper_state._replace(
                               stepsizes=new_stepsizes,
                               )

    return GMMWrapper(init_gmm_wrapper_state=init_gmm_wrapper_state,
                      add_component=add_component,
                      remove_component=remove_component,
                      store_rewards=update_rewards,
                      update_stepsizes=update_stepsizes,
                      replace_weights=update_weights,
                      log_density=gmm.log_density,
                      average_entropy=gmm.average_entropy,
                      component_log_densities=gmm.component_log_densities,
                      log_densities_also_individual=gmm.log_densities_also_individual,
                      replace_components=gmm.replace_components,
                      sample_from_components_no_shuffle=gmm.sample_from_components_no_shuffle,
                      sample_from_components_shuffle=gmm.sample_from_components_shuffle,
                      log_density_and_grad=gmm.log_density_and_grad,
                      sample=gmm.sample,
                      bijector=gmm.bijector,
                      inv_bijector=gmm.inv_bijector,
                      bijector_log_prob=gmm.bijector_log_prob)


def _setup_initial_mixture_params(NUM_DIM, key, diagonal_covs, MAX_COMPONENTS, num_initial_components, prior_mean, prior_scale,
                                  initial_cov=None):

    if jnp.isscalar(prior_mean):
        prior_mean = prior_mean * jnp.ones(NUM_DIM)

    if jnp.isscalar(prior_scale):
        prior_scale = prior_scale * jnp.ones(NUM_DIM)
    prior = jnp.array(prior_scale) ** 2

    weights = jnp.ones(num_initial_components, dtype=jnp.float32) / num_initial_components
    weights = jnp.concatenate([weights,  jnp.zeros(MAX_COMPONENTS-num_initial_components)])
    means = jnp.zeros((MAX_COMPONENTS, NUM_DIM), dtype=jnp.float32)

    if diagonal_covs:
        if initial_cov is None:
            initial_cov = prior  # use the same initial covariance that was used for sampling the mean
        else:
            initial_cov = initial_cov * jnp.ones(NUM_DIM)

        covs = jnp.full((num_initial_components, NUM_DIM), initial_cov, dtype=jnp.float32)
        for i in range(0, num_initial_components):
            key, subkey = jax.random.split(key)
            if num_initial_components == 1:
                means = means.at[i].set(prior_mean)
            else:
                rand_samples = jax.random.normal(subkey, (NUM_DIM,))
                means = means.at[i].set(prior_mean + jnp.sqrt(prior) * rand_samples)

    else:
        prior = jnp.diag(prior)
        if initial_cov is None:
            initial_cov = prior  # use the same initial covariance that was used for sampling the mean
        else:
            initial_cov = initial_cov * jnp.eye(NUM_DIM)

        covs = jnp.full((num_initial_components, NUM_DIM, NUM_DIM), initial_cov, dtype=jnp.float32)
        for i in range(0, num_initial_components):
            key, subkey = jax.random.split(key)
            if num_initial_components == 1:
                means = means.at[i].set(prior_mean)
            else:
                rand_samples = jax.random.normal(subkey, (NUM_DIM,))
                means = means.at[i].set(prior_mean + jnp.linalg.cholesky(prior) @ rand_samples)

    if diagonal_covs:
        chol_covs = jnp.stack([jnp.sqrt(cov) for cov in covs])
        chol_covs = jnp.concatenate([chol_covs, jnp.ones((MAX_COMPONENTS-num_initial_components, NUM_DIM))], axis=0)
    else:
        chol_covs = jnp.stack([jnp.linalg.cholesky(cov) for cov in covs])
        chol_covs = jnp.concatenate([chol_covs, jnp.full((MAX_COMPONENTS-num_initial_components, NUM_DIM, NUM_DIM), jnp.eye(NUM_DIM), dtype=jnp.float32)], axis=0)
    return weights, means, chol_covs


def setup_sample_fn(sample_from_component_fn: Callable):
    @functools.partial(jax.jit, static_argnames=("num_samples",))
    def sample(gmm_state, seed: chex.PRNGKey, num_samples: int) -> Tuple[chex.Array, chex.Array]:
        key_comp, key_draw = jax.random.split(seed, 2)

        # K must come from a static shape (NOT gmm_state.num_components)
        K = gmm_state.log_weights.shape[0]

        mask = gmm_state.component_mask.astype(bool)          # (K,)
        valid = mask & jnp.isfinite(gmm_state.log_weights)

        logits = jnp.where(valid, gmm_state.log_weights, -jnp.inf)

        # avoid "all -inf" -> NaNs
        logits = jax.lax.cond(
            jnp.any(valid),
            lambda l: l,
            lambda l: jnp.zeros((K,), dtype=l.dtype),  # uniform fallback
            logits,
        )

        # sample component indices ~ softmax(logits)
        comps = jax.random.categorical(key_comp, logits, shape=(num_samples,)).astype(jnp.int32)

        keys = jax.random.split(key_draw, num_samples)

        def draw_one(k, comp_idx):
            x = sample_from_component_fn(gmm_state, comp_idx, 1, k)  # (1, *event_shape)
            return x.squeeze()#x[0]  # (*event_shape,)

        samples = jax.vmap(draw_one)(keys, comps)  # (num_samples, *event_shape)
        return samples, comps

    return sample

def setup_sample_from_components_shuffle_fn(sample_from_component_fn: Callable):
    def sample_from_component(gmm_state: GMMState, index: int, num_samples: int, seed: chex.Array) -> chex.Array:

        return jnp.transpose(jnp.expand_dims(gmm_state.means[index], axis=-1)
                             + gmm_state.chol_covs[index] @ jax.random.normal(key=seed, shape=(2, num_samples)))
    @jax.jit
    def sample_from_components_shuffle(gmm_state: GMMState, mapping : jnp.ndarray,
                                          seed: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        
        # Make per-sample keys
        keys = jax.random.split(seed, mapping.shape[0])

        def sample_one(key, comp_idx):
            # sample a single point from component `comp_idx`
            # Ensure sample_from_component_fn returns shape (N, D)
            x = sample_from_component_fn(gmm_state, comp_idx, 1, key)
            return x.squeeze()# x[0]  # (D,)

        samples = jax.vmap(sample_one)(keys, mapping)  # (TOTAL_SAMPLES, D)
        return samples, None

    return sample_from_components_shuffle


def setup_sample_from_components_no_shuffle_fn(sample_from_component_fn: Callable):
    

    def sample_from_components_no_shuffle(gmm_state: GMMState, DESIRED_SAMPLES, num_components,
                                          seed: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        mapping = jnp.repeat(jnp.arange(num_components), DESIRED_SAMPLES)
        samples = jax.vmap(sample_from_component_fn, in_axes=(None, 0, None, 0))(gmm_state,
                                                                                 jnp.arange(num_components),
                                                                                 DESIRED_SAMPLES,
                                                                                 jax.random.split(seed, (num_components,)))

        return jnp.vstack(samples), mapping

    return sample_from_components_no_shuffle


def _normalize_weights(log_w, mask):
    # Put -inf on inactive comps so they don’t affect logsumexp
    log_w = jnp.where(mask > 0, log_w, -jnp.inf)
    logZ = jax.nn.logsumexp(log_w)
    return jnp.where(mask > 0, log_w - logZ, -jnp.inf)

# def _normalize_weights(new_log_weights: chex.Array):
#     return new_log_weights - jax.nn.logsumexp(new_log_weights)


def replace_weights(gmm_state: GMMState, new_log_weights: chex.Array):
    return gmm_state._replace(log_weights=_normalize_weights(new_log_weights, gmm_state.component_mask))


def remove_component(gmm_state: GMMState, bad_mask, MAX_COMPONENTS, DIM):
    mask = gmm_state.component_mask * (1.-bad_mask)
    log_weights = jnp.where(mask>0, gmm_state.log_weights, -jnp.inf)
    means = gmm_state.means * mask[:,None]
    chols = gmm_state.chol_covs * mask[:,None, None] + \
        jnp.full((MAX_COMPONENTS, DIM, DIM),jnp.eye(DIM), dtype=jnp.float32) * (1- mask[:, None, None])
    return gmm_state._replace(log_weights=_normalize_weights(log_weights, mask),
                    means = means,
                    chol_covs = chols,
                    num_components=jnp.sum(mask, dtype=jnp.int32),
                    component_mask=mask)


def replace_components(gmm_state: GMMState, new_means: chex.Array, new_chols: chex.Array) -> GMMState:
    new_means = jnp.stack(new_means, axis=0)
    new_chols = jnp.stack(new_chols, axis=0)
    return gmm_state._replace(
                    means=new_means,
                    chol_covs=new_chols,)

def setup_get_average_entropy_fn(gaussian_entropy_fn: Callable):
    def get_average_entropy(gmm_state: GMMState) -> jnp.float32:
        gaussian_entropies = jax.vmap(gaussian_entropy_fn)(gmm_state.chol_covs)
        return jnp.sum(jnp.exp(gmm_state.log_weights) * gaussian_entropies)
    return get_average_entropy


def setup_log_density_fn(component_log_densities_fn: Callable, inv_bijector, bijector_log_prob):
    def log_density(gmm_state: GMMState, sample: chex.Array) -> chex.Array:
        bounded_sample = inv_bijector(sample)
        log_densities = component_log_densities_fn(gmm_state, bounded_sample)
        log_densities = jnp.where(gmm_state.component_mask > 0, log_densities, -jnp.inf)
        weighted_densities = log_densities + gmm_state.log_weights
        return jax.nn.logsumexp(weighted_densities) + bijector_log_prob(sample)

    return log_density


def setup_log_density_and_grad_fn(component_log_densities_fn: Callable, inv_bijector, bijector_log_prob):
    def log_density_and_grad(gmm_state: GMMState, sample: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        def compute_log_densities(sample):
            bounded_sample = inv_bijector(sample)
            log_densities = component_log_densities_fn(gmm_state, bounded_sample)
            log_densities = jnp.where(gmm_state.component_mask > 0, log_densities, -jnp.inf)
            weighted_densities = log_densities + gmm_state.log_weights

            x = jax.nn.logsumexp(weighted_densities, axis=0) + bijector_log_prob(sample)

            return x, log_densities
        (log_densities, log_component_densities), log_densities_grad = jax.value_and_grad(compute_log_densities,
                                                                                          has_aux=True)(sample)

        return log_densities, log_densities_grad, log_component_densities

    return log_density_and_grad


def setup_log_densities_also_individual_fn(component_log_densities_fn: Callable, inv_bijector, bijector_log_prob):
    def log_densities_also_individual(gmm_state: GMMState, sample: chex.Array) -> Tuple[chex.Array, chex.Array]:
        bounded_sample = inv_bijector(sample)
        log_densities = component_log_densities_fn(gmm_state, bounded_sample)
        log_densities = jnp.where(gmm_state.component_mask > 0, log_densities, -jnp.inf)
        weighted_densities = log_densities + gmm_state.log_weights 
        return jax.nn.logsumexp(weighted_densities) + bijector_log_prob(sample), log_densities

    return log_densities_also_individual


# def setup_diagonal_gmm(DIM) -> GMM:
#     def init_diagonal_gmm_state(seed, num_initial_components, prior_mean, prior_scale, diagonal_covs, initial_cov=None):
#         weights, means, chol_covs = _setup_initial_mixture_params(DIM, seed, diagonal_covs, num_initial_components,
#                                                                   prior_mean, prior_scale, initial_cov)

#         return GMMState(log_weights=_normalize_weights(jnp.log(weights)),
#                         means=means,
#                         chol_covs=chol_covs,
#                         num_components=num_initial_components)

#     def sample_from_component(gmm_state: GMMState, index: int, num_samples: int, seed: chex.PRNGKey) -> chex.Array:

#         samples = jnp.transpose(jnp.expand_dims(gmm_state.means[index], 1) + jnp.expand_dims(gmm_state.chol_covs[index], 1)
#                                 * jax.random.normal(seed, (DIM, num_samples)))
#         return samples

#     def component_log_densities(gmm_state: GMMState, samples: chex.Array) -> chex.Array:
#         diffs = jnp.expand_dims(samples, 0) - gmm_state.means
#         inv_chol = 1. / gmm_state.chol_covs  # Inverse of diagonal elements
#         mahalas = -0.5 * jnp.sum(jnp.square(diffs * inv_chol), axis=-1)
#         const_parts = -jnp.sum(jnp.log(gmm_state.chol_covs), axis=1) - 0.5 * DIM * jnp.log(2 * jnp.pi)
#         log_pdfs = mahalas + const_parts
#         return log_pdfs

#     def gaussian_entropy(chol: chex.Array) -> chex.Array:
#         return 0.5 * DIM * (jnp.log(2 * jnp.pi) + 1) + jnp.sum(jnp.log(chol))

#     def add_component(gmm_state: GMMState, idx : int, initial_weight: chex.Array, initial_mean: chex.Array,
#                       initial_cov: chex.Array):
#         return GMMState(
#             log_weights=_normalize_weights(gmm_state.log_weights.at[idx].set(jnp.log(initial_weight)))
#         )
#         # return GMMState(log_weights=_normalize_weights(jnp.concatenate((gmm_state.log_weights,
#         #                                                                 jnp.expand_dims(jnp.log(initial_weight),
#         #                                                                                 axis=0)),
#         #                                                                axis=0)),
#         #                 means=jnp.concatenate((gmm_state.means, jnp.expand_dims(initial_mean, axis=0)), axis=0),
#         #                 chol_covs=jnp.concatenate(
#         #                     (gmm_state.chol_covs, jnp.expand_dims(jnp.sqrt(initial_cov), axis=0)), axis=0),
#         #                 num_components=gmm_state.num_components + 1)

#     return GMM(init_gmm_state=init_diagonal_gmm_state,
#                sample=setup_sample_fn(sample_from_component),
#                sample_from_components_no_shuffle=setup_sample_from_components_no_shuffle_fn(sample_from_component),
#                sample_from_components_shuffle=setup_sample_from_components_shuffle_fn(sample_from_component),
#                add_component=add_component,
#                remove_component=remove_component,
#                replace_components=replace_components,
#                average_entropy=setup_get_average_entropy_fn(gaussian_entropy),
#                replace_weights=replace_weights,
#                component_log_densities=component_log_densities,
#                log_density=setup_log_density_fn(component_log_densities),
#                log_densities_also_individual=setup_log_densities_also_individual_fn(component_log_densities),
#                log_density_and_grad=setup_log_density_and_grad_fn(component_log_densities))


def setup_full_cov_gmm(DIM, MAX_COMPONENTS, bound_info=None) -> GMM:
    if bound_info is not None:
        eps = 1e-6
        low, high = bound_info
        bijector = lambda x:  jnp.tanh(x)*(high-low)/2 + (low+high)/2
        inv_bijector = lambda x: 0.5* (jnp.log1p(jnp.clip((2*x-(low+high))/(high-low), -1. + eps , 1.-eps))- jnp.log1p(-jnp.clip((2*x-(low+high))/(high-low), -1. + eps , 1.-eps)))
        # bijector_log_prob = lambda x : jnp.log(2 * jnp.ones_like(low)).sum(-1) -jnp.log(high-low).sum(-1)-jnp.log(1- ((2*x-(low+high))/(high-low))**2).sum(-1)
        bijector_log_prob = lambda x : jnp.sum(jnp.log(2.0) - jnp.log(high - low)) \
            -jnp.log1p(-jnp.clip((2*x-(low+high))/(high-low), -1.+eps, 1.-eps)**2).sum(-1)
    else:
        bijector = lambda x: x
        inv_bijector = lambda x: x
        bijector_log_prob = lambda x : 0
    def init_full_cov_gmm_state(seed, num_initial_components, prior_mean, prior_scale, diagonal_covs, initial_cov=None):
        weights, means, chol_covs = _setup_initial_mixture_params(DIM, seed, diagonal_covs, MAX_COMPONENTS, num_initial_components,
                                                                  prior_mean, prior_scale, initial_cov)
        mask = jnp.concatenate([jnp.ones(num_initial_components), jnp.zeros(MAX_COMPONENTS-num_initial_components)])
        return GMMState(log_weights=_normalize_weights(jnp.log(weights), mask),
                        means=means,
                        chol_covs=chol_covs,
                        num_components=num_initial_components,
                        component_mask=mask,)

    def sample_from_component(gmm_state: GMMState, index: int, num_samples: int, seed: chex.Array) -> chex.Array:
        Z = jax.random.normal(seed, shape=(DIM, num_samples))
        return (gmm_state.means[index][:, None] + gmm_state.chol_covs[index] @Z).T
    def sample_from_component_bijected(gmm_state: GMMState, index: int, num_samples: int, seed: chex.Array) -> chex.Array:
        Z = jax.random.normal(seed, shape=(DIM, num_samples))
        return bijector((gmm_state.means[index][:, None] + gmm_state.chol_covs[index] @Z).T)
    def component_log_densities(gmm_state: GMMState, sample: chex.Array) -> chex.Array:
        mask = gmm_state.component_mask
        chol_safe = gmm_state.chol_covs * mask[:, None, None] + jnp.eye(DIM)[None, :, :] * (1.0 - mask[:, None, None])

        diffs = jnp.expand_dims(sample, 0) - gmm_state.means 
        sqrts = jax.scipy.linalg.solve_triangular(chol_safe, diffs, lower=True)
        mahalas = - 0.5 * jnp.sum(sqrts * sqrts, axis=1)
        const_parts = - 0.5 * jnp.sum(jnp.log(jnp.square(jnp.diagonal(chol_safe, axis1=1, axis2=2))),
                                      axis=1) - 0.5 * DIM * jnp.log(2 * jnp.pi)
        return (mahalas + const_parts) * mask

    def gaussian_entropy(chol: chex.Array) -> chex.Array:
        ent =  0.5 * DIM * (jnp.log(2 * jnp.pi) + 1) + jnp.sum(jnp.log(jnp.diag(chol)))
        return ent

    def add_component(gmm_state: GMMState, idx:int, initial_weight: chex.Array, initial_mean: chex.Array,
                      initial_cov: chex.Array):
        mask=gmm_state.component_mask.at[idx].set(1)
        return GMMState(
            log_weights=_normalize_weights(gmm_state.log_weights.at[idx].set(jnp.log(initial_weight)), mask),
            means=gmm_state.means.at[idx].set(initial_mean),
            chol_covs=gmm_state.chol_covs.at[idx].set(jnp.linalg.cholesky(initial_cov)),
            num_components=gmm_state.num_components +1,
            component_mask=mask,
        )
        # return GMMState(log_weights=_normalize_weights(jnp.concatenate((gmm_state.log_weights,
        #                                                                 jnp.expand_dims(jnp.log(initial_weight),
        #                                                                                 axis=0)),
        #                                                                axis=0)),
        #                 means=jnp.concatenate((gmm_state.means, jnp.expand_dims(initial_mean, axis=0)), axis=0),
        #                 chol_covs=jnp.concatenate(
        #                     (gmm_state.chol_covs, jnp.expand_dims(jnp.linalg.cholesky(initial_cov), axis=0)), axis=0),
        #                 num_components=gmm_state.num_components + 1)

    return GMM(init_gmm_state=init_full_cov_gmm_state,
               sample=setup_sample_fn(sample_from_component_bijected),
               sample_from_components_no_shuffle=setup_sample_from_components_no_shuffle_fn(sample_from_component_bijected),
               sample_from_components_shuffle=setup_sample_from_components_shuffle_fn(sample_from_component_bijected),
               add_component=add_component,
               remove_component=functools.partial(remove_component, MAX_COMPONENTS=MAX_COMPONENTS, DIM=DIM),
               replace_components=replace_components,
               average_entropy=setup_get_average_entropy_fn(gaussian_entropy),
               replace_weights=replace_weights,
               component_log_densities=component_log_densities,
               log_density=setup_log_density_fn(component_log_densities, \
                                                inv_bijector=inv_bijector, bijector_log_prob=bijector_log_prob),
               log_densities_also_individual=setup_log_densities_also_individual_fn(component_log_densities, \
                                                inv_bijector=inv_bijector, bijector_log_prob=bijector_log_prob),
               log_density_and_grad=setup_log_density_and_grad_fn(component_log_densities, \
                                                inv_bijector=inv_bijector, bijector_log_prob=bijector_log_prob),
               bijector=bijector,
               inv_bijector=inv_bijector,
               bijector_log_prob=bijector_log_prob
            )       
