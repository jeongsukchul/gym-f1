
from functools import partial
from typing import Collection, Tuple, NamedTuple, Callable, Optional
import chex
import jax
import jax.numpy as jnp
from jax._src.tree_util import Partial
from gymkhana.jax_sampler_ppo.gmmvi.gmm_setup import GMMState, GMMWrapperState
from gymkhana.jax_sampler_ppo.gmmvi.utils import reduce_weighted_logsumexp

from gymkhana.jax_sampler_ppo.gmmvi.least_squares import QuadRegression, QuadRegressionState

def get_ng_update_fns(gmm_wrapper, quad_regression_fn, DIM, DIAGONAL_COVS, USE_SELF_NORMALIZED_IMPORTANCE_WEIGHTS, TEMPERATURE: float, INITIAL_REGULARIZER: float):
    def _safe_log_softmax(logits, axis=-1):
        finite = jnp.isfinite(logits)
        any_finite = jnp.any(finite, axis=axis, keepdims=True)

        # ignore -inf entries in the sum
        lse = jax.nn.logsumexp(jnp.where(finite, logits, -jnp.inf), axis=axis, keepdims=True)
        logp = logits - lse

        fallback = jnp.full_like(logp, -jnp.inf)

        return jnp.where(any_finite, logp, fallback)

    def get_more_expected_hessian_and_grad(gmm_wrapper_state: GMMWrapperState,
                                      samples: chex.Array, 
                                      background_densities: chex.Array, target_lnpdfs: chex.Array,target_lnpdf_grads : chex.Array=None,
                                      ) -> Tuple[chex.Array, chex.Array]:

        means = gmm_wrapper_state.gmm_state.means
        chol_covs = gmm_wrapper_state.gmm_state.chol_covs

        model_densities, component_log_densities = jax.vmap(Partial(gmm_wrapper.log_densities_also_individual, gmm_wrapper_state.gmm_state))(samples)
        component_log_densities = jnp.transpose(component_log_densities)
        num_samples = samples.shape[0]
        log_ratios = target_lnpdfs - model_densities
        # target_mask = jnp.isfinite(target_lnpdfs)
        # model_mask = jnp.isfinite(model_densities)
        # jax.debug.print("target isfinite : {x}", x=target_mask)
        # jax.debug.print("model_mask : {x}", x=model_mask)
        gmm_samples=gmm_wrapper.inv_bijector(samples)
        def per_component(component_log_density, l2_regularizer, mean, chol_cov, mask):
            # Importance weights
            log_weights = _safe_log_softmax(component_log_density - background_densities, axis=0) #[BATCH]
            weights= jnp.exp(log_weights)                                                  #[BATCH]
            # Fit quadratic reward around current component mean/cov
            G_hat, g_hat_lin, const_term = quad_regression_fn(
                l2_regularizer,
                gmm_samples, log_ratios, num_samples, weights,
                mean,                                  # center
                chol_cov,
            )  # G_hat: (D,D), g_hat_lin: (D,)

            # gradient in canonical form: G mu - g
            G_hat = jnp.where(mask, G_hat, jnp.eye(DIM))
            g_vec = (G_hat @ mean[:, None] - g_hat_lin[:, None]).reshape((DIM,))
            
            g_vec = jnp.where(mask, g_vec, jnp.zeros(DIM,))
            return G_hat, g_vec

        # Vectorize over components
        expected_hessian, expected_gradient = jax.vmap(per_component)(component_log_densities, gmm_wrapper_state.l2_regularizers, means, chol_covs,
                                                                      gmm_wrapper_state.gmm_state.component_mask)  # Gs: (K,D,D), gs: (K,D)
        mask = (gmm_wrapper_state.gmm_state.component_mask > 0).astype(expected_gradient.dtype)  # [K]
        expected_gradient  = expected_gradient  * mask[:, None]            # [K,D]
        batch_size = expected_gradient.shape[0]
        expected_hessian = expected_hessian * mask[:,None, None]  + jnp.full((batch_size, DIM, DIM) , jnp.eye(DIM)) * (1 - mask[:, None, None])     # [K,D,D]
        
        return expected_hessian, expected_gradient


    def _stable_expectation(log_weights, log_values):
        n = jnp.array(jnp.shape(log_weights)[0], jnp.float32)
        lswe, signs = reduce_weighted_logsumexp(jnp.expand_dims(log_weights, 1) + jnp.log(jnp.abs(log_values)),
                                                w=jnp.sign(log_values), axis=0, return_sign=True)
        return 1 / n * signs * jnp.exp(lswe)

    def _get_expected_gradient_and_hessian_standard_iw(chol_cov, mean, component_log_densities, samples,
                                                        background_mixture_densities, log_ratio_grads):
        log_importance_weights = component_log_densities - background_mixture_densities
        expected_gradient = _stable_expectation(log_importance_weights, log_ratio_grads)

        if DIAGONAL_COVS:
            prec_times_diff = jnp.expand_dims(1 / (chol_cov ** 2), 1) \
                                * jnp.transpose(samples - mean)
            prec_times_diff_times_grad = jnp.transpose(prec_times_diff) * log_ratio_grads
        else:
            prec_times_diff = jax.scipy.linalg.cho_solve(chol_cov, jnp.transpose(samples - mean))
            prec_times_diff_times_grad = \
                jnp.expand_dims(jnp.transpose(prec_times_diff), 1) * jnp.expand_dims(log_ratio_grads, -1)
            log_importance_weights = jnp.expand_dims(log_importance_weights, 1)
        expected_hessian = _stable_expectation(log_importance_weights, prec_times_diff_times_grad)
        return expected_gradient, expected_hessian

    # def _safe_normalization(logits, axis=-1):
        # return jnp.where(weights=)
    def _get_expected_gradient_and_hessian_self_normalized_iw(chol_cov, mean,
                                                                component_log_densities, samples,
                                                                background_mixture_densities, log_ratio_grads):
        log_weights = _safe_log_softmax(component_log_densities - background_mixture_densities, axis=0)
        weights = jnp.exp(log_weights)                                                 #[BATCH]

        importance_weights = weights / jnp.sum(weights, axis=0, keepdims=True)         #[BATCH]

        weighted_gradients = jnp.expand_dims(importance_weights, 1) * log_ratio_grads  #[BATCH, DIM]
        if DIAGONAL_COVS:
            prec_times_diff = jnp.expand_dims(1 / (chol_cov ** 2), 1) * jnp.transpose(samples - mean)
            expected_hessian = jnp.sum(jnp.transpose(prec_times_diff) * weighted_gradients, 0)
        else:
            prec_times_diff = jax.scipy.linalg.cho_solve((chol_cov, True), jnp.transpose(samples - mean))  #[DIM, BATCH]
            expected_hessian = jnp.sum(
                jnp.expand_dims(jnp.transpose(prec_times_diff), 1) * jnp.expand_dims(weighted_gradients, -1), 0)
            
            expected_hessian = 0.5 * (expected_hessian + jnp.transpose(expected_hessian))
        expected_gradient = jnp.sum(weighted_gradients, 0)
        return expected_gradient, expected_hessian

    @jax.jit
    def get_expected_hessian_and_grad(gmm_wrapper_state,
                                samples: chex.Array, background_densities: chex.Array,
                                target_lnpdfs: chex.Array, target_lnpdfs_grads: chex.Array,):

        model_densities, model_densities_grad, component_log_densities = jax.vmap(Partial(gmm_wrapper.log_density_and_grad, gmm_wrapper_state.gmm_state))(samples)
        component_log_densities = jnp.transpose(component_log_densities)
        log_ratios = target_lnpdfs - model_densities
        log_ratio_grads = target_lnpdfs_grads - model_densities_grad
        gmm_samples = gmm_wrapper.inv_bijector(samples)
        def _get_expected_gradient_and_hessian_for_comp(component_log_densities, component_means, component_chol_covs,):
            if USE_SELF_NORMALIZED_IMPORTANCE_WEIGHTS:
                expected_gradient, expected_hessian = \
                    _get_expected_gradient_and_hessian_self_normalized_iw(component_chol_covs, component_means,
                                                                            component_log_densities, gmm_samples, background_densities, log_ratio_grads)
            else:
                expected_gradient, expected_hessian = \
                    _get_expected_gradient_and_hessian_standard_iw(component_chol_covs, component_means,
                                                                    component_log_densities, gmm_samples, background_densities, log_ratio_grads)
            return expected_gradient, expected_hessian

        expected_gradient, expected_hessian =   jax.vmap(_get_expected_gradient_and_hessian_for_comp)\
                (component_log_densities, gmm_wrapper_state.gmm_state.means, gmm_wrapper_state.gmm_state.chol_covs)
        
        mask = (gmm_wrapper_state.gmm_state.component_mask > 0).astype(expected_gradient.dtype)  # [K]
        expected_gradient  = expected_gradient  * mask[:, None]            # [K,D]
        batch_size = expected_gradient.shape[0]
        expected_hessian = expected_hessian * mask[:,None, None]  + jnp.full((batch_size, DIM, DIM) , jnp.eye(DIM)) * (1 - mask[:, None, None])     # [K,D,D]
        
        return -expected_hessian, -expected_gradient


    ##Component Updates
    def _kl(eta: jnp.float32, old_lin_term: chex.Array, old_precision: chex.Array, old_inv_chol: chex.Array,
            reward_lin: chex.Array, reward_quad: chex.Array, kl_const_part: jnp.float32, old_mean: chex.Array,
            eta_in_logspace: bool) -> Tuple[jnp.float32, chex.Array, chex.Array, chex.Array]:

        eta = jax.lax.cond(eta_in_logspace,
                            lambda eta: jnp.exp(eta),
                            lambda eta: eta,
                            eta)

        new_lin = (eta * old_lin_term + reward_lin) / eta
        new_precision = (eta * old_precision + reward_quad) / eta
        if DIAGONAL_COVS:
            chol_precision = jnp.sqrt(new_precision)
            new_mean = 1./new_precision * new_lin
            inv_chol_inv = 1./chol_precision
            diff = old_mean - new_mean
            # this is numerically more stable:
            kl = 0.5 * (jnp.maximum(0., jnp.sum(jnp.log(new_precision / old_precision)
                        + old_precision / new_precision) - DIM)
                        + jnp.sum(jnp.square(old_inv_chol * diff)))
        else:
            chol_precision = jnp.linalg.cholesky(new_precision)

            def true_fn():
                new_mean = old_mean
                inv_chol_inv = old_inv_chol
                new_precision = old_precision
                kl = jnp.finfo(jnp.float32).max

                return kl, new_mean, new_precision, inv_chol_inv

            def false_fn():
                new_mean = jnp.reshape(jax.scipy.linalg.cho_solve((chol_precision, True), jnp.expand_dims(new_lin, 1)),
                                        [-1])
                inv_chol_inv = jnp.linalg.inv(chol_precision)

                new_logdet = -2 * jnp.sum(jnp.log(jnp.diag(chol_precision)))
                trace_term = jnp.square(jnp.linalg.norm(inv_chol_inv @ jnp.transpose(old_inv_chol)))
                diff = old_mean - new_mean
                kl = 0.5 * (kl_const_part - new_logdet + trace_term + jnp.sum(jnp.square(jnp.dot(old_inv_chol, diff))))

                return kl, new_mean, new_precision, inv_chol_inv

            kl, new_mean, new_precision, inv_chol_inv = jax.lax.cond(jnp.any(jnp.isnan(chol_precision)),
                                                                        true_fn,
                                                                        false_fn)

        return kl, new_mean, new_precision, inv_chol_inv

    # always in log_space
    def _bracketing_search(KL_BOUND: jnp.float32, lower_bound: jnp.float32,
                            upper_bound: jnp.float32, old_lin_term: chex.Array, old_precision: chex.Array,
                            old_inv_chol: chex.Array, reward_lin_term: chex.Array, reward_quad_term: chex.Array,
                            kl_const_part: jnp.float32, old_mean: chex.Array) -> Tuple[jnp.float32, jnp.float32]:

        def cond_fn(carry):
            it, lower_bound, upper_bound, eta, kl, _ = carry
            diff = jnp.minimum(jnp.exp(upper_bound) - jnp.exp(eta), jnp.exp(eta) - jnp.exp(lower_bound))
            return (it < 1000) & (diff >= 1e-1) & ((jnp.abs(KL_BOUND - kl) >= 1e-1 * KL_BOUND) | jnp.isnan(kl))

        def body_fn(carry):
            it, lower_bound, upper_bound, eta, _, upper_bound_satisfies_constraint = carry
            kl = _kl(eta, old_lin_term, old_precision, old_inv_chol, reward_lin_term,
                        reward_quad_term, kl_const_part, old_mean, True)[0]

            def true_fn():
                new_lower_bound = new_upper_bound = eta
                return it+1, new_lower_bound, new_upper_bound, eta, kl, upper_bound_satisfies_constraint

            def false_fn():
                new_upper_bound, new_lower_bound, new_upper_bound_satisfies_constraint = jax.lax.cond(KL_BOUND > kl,
                                                                                                        lambda upper_bound, lower_bound, eta: (eta, lower_bound, True),
                                                                                                        lambda upper_bound, lower_bound, eta: (upper_bound, eta, False),
                                                                                                        upper_bound,
                                                                                                        lower_bound,
                                                                                                        eta)
                new_eta = 0.5 * (new_upper_bound + new_lower_bound)
                return it+1, new_lower_bound, new_upper_bound, new_eta, kl, new_upper_bound_satisfies_constraint

            return jax.lax.cond(jnp.abs(KL_BOUND - kl) < 1e-1 * KL_BOUND, true_fn, false_fn)

        _, lower_bound, upper_bound, eta, kl, upper_bound_satisfies_constraint = jax.lax.while_loop(cond_fn, body_fn, init_val=(0, lower_bound, upper_bound, 0.5 * (upper_bound + lower_bound), -1000, False))

        lower_bound = jax.lax.cond(upper_bound_satisfies_constraint,
                                    lambda lower_bound, upper_bound: upper_bound,
                                    lambda lower_bound, upper_bound: lower_bound,
                                    lower_bound,
                                    upper_bound)

        return jnp.exp(lower_bound), jnp.exp(upper_bound)
    @jax.jit
    def apply_ng_update(gmm_wrapper_state: GMMWrapperState, expected_hessians_neg: chex.Array,
                        expected_gradients_neg: chex.Array, stepsizes: chex.Array):

        def _apply_gn_update_per_comp(old_chol, old_mean, last_eta, eps, reward_quad, expected_gradients_neg):
            if DIAGONAL_COVS:
                reward_lin = reward_quad * old_mean - expected_gradients_neg
                old_logdet = 2 * jnp.sum(jnp.log(old_chol))
                old_inv_chol = 1./old_chol
                old_precision = old_inv_chol**2
                old_lin_term = old_precision * old_mean
                kl_const_part = old_logdet - DIM
            else:
                reward_lin = jnp.squeeze(reward_quad @ jnp.expand_dims(old_mean, 1)) - expected_gradients_neg
                old_logdet = 2 * jnp.sum(jnp.log(jnp.diag(old_chol)))
                old_inv_chol = jnp.linalg.inv(old_chol)
                old_precision = jnp.transpose(old_inv_chol) @ old_inv_chol
                old_lin_term = jnp.dot(old_precision, old_mean)
                kl_const_part = old_logdet - DIM

            lower_bound_const, upper_bound_const = jax.lax.cond(last_eta < 0,
                                                                lambda last_eta: (jnp.array(-20.), jnp.array(80.)),
                                                                lambda last_eta: (jnp.maximum(0., jnp.log(last_eta) - 3), jnp.log(last_eta) + 3),
                                                                last_eta)

            new_lower, new_upper = _bracketing_search(eps, lower_bound_const, upper_bound_const,
                                                        old_lin_term, old_precision, old_inv_chol, reward_lin,
                                                        reward_quad, kl_const_part, old_mean)
            eta = jnp.maximum(new_lower, TEMPERATURE)

            def true_lower_equals_upper():
                new_kl, new_mean, _, new_inv_chol_inv = _kl(eta, old_lin_term, old_precision,
                                                            old_inv_chol, reward_lin, reward_quad,
                                                            kl_const_part, old_mean, False)
                if DIAGONAL_COVS:
                    new_cov = jnp.square(new_inv_chol_inv)
                    new_chol = jnp.sqrt(new_cov)
                else:
                    new_cov = jnp.transpose(new_inv_chol_inv) @ new_inv_chol_inv
                    new_chol = jnp.linalg.cholesky(new_cov)

                return jax.lax.cond((new_kl < jnp.finfo(jnp.float32).max) & (~jnp.any(jnp.isnan(new_chol))),
                                    lambda: (True, new_mean, new_chol, new_kl),
                                    lambda: (False, old_mean, old_chol, -1.),   # values will be ignored anyway, if success is false
                                    )

            success, new_mean, new_chol, new_kl = jax.lax.cond(new_lower == new_upper,
                                                                true_lower_equals_upper,
                                                                lambda: (False, old_mean, old_chol, -1.))
            return jax.lax.cond(success,
                                lambda: (new_chol, new_mean, new_kl, True, eta),
                                lambda: (old_chol, old_mean, -1., False, -1.))

        chols, means, kls, successes, etas = jax.vmap(_apply_gn_update_per_comp)(gmm_wrapper_state.gmm_state.chol_covs,
                                                                                    gmm_wrapper_state.gmm_state.means,
                                                                                    gmm_wrapper_state.last_log_etas,
                                                                                    stepsizes,
                                                                                    expected_hessians_neg,
                                                                                    expected_gradients_neg)
        new_gmm_state = gmm_wrapper.replace_components(gmm_wrapper_state.gmm_state, means, chols)
        updated_l2_reg = jnp.where(successes,
                                    jnp.maximum(0.5 * gmm_wrapper_state.l2_regularizers, INITIAL_REGULARIZER),
                                    jnp.minimum(1e-6, 10 * gmm_wrapper_state.l2_regularizers))

        return gmm_wrapper_state._replace(
                                    gmm_state=new_gmm_state,
                                    l2_regularizers=updated_l2_reg,
                                    last_log_etas=etas,
                                    num_received_updates=gmm_wrapper_state.num_received_updates + 1,
                                )

    return apply_ng_update, get_expected_hessian_and_grad, get_more_expected_hessian_and_grad
