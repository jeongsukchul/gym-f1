import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import chex
from typing import List, Optional, Sequence, Tuple
import wandb
import numpy as np
from matplotlib.ticker import MaxNLocator

def reduce_weighted_logsumexp(logx, w=None, axis=None, keep_dims=False, return_sign=False,):
    if w is None:
      lswe = jax.nn.logsumexp(
          logx,
          axis=axis,
          keepdims=keep_dims)

      if return_sign:
        sgn = jnp.ones_like(lswe)
        return lswe, sgn
      return lswe

    log_absw_x = logx + jnp.log(jnp.abs(w))
    max_log_absw_x = jnp.max(
        log_absw_x, axis=axis, keepdims=True,)
    # If the largest element is `-inf` or `inf` then we don't bother subtracting
    # off the max. We do this because otherwise we'd get `inf - inf = NaN`. That
    # this is ok follows from the fact that we're actually free to subtract any
    # value we like, so long as we add it back after taking the `log(sum(...))`.
    max_log_absw_x = jnp.where(
        jnp.isinf(max_log_absw_x),
        jnp.zeros([], max_log_absw_x.dtype),
        max_log_absw_x)
    wx_over_max_absw_x = (jnp.sign(w) * jnp.exp(log_absw_x - max_log_absw_x))
    sum_wx_over_max_absw_x = jnp.sum(
        wx_over_max_absw_x, axis=axis, keepdims=keep_dims)
    if not keep_dims:
      max_log_absw_x = jnp.squeeze(max_log_absw_x, axis)
    sgn = jnp.sign(sum_wx_over_max_absw_x)
    lswe = max_log_absw_x + jnp.log(sgn * sum_wx_over_max_absw_x)
    if return_sign:
      return lswe, sgn
    return lswe

def visualise(log_prob_fn, dr_range_low:chex.Array, dr_range_high : chex.Array, samples: chex.Array = None, eval_samples = None, bijector_log_prob=None, show=False) -> dict:
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot()
    low, high = dr_range_low, dr_range_high
    x, y = jnp.meshgrid(jnp.linspace(low[0]-0.5, high[0]+0.5, 100), jnp.linspace(low[1]-0.5, high[1]+0.5, 100))
    grid = jnp.c_[x.ravel(), y.ravel()]
    pdf_values = jax.vmap(jnp.exp)(log_prob_fn(sample=grid))
    pdf_values = jnp.reshape(pdf_values, x.shape)
    ctf = plt.contourf(x, y, pdf_values, levels=20, cmap='viridis')
    cbar = fig.colorbar(ctf)
    handles= []
    if samples is not None:
      idx = jax.random.choice(jax.random.PRNGKey(0), samples.shape[0], (300,))
      sample_x = samples[idx,0]
      sample_y = samples[idx,1]
      h1 = ax.scatter(sample_x, sample_y, c='r', alpha=0.5, marker='x')
      handles.append(h1)
    if eval_samples is not None:
      idx = jax.random.choice(jax.random.PRNGKey(0), eval_samples.shape[0], (300,))
      sample_x = eval_samples[idx,0]
      sample_y = eval_samples[idx,1]
      h2 = ax.scatter(sample_x, sample_y, c='b', alpha=0.5, marker='x')
      handles.append(h2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
    fig.subplots_adjust(right=0.80)
    if handles:
      ax.legend(
          handles=handles,
          loc="center left",
          bbox_to_anchor=(1.02, 0.5),
          frameon=True,
          borderaxespad=0.0,
      )
    fig2 = None
    if bijector_log_prob is not None:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

        # bijector_log_prob should return log |det J| or something same-shape as logp
        bij_part = bijector_log_prob(grid)              # (10000,)
        bij_part = jnp.reshape(bij_part, x.shape)       # (100,100)

        # model minus bijector term (still jax)
        combined = pdf_values - jax.vmap(jnp.exp)(bij_part)

        combined_np = np.asarray(combined)

        ctf2 = ax2.contourf(x, y, combined_np, levels=20, cmap='viridis')
        cbar2 = fig2.colorbar(ctf2, ax=ax2)
        cbar2.set_label('log p(x) - bijector_term')

        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_xlim(float(low[0]), float(high[0]))
        ax2.set_ylim(float(low[1]), float(high[1]))

        ax2.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
        ax2.minorticks_on()
        ax2.grid(True, which="major", alpha=0.25, linewidth=0.8)
        ax2.grid(True, which="minor", alpha=0.12, linewidth=0.6)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.xlim(-10, 5)
    # plt.ylim(-5, 5)

    # plt.savefig(os.path.join(project_path('./samples/funnel/'), f"{prefix}funnel.pdf"), bbox_inches='tight', pad_inches=0.1)

    # wb = {"figures/model": [wandb.Image(fig)]}

    if show:
        plt.show()

    return fig, fig2


def _subsample_points(
    x: Optional[chex.Array],
    max_points: int,
    key: jax.Array,
) -> Optional[chex.Array]:
    if x is None:
        return None
    n = x.shape[0]
    if n <= max_points:
        return x
    idx = jax.random.choice(key, n, shape=(max_points,), replace=False)
    return x[idx]


def _make_2d_grid(low_i, high_i, low_j, high_j, num_grid):
    xi = jnp.linspace(low_i - 0.5, high_i + 0.5, num_grid)
    xj = jnp.linspace(low_j - 0.5, high_j + 0.5, num_grid)
    X, Y = jnp.meshgrid(xi, xj)
    return X, Y


def _plot_safe_hist(ax, values, *, low: float, high: float, bins: int = 30) -> None:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    hist_low = float(low)
    hist_high = float(high)
    if not (np.isfinite(hist_low) and np.isfinite(hist_high) and hist_low < hist_high):
        hist_low = float(np.min(vals))
        hist_high = float(np.max(vals))

    if not (np.isfinite(hist_low) and np.isfinite(hist_high) and hist_low < hist_high):
        center = float(vals[0])
        eps = max(abs(center) * 1e-6, 1e-6)
        hist_low = center - eps
        hist_high = center + eps

    bin_count = min(int(bins), max(1, int(vals.size)))
    try:
        ax.hist(vals, bins=bin_count, range=(hist_low, hist_high), density=True, alpha=0.8)
    except ValueError:
        ax.axvline(float(np.mean(vals)), alpha=0.8)


def _evaluate_pair_marginal_mc(
    log_prob_fn,
    X: chex.Array,            # (G, G)
    Y: chex.Array,            # (G, G)
    dim_x: int,
    dim_y: int,
    context_samples: chex.Array,   # (M, D)
) -> np.ndarray:
    """
    Approximate pairwise marginal by averaging p(x_i, x_j, x_-ij)
    over Monte Carlo context samples for the remaining dimensions.
    """
    G1, G2 = X.shape
    M, D = context_samples.shape
    N = G1 * G2

    xy = jnp.stack([X.ravel(), Y.ravel()], axis=-1)  # (N, 2)

    # Repeat each grid point for each context sample
    full = jnp.repeat(context_samples[None, :, :], N, axis=0)   # (N, M, D)
    full = full.at[:, :, dim_x].set(xy[:, 0][:, None])
    full = full.at[:, :, dim_y].set(xy[:, 1][:, None])

    full_flat = full.reshape(N * M, D)  # (N*M, D)

    logp = log_prob_fn(sample=full_flat)   # (N*M,)
    p = jnp.exp(logp).reshape(N, M)        # (N, M)

    # Monte Carlo average over other dimensions
    p_avg = jnp.mean(p, axis=1)            # (N,)

    return np.asarray(p_avg.reshape(G1, G2))


def visualise_pairwise_2d_marginal(
    log_prob_fn,
    dr_range_low: chex.Array,
    dr_range_high: chex.Array,
    samples: Optional[chex.Array] = None,
    eval_samples: Optional[chex.Array] = None,
    dims: Optional[Sequence[int]] = None,
    max_scatter_points: int = 300,
    marginal_mc_samples: int = 64,
    num_grid: int = 80,
    max_dims: Optional[int] = 8,
    dpi: int = 100,
    show: bool = False,
):
    """
    Pairwise 2D visualization using Monte Carlo averaging over non-plotted dimensions.

    If D=2, this reduces to a normal 2D contour.
    If D>2, each subplot approximates the pairwise marginal by averaging over
    the remaining dimensions using `samples` or `eval_samples`.
    """
    plt.close("all")

    dr_range_low = jnp.asarray(dr_range_low)
    dr_range_high = jnp.asarray(dr_range_high)
    D = dr_range_low.shape[0]

    if dims is None:
        if max_dims is not None and D > max_dims:
            dims = tuple(np.linspace(0, D - 1, max_dims, dtype=int).tolist())
        else:
            dims = tuple(range(D))
    else:
        dims = tuple(dims)

    if len(dims) < 2:
        raise ValueError("Need at least 2 dimensions.")

    # Context samples used for marginalization
    if samples is not None:
        context_source = samples
    elif eval_samples is not None:
        context_source = eval_samples
    else:
        raise ValueError(
            "For averaged pairwise plots with D>2, provide `samples` or `eval_samples` "
            "to use as Monte Carlo context samples."
        )

    key_ctx = jax.random.PRNGKey(123)
    context_samples = _subsample_points(context_source, marginal_mc_samples, key_ctx)

    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(1)
    samples_plot = _subsample_points(samples, max_scatter_points, key1) if samples is not None else None
    eval_samples_plot = _subsample_points(eval_samples, max_scatter_points, key2) if eval_samples is not None else None

    K = len(dims)
    fig, axes = plt.subplots(K, K, figsize=(3.5 * K, 3.5 * K), dpi=dpi, squeeze=False)
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    if K < D:
        fig.suptitle(f"Pairwise marginals for dims {dims} of {D}")

    mappable = None

    for row, dim_i in enumerate(dims):
        for col, dim_j in enumerate(dims):
            ax = axes[row, col]

            if row < col:
                ax.axis("off")
                continue

            if row == col:
                # Simple histogram on diagonal
                if context_source is not None:
                    _plot_safe_hist(
                        ax,
                        context_source[:, dim_i],
                        low=float(dr_range_low[dim_i]),
                        high=float(dr_range_high[dim_i]),
                    )
                ax.set_xlim(float(dr_range_low[dim_i]), float(dr_range_high[dim_i]))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
                if row == K - 1:
                    ax.set_xlabel(f"dim {dim_i}")
                else:
                    ax.set_xticklabels([])
                ax.set_yticklabels([])
                continue

            X, Y = _make_2d_grid(
                float(dr_range_low[dim_j]),
                float(dr_range_high[dim_j]),
                float(dr_range_low[dim_i]),
                float(dr_range_high[dim_i]),
                num_grid,
            )

            Z = _evaluate_pair_marginal_mc(
                log_prob_fn=log_prob_fn,
                X=X,
                Y=Y,
                dim_x=dim_j,   # x-axis
                dim_y=dim_i,   # y-axis
                context_samples=context_samples,
            )

            ctf = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
            mappable = ctf

            if samples_plot is not None:
                ax.scatter(
                    np.asarray(samples_plot[:, dim_j]),
                    np.asarray(samples_plot[:, dim_i]),
                    c="r",
                    alpha=0.35,
                    marker="x",
                    s=18,
                    label="samples" if (row == 1 and col == 0) else None,
                )

            if eval_samples_plot is not None:
                ax.scatter(
                    np.asarray(eval_samples_plot[:, dim_j]),
                    np.asarray(eval_samples_plot[:, dim_i]),
                    c="b",
                    alpha=0.35,
                    marker="x",
                    s=18,
                    label="eval_samples" if (row == 1 and col == 0) else None,
                )

            ax.set_xlim(float(dr_range_low[dim_j]), float(dr_range_high[dim_j]))
            ax.set_ylim(float(dr_range_low[dim_i]), float(dr_range_high[dim_i]))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

            if row == K - 1:
                ax.set_xlabel(f"dim {dim_j}")
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(f"dim {dim_i}")
            else:
                ax.set_yticklabels([])

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, shrink=0.9, pad=0.02)
        cbar.set_label("MC-averaged density")

    handles, labels = axes[min(1, K - 1), 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=True)

    if show:
        plt.show()

    return fig, None
