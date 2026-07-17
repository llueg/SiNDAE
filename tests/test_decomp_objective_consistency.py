"""Regression tests for the decomposition objective/gradient consistency fixes.

These pin two invariants that were silently violated before:
  * the JAX objective differentiated for the KKT right-hand side must equal the
    Pyomo objective that is reported and minimised (sum / chi-squared convention);
  * the subsampled gradient must be an unbiased estimator of the full gradient.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pyomo.environ as pyo

from sindae.algorithms.model_builder_utils import build_data_fit_expr, NORM_OBS_NAME
from sindae.algorithms.decomp.train import make_batch_obj_fn
from sindae.algorithms.decomp.kkt_utils import eval_del_obj_del_rho


def _synthetic_model(n_traj, obs_dim, times, targets):
    m = pyo.ConcreteModel()
    m.trajectories = pyo.Block(range(n_traj))
    for ii in range(n_traj):
        blk = m.trajectories[ii]
        blk.tset = pyo.Set(initialize=times, ordered=True)
        blk.jset = pyo.RangeSet(0, obs_dim - 1)
        setattr(blk, NORM_OBS_NAME, pyo.Var(blk.tset, blk.jset, initialize=0.0))
    m._traj_norm_target = targets
    return m


def test_jax_objective_matches_pyomo_objective_multi_trajectory():
    # With more than one trajectory the old flat-mean JAX objective was 1/N of the
    # sum-of-per-trajectory-means Pyomo objective; both are now the plain sum.
    n_traj, obs_dim, n_t = 3, 2, 4
    times = [0.0, 1.0, 2.0, 3.0]
    rng = np.random.default_rng(0)
    targets = [rng.normal(size=(n_t, obs_dim)) for _ in range(n_traj)]
    m = _synthetic_model(n_traj, obs_dim, times, targets)
    m.obj = pyo.Objective(
        expr=build_data_fit_expr(m, n_traj, [times] * n_traj, targets, obs_dim,
                                 reduction="sum")
    )

    # Arbitrary (non-uniform) residuals so a factor-of-N slip cannot hide.
    for ii in range(n_traj):
        no = getattr(m.trajectories[ii], NORM_OBS_NAME)
        for ti, t in enumerate(times):
            for j in range(obs_dim):
                no[t, j].set_value(float(targets[ii][ti, j]) + 0.1 * (ii + 1) + 0.01 * ti)

    pyomo_val = float(pyo.value(m.obj))
    stacked = np.vstack([
        np.array([[pyo.value(getattr(m.trajectories[ii], NORM_OBS_NAME)[t, j])
                   for j in range(obs_dim)] for t in times])
        for ii in range(n_traj)
    ])
    zeros = np.zeros_like(stacked)
    jax_val = float(make_batch_obj_fn(m)(stacked, zeros, zeros, 1.0))
    assert np.isclose(pyomo_val, jax_val), (pyomo_val, jax_val)


def test_subsampled_gradient_is_unbiased():
    n_pts = 100
    xz = np.zeros((n_pts, 1))
    grad_fn = lambda x, sp, sn: (jnp.ones((n_pts, 1)),
                                 jnp.zeros((n_pts, 1)), jnp.zeros((n_pts, 1)))
    xi = np.arange(n_pts).reshape(n_pts, 1)
    spi = np.arange(n_pts, 2 * n_pts).reshape(n_pts, 1)
    sni = np.arange(2 * n_pts, 3 * n_pts).reshape(n_pts, 1)

    full = eval_del_obj_del_rho(xz, xz, xz, xi, spi, sni, grad_fn, 3 * n_pts,
                                subsample_frac=1.0).sum()
    sampled = np.mean([
        eval_del_obj_del_rho(xz, xz, xz, xi, spi, sni, grad_fn, 3 * n_pts,
                             subsample_frac=0.5, key=jax.random.PRNGKey(k)).sum()
        for k in range(200)
    ])
    # E[subsampled] should equal the full-batch gradient sum, not a fraction of it.
    assert abs(sampled - full) < 0.05 * full, (sampled, full)
