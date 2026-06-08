"""
NNSimulGreyBoxModel: ExternalGreyBoxModel for the simultaneous approach.

GBM inputs:  norm_input_flat (n_pts * in_size) + flat_theta (n_params)
GBM outputs: norm_output_flat (n_pts * out_size)

The GBM enforces:
    norm_output[p, :] = NN(norm_input[p, :]; theta)   for p = 0, ..., n_pts - 1

theta is a Pyomo decision variable (part of the NLP).

Jacobian structure
------------------
Rows = outputs, Cols = inputs:

       norm_input_0  ...  norm_input_{n-1}   flat_theta
  z_0  [ J_x_0            0                  J_t_0  ]
  z_1  [ 0            ...  0                  J_t_1  ]
   :
  z_{n-1}[ 0          ... J_x_{n-1}           J_t_{n-1}]

  J_x_i  in (out_size, in_size)  — block-diagonal
  J_t_i  in (out_size, n_params) — dense columns (same column block for all rows)

No Hessian — use L-BFGS in IPOPT (hessian_approximation=limited-memory).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse as sps

from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
from sindae.nn_utils import SimpleMLP, flatten_fn, make_unflatten_fn

jax.config.update("jax_enable_x64", True)


def _make_simul_jax_fns(unflatten_fn):
    """Create JIT-compiled JAX functions for NNSimulGreyBoxModel."""

    @jax.jit
    def eval_fn(flat_theta, x_batch):
        """Forward pass.  Returns (n_pts, out_size)."""
        mlp = unflatten_fn(flat_theta)
        return jax.vmap(mlp)(x_batch)

    @jax.jit
    def jac_x_fn(flat_theta, x_batch):
        """Jacobian d(output)/d(input) per point.  Returns (n_pts, out_size, in_size)."""
        mlp = unflatten_fn(flat_theta)
        return jax.vmap(jax.jacobian(mlp))(x_batch)

    @jax.jit
    def jac_theta_fn(flat_theta, x_batch):
        """Jacobian d(all outputs)/d(theta).  Returns (n_pts * out_size, n_params)."""
        def f(ft):
            mlp = unflatten_fn(ft)
            return jax.vmap(mlp)(x_batch).reshape(-1)
        return jax.jacobian(f)(flat_theta)

    return eval_fn, jac_x_fn, jac_theta_fn


class NNSimulGreyBoxModel(ExternalGreyBoxModel):
    """
    Grey-box model for the simultaneous approach where theta is an NLP variable.

    Parameters
    ----------
    mlp       : SimpleMLP  (provides structure + initial weights for JAX tracing)
    num_points: int        (total discretisation points across all trajectories)
    """

    def __init__(self, mlp: SimpleMLP, num_points: int):
        super().__init__()

        in_size          = mlp.in_size
        out_size         = mlp.out_size
        flat_theta_init  = np.array(flatten_fn(mlp))
        n_params         = len(flat_theta_init)
        unflatten_fn     = make_unflatten_fn(mlp)

        self._in_size    = in_size
        self._out_size   = out_size
        self._num_points = num_points
        self._n_params   = n_params

        self._x_batch    = np.zeros((num_points, in_size))
        self._flat_theta = flat_theta_init.copy()

        self._num_inputs  = num_points * in_size + n_params
        self._num_outputs = num_points * out_size

        self._input_names = (
            [f'x_p{p}_d{j}' for p in range(num_points) for j in range(in_size)]
            + [f'theta_{j}'  for j in range(n_params)]
        )
        self._output_names = [
            f'z_p{p}_d{k}' for p in range(num_points) for k in range(out_size)
        ]

        # ── Jacobian sparsity (precomputed once) ─────────────────────────────
        # x-part: block-diagonal (out_size × in_size per point)
        rows_x, cols_x = [], []
        for p in range(num_points):
            r  = np.arange(p * out_size, (p + 1) * out_size)
            c  = np.arange(p * in_size,  (p + 1) * in_size)
            rr, cc = np.meshgrid(r, c, indexing='ij')
            rows_x.append(rr.ravel())
            cols_x.append(cc.ravel())
        rows_x = np.concatenate(rows_x)
        cols_x = np.concatenate(cols_x)

        # theta-part: dense (n_pts * out_size) × n_params block
        theta_col_start = num_points * in_size
        rows_t = np.repeat(np.arange(num_points * out_size), n_params)
        cols_t = np.tile(
            np.arange(theta_col_start, theta_col_start + n_params),
            num_points * out_size,
        )

        self._jac_rows  = np.concatenate([rows_x, rows_t])
        self._jac_cols  = np.concatenate([cols_x, cols_t])
        self._jac_shape = (self._num_outputs, self._num_inputs)

        # ── JIT-compile JAX functions ─────────────────────────────────────────
        self._eval_fn, self._jac_x_fn, self._jac_theta_fn = (
            _make_simul_jax_fns(unflatten_fn)
        )

    # ── ExternalGreyBoxModel interface ────────────────────────────────────────

    def input_names(self):
        return self._input_names

    def output_names(self):
        return self._output_names

    def set_input_values(self, input_values: np.ndarray) -> None:
        n_x = self._num_points * self._in_size
        self._x_batch    = input_values[:n_x].reshape(self._num_points, self._in_size)
        self._flat_theta = input_values[n_x:]

    def evaluate_outputs(self) -> np.ndarray:
        z = self._eval_fn(jnp.array(self._flat_theta), jnp.array(self._x_batch))
        return np.array(z, dtype=np.float64).ravel()

    def evaluate_jacobian_outputs(self) -> sps.coo_matrix:
        ft  = jnp.array(self._flat_theta)
        xb  = jnp.array(self._x_batch)
        # (n_pts, out_size, in_size) and (n_pts * out_size, n_params)
        jac_x = np.array(self._jac_x_fn(ft, xb),     dtype=np.float64)
        jac_t = np.array(self._jac_theta_fn(ft, xb),  dtype=np.float64)
        # C-order ravel matches precomputed sparsity indices (verified by construction)
        data = np.concatenate([jac_x.ravel(), jac_t.ravel()])
        return sps.coo_matrix(
            (data, (self._jac_rows, self._jac_cols)), shape=self._jac_shape
        )
