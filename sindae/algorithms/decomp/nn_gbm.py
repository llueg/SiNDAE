"""
NNGreyBoxModel: ExternalGreyBoxModel for a neural network evaluated at multiple points.

GBM inputs:  [x[0,:], x[1,:], ..., x[num_points-1,:]]  (flattened, length num_points * in_size)
GBM outputs: [nn_z[0,:], ..., nn_z[num_points-1,:]]     (flattened, length num_points * out_size)

Output constraint (enforced by ExternalGreyBoxBlock):
    nn_z[i, :] = NN(x[i, :]; theta)  for i = 0, ..., num_points-1

theta is held internally and updated via update_mlp() between training steps.
Jacobian sparsity is block-diagonal (each output point depends only on its own input point).
"""
import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse as sps
import equinox as eqx

from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
from sindae.nn_utils import SimpleMLP

jax.config.update("jax_enable_x64", True)


@eqx.filter_jit
def _eval_all(mlp: SimpleMLP, xs: jnp.ndarray) -> jnp.ndarray:
    """Vectorized forward pass.
    xs: (num_points, in_size) -> returns (num_points, out_size)
    """
    return jax.vmap(mlp)(xs)


@eqx.filter_jit
def _jac_all(mlp: SimpleMLP, xs: jnp.ndarray) -> jnp.ndarray:
    """Vectorized Jacobian of NN outputs w.r.t. inputs.
    xs: (num_points, in_size) -> returns (num_points, out_size, in_size)
    """
    return jax.vmap(jax.jacobian(mlp))(xs)


@eqx.filter_jit
def _hess_all(mlp: SimpleMLP, xs: jnp.ndarray, lams: jnp.ndarray) -> jnp.ndarray:
    """Vectorized Hessian of Lagrangian w.r.t. inputs.
    lams: (num_points, out_size) — output constraint multipliers
    xs:   (num_points, in_size)
    returns: (num_points, in_size, in_size)
    """
    def hess_single(x, lam):
        return jax.hessian(lambda _x: jnp.dot(mlp(_x), lam))(x)
    return jax.vmap(hess_single)(xs, lams)


class NNGreyBoxModel(ExternalGreyBoxModel):
    """
    Grey-box model for a neural network z = NN(x; theta) evaluated at num_points points.

    Parameters
    ----------
    mlp : SimpleMLP
        The neural network (equinox module).
    num_points : int
        Number of evaluation points (e.g. discretization points in the DAE).
    """

    def __init__(self, mlp: SimpleMLP, num_points: int):
        super().__init__()
        self._mlp = mlp
        self._num_points = num_points
        self._in_size = mlp.in_size
        self._out_size = mlp.out_size
        self._num_inputs = num_points * mlp.in_size
        self._num_outputs = num_points * mlp.out_size

        self._inputs = np.zeros((num_points, mlp.in_size))
        self._output_constraint_multipliers = np.zeros((num_points, mlp.out_size))

        self._input_names = [
            f'x_p{i}_d{j}' for i in range(num_points) for j in range(mlp.in_size)
        ]
        self._output_names = [
            f'z_p{i}_d{j}' for i in range(num_points) for j in range(mlp.out_size)
        ]

        # Precompute Jacobian sparsity structure (block diagonal).
        # Block i occupies rows [i*out_size, (i+1)*out_size) x cols [i*in_size, (i+1)*in_size).
        rows, cols = [], []
        for i in range(num_points):
            r = np.arange(i * mlp.out_size, (i + 1) * mlp.out_size)
            c = np.arange(i * mlp.in_size,  (i + 1) * mlp.in_size)
            rr, cc = np.meshgrid(r, c, indexing='ij')
            rows.append(rr.ravel())
            cols.append(cc.ravel())
        self._jac_rows = np.concatenate(rows)
        self._jac_cols = np.concatenate(cols)
        self._jac_shape = (self._num_outputs, self._num_inputs)

        # Same block structure for the Hessian (each block is in_size x in_size).
        h_rows, h_cols = [], []
        for i in range(num_points):
            c = np.arange(i * mlp.in_size, (i + 1) * mlp.in_size)
            rr, cc = np.meshgrid(c, c, indexing='ij')
            # lower triangular only (pynumero convention)
            mask = rr >= cc
            h_rows.append(rr[mask])
            h_cols.append(cc[mask])
        self._hess_rows = np.concatenate(h_rows)
        self._hess_cols = np.concatenate(h_cols)
        self._hess_shape = (self._num_inputs, self._num_inputs)

    def update_mlp(self, new_mlp: SimpleMLP) -> None:
        """Replace the internal MLP (e.g. after a gradient step). Thread-safe w.r.t. structure."""
        self._mlp = new_mlp

    # ------------------------------------------------------------------
    # ExternalGreyBoxModel interface
    # ------------------------------------------------------------------

    def input_names(self):
        return self._input_names

    def output_names(self):
        return self._output_names

    def set_input_values(self, input_values: np.ndarray) -> None:
        self._inputs[:] = input_values.reshape(self._num_points, self._in_size)

    def set_output_constraint_multipliers(self, multiplier_values: np.ndarray) -> None:
        self._output_constraint_multipliers[:] = multiplier_values.reshape(
            self._num_points, self._out_size
        )

    def evaluate_outputs(self) -> np.ndarray:
        outputs = _eval_all(self._mlp, jnp.array(self._inputs))
        return np.array(outputs, dtype=np.float64).ravel()

    def evaluate_jacobian_outputs(self) -> sps.coo_matrix:
        # jacs shape: (num_points, out_size, in_size)
        # Flattening in C-order matches the precomputed row/col index ordering.
        jacs = _jac_all(self._mlp, jnp.array(self._inputs))
        data = np.array(jacs, dtype=np.float64).ravel()
        return sps.coo_matrix((data, (self._jac_rows, self._jac_cols)), shape=self._jac_shape)

    def evaluate_hessian_outputs(self) -> sps.coo_matrix:
        # Hessian of sum_ij lambda_ij * NN_j(x_i) w.r.t. all inputs.
        # Full (dense) blocks on the diagonal; returned as lower-triangular COO.
        lams = jnp.array(self._output_constraint_multipliers)
        xs = jnp.array(self._inputs)
        # hess shape: (num_points, in_size, in_size)
        hess = _hess_all(self._mlp, xs, lams)
        hess_np = np.array(hess, dtype=np.float64)

        # Extract lower-triangular entries block by block.
        data = []
        for i in range(self._num_points):
            block = hess_np[i]  # (in_size, in_size)
            for j in range(self._in_size):
                for k in range(j + 1):  # lower triangular: k <= j
                    data.append(block[j, k])
        data = np.array(data, dtype=np.float64)
        return sps.coo_matrix(
            (data, (self._hess_rows, self._hess_cols)), shape=self._hess_shape
        )
