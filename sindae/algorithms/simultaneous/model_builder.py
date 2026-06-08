"""
model_builder.py  (simultaneous approach)

Two model-building functions that mirror build_smoother_model / build_decomp_model:

  build_simultaneous_model
      Expression-writing: NNBlock weights/biases as Pyomo Vars; NN forward pass
      written as Pyomo arithmetic expressions.  Exact Hessian available.

  build_simultaneous_model_gbm
      Grey-box: flat theta Pyomo Vars + NNSimulGreyBoxModel; Jacobian via JAX.
      Requires L-BFGS (no Hessian provided).

Both functions:
  - Accept an optional ``smoother_model`` to reuse the already-solved and
    discretised smoother NLP (same pattern as build_decomp_model).
  - Add a data-fit objective with optional L2 regularisation on NN parameters.

After solving, call ``extract_mlp(m)`` to recover a SimpleMLP with the
optimised weights.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock

from sindae.data_utils import InstanceData
from sindae.nn_utils import SimpleMLP, flatten_fn, make_unflatten_fn
from sindae.algorithms.simultaneous.nn_block import (
    make_nn_output_expression,
    eqx_mlp_to_nn_block,
    nn_block_to_eqx_mlp,
)
from sindae.problem import ProblemDefinition

from sindae.algorithms.model_builder_utils import (
    NORM_INPUT_NAME,
    NORM_OUTPUT_NAME,
    _remove_smoother_components,
    _add_norm_and_io_constr_post_disc,
    _build_fresh_base,
    _unfix_nn_inputs_and_outputs,
    build_data_fit_expr,
)
from sindae.algorithms.simultaneous.gbm import NNSimulGreyBoxModel


# ---------------------------------------------------------------------------
# Public utility: extract trained MLP from a solved simultaneous model
# ---------------------------------------------------------------------------

def extract_mlp(m: pyo.ConcreteModel) -> SimpleMLP:
    """
    Extract a SimpleMLP with the optimised weights from a solved simultaneous model.

    Works for both the expression-writing path (reads Pyomo NNBlock Var values)
    and the GBM path (reads flat Pyomo Var values).
    """
    if hasattr(m, '_nn_block'):
        return nn_block_to_eqx_mlp(m._nn_block)
    if hasattr(m, '_nn_params_unflatten'):
        import jax.numpy as jnp
        flat = np.array([pyo.value(m.nn_params[i]) for i in range(len(list(m.nn_params)))])
        return m._nn_params_unflatten(jnp.array(flat))
    raise ValueError(
        "Cannot find NN components on model — was it built with "
        "build_simultaneous_model or build_simultaneous_model_gbm?"
    )


# ---------------------------------------------------------------------------
# Expression-writing simultaneous model
# ---------------------------------------------------------------------------

def build_simultaneous_model(
    problem: ProblemDefinition,
    mlp: SimpleMLP,
    traj_indices: List[int],
    data: InstanceData,
    smoother_model: Optional[pyo.ConcreteModel] = None,
    reg_coef: float = 0.0,
    unfix_io: bool = True,
) -> pyo.ConcreteModel:
    """
    Build a simultaneous NLP using expression-writing.

    NN weights and biases are Pyomo decision variables (as an ``NNBlock``).
    The NN forward pass is written symbolically as Pyomo arithmetic expressions,
    yielding exact second-order information (Hessian available for IPOPT).

    Parameters
    ----------
    problem        : ProblemDefinition
    mlp            : SimpleMLP          (used for architecture + initial weights)
    traj_indices   : List[int]
    data           : InstanceData
        Provides normalization statistics (input_mean/std, output_mean/std).
    smoother_model : pyo.ConcreteModel, optional
        When provided, reuses the solved smoother NLP in-place (no rebuild /
        re-discretisation); IPOPT warm-starts from the smoother solution.
    reg_coef       : float
        L2 regularisation coefficient on all NN weights and biases.

    Returns
    -------
    m : pyo.ConcreteModel
        Extra Python attributes:
          ``m._nn_block``          : NNBlock (Pyomo weight/bias Vars)
          ``m._traj_t_sorted``     : List[List[float]]
          ``m._traj_norm_target``  : List[np.ndarray]
    """
    num_traj   = len(traj_indices)
    input_dim  = mlp.in_size
    output_dim = mlp.out_size

    # Build or reuse base model
    if smoother_model is not None:
        m = smoother_model
        _remove_smoother_components(m, num_traj)
        traj_t_sorted    = m._traj_t_sorted
        traj_norm_target = m._traj_norm_target
        # Add norm_input / norm_output post-discretisation
        for ii in range(num_traj):
            _add_norm_and_io_constr_post_disc(
                m.trajectories[ii], input_dim, output_dim,
                data.input_mean, data.input_std, data.output_mean, data.output_std,
                problem.get_input_vars, problem.get_output_vars,
            )
    else:
        m, traj_t_sorted, traj_norm_target = _build_fresh_base(
            problem, mlp, traj_indices, data
        )

    obs_dim = traj_norm_target[0].shape[1]

    if unfix_io:
        _unfix_nn_inputs_and_outputs(m, problem)

    # NNBlock: weights / biases as Pyomo Vars (shared across trajectories)
    nn_block    = eqx_mlp_to_nn_block(mlp)
    m._nn_block = nn_block   # stored for extract_mlp() after solve

    # NN expression constraints (post-discretisation)
    for ii in range(num_traj):
        block  = m.trajectories[ii]
        t_list = traj_t_sorted[ii]

        nn_vals = {
            t: make_nn_output_expression(
                nn_block,
                [getattr(block, NORM_INPUT_NAME)[t, i] for i in range(input_dim)],
            )
            for t in t_list
        }

        block.nn_output_constr = pyo.Constraint(
            t_list, block.nn_output_set,
            rule=lambda b, t, k: getattr(b, NORM_OUTPUT_NAME)[t, k] == nn_vals[t][k],
        )

    # Objective: data fit + optional L2 regularisation
    @m.Objective()
    def obj(mo):
        data_fit = build_data_fit_expr(mo, num_traj, traj_t_sorted, traj_norm_target, obs_dim)
        reg = pyo.quicksum(
            nn_block.layers[li].weight[j, i] ** 2
            for li in range(len(nn_block.layers))
            for j in range(nn_block.layers[li].out_features)
            for i in range(nn_block.layers[li].in_features)
        ) + pyo.quicksum(
            nn_block.layers[li].bias[j] ** 2
            for li in range(len(nn_block.layers))
            for j in range(nn_block.layers[li].out_features)
        )
        return data_fit + reg_coef * reg

    return m


# ---------------------------------------------------------------------------
# GBM simultaneous model
# ---------------------------------------------------------------------------

def build_simultaneous_model_gbm(
    problem: ProblemDefinition,
    mlp: SimpleMLP,
    traj_indices: List[int],
    data: InstanceData,
    smoother_model: Optional[pyo.ConcreteModel] = None,
    reg_coef: float = 0.0,
    unfix_io: bool = True,
) -> pyo.ConcreteModel:
    """
    Build a simultaneous NLP using the grey-box (GBM) formulation.

    The NN parameters theta are flat Pyomo ``Var`` objects (``m.nn_params``).
    ``NNSimulGreyBoxModel`` evaluates ``NN(norm_input; theta)`` and provides
    the Jacobian w.r.t. both ``norm_input`` and ``theta`` via JAX.

    Because no Hessian is provided, IPOPT must use L-BFGS
    (``hessian_approximation='limited-memory'``).

    Parameters
    ----------
    problem        : ProblemDefinition
    mlp            : SimpleMLP          (architecture + initial weights)
    traj_indices   : List[int]
    data           : InstanceData
        Provides normalization statistics (input_mean/std, output_mean/std).
    smoother_model : pyo.ConcreteModel, optional
    reg_coef       : float  (L2 on theta; 0.0 omits the term)

    Returns
    -------
    m : pyo.ConcreteModel
        Extra Python attributes:
          ``m.nn_params``             : pyo.Var (flat theta)
          ``m._nn_params_unflatten``  : callable flat->SimpleMLP (for extract_mlp)
          ``m._traj_t_sorted``        : List[List[float]]
          ``m._traj_norm_target``     : List[np.ndarray]
    """
    num_traj   = len(traj_indices)
    input_dim  = mlp.in_size
    output_dim = mlp.out_size

    flat_theta_init = np.array(flatten_fn(mlp))
    n_params        = len(flat_theta_init)
    unflatten_fn    = make_unflatten_fn(mlp)

    # Build or reuse base model
    if smoother_model is not None:
        m = smoother_model
        _remove_smoother_components(m, num_traj)
        traj_t_sorted    = m._traj_t_sorted
        traj_norm_target = m._traj_norm_target
        # Add norm_input / norm_output post-discretisation
        for ii in range(num_traj):
            _add_norm_and_io_constr_post_disc(
                m.trajectories[ii], input_dim, output_dim,
                data.input_mean, data.input_std, data.output_mean, data.output_std,
                problem.get_input_vars, problem.get_output_vars,
            )
    else:
        m, traj_t_sorted, traj_norm_target = _build_fresh_base(
            problem, mlp, traj_indices, data
        )

    obs_dim = traj_norm_target[0].shape[1]

    if unfix_io:
        _unfix_nn_inputs_and_outputs(m, problem)

    # Flat theta as Pyomo Vars
    _init = flat_theta_init.copy()
    m.nn_params = pyo.Var(
        range(n_params),
        initialize=lambda _m, i: float(_init[i]),
    )
    m._nn_params_unflatten = unflatten_fn   # stored for extract_mlp()

    # Collect GBM inputs / outputs in consistent ordering:
    #   Inputs:  [norm_input vars (all traj, time, dim)] + [nn_params vars]
    #   Outputs: [norm_output vars (all traj, time, dim)]
    nn_inputs  = []
    nn_outputs = []
    for ii in range(num_traj):
        block    = m.trajectories[ii]
        norm_in  = getattr(block, NORM_INPUT_NAME)
        norm_out = getattr(block, NORM_OUTPUT_NAME)
        for t in traj_t_sorted[ii]:
            for j in range(input_dim):
                nn_inputs.append(norm_in[t, j])
            for k in range(output_dim):
                nn_outputs.append(norm_out[t, k])

    nn_inputs += [m.nn_params[j] for j in range(n_params)]

    total_points = sum(len(ts) for ts in traj_t_sorted)
    gbm = NNSimulGreyBoxModel(mlp, total_points)

    m.nn_block_gbm = ExternalGreyBoxBlock()
    m.nn_block_gbm.set_external_model(gbm, inputs=nn_inputs, outputs=nn_outputs)

    # Objective: data fit + optional L2 regularisation on theta
    @m.Objective()
    def obj(mo):
        data_fit = build_data_fit_expr(mo, num_traj, traj_t_sorted, traj_norm_target, obs_dim)
        reg = pyo.quicksum(mo.nn_params[i] ** 2 for i in range(n_params))
        return data_fit + reg_coef * reg

    return m
