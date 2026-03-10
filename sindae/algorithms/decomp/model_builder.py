"""
model_builder.py  (decomposition approach)

Builds the multi-trajectory NLP with NNGreyBoxModel for the decomposition approach.

  build_decomp_model(problem, mlp, traj_indices, data, slack_coef=1.0,
                     smoother_model=None) -> (ConcreteModel, NNGreyBoxModel)

The smoother NLP is in ``sindae.algorithms.smoother``.
Shared building blocks (constants, block builders, helpers) are in
``sindae.algorithms.model_builder_utils`` and re-exported here for
backward compatibility.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock

from sindae.data_utils import InstanceData
from sindae.nn_utils import SimpleMLP
from sindae.algorithms.decomp.nn_gbm import NNGreyBoxModel
from sindae.problem import ProblemDefinition
from sindae.algorithms.decomp.kkt_utils import (
    NN_SLACK_POS_NAME, NN_SLACK_NEG_NAME,
)

from sindae.algorithms.model_builder_utils import (
    # constants — re-exported so existing imports from this module keep working
    NORM_INPUT_NAME,
    NORM_OUTPUT_NAME,
    NORM_OUTPUT_DERIV_NAME,
    NN_Z_NAME,
    NORM_OBS_NAME,
    # block builders
    add_norm_vars_to_block,
    add_normalization_to_block,
    add_obs_normalization_to_block,
    # objective helper
    build_data_fit_expr,
    # internal helpers
    _compute_norm_targets,
    _remove_smoother_components,
    _add_norm_and_io_constr_post_disc,
    _build_fresh_base,
    _unfix_nn_inputs_and_outputs
)

__all__ = [
    'NORM_INPUT_NAME', 'NORM_OUTPUT_NAME', 'NORM_OUTPUT_DERIV_NAME',
    'NN_Z_NAME', 'NORM_OBS_NAME',
    'add_norm_vars_to_block', 'add_normalization_to_block',
    'add_obs_normalization_to_block',
    'build_data_fit_expr',
    '_compute_norm_targets', '_remove_smoother_components', '_build_fresh_base',
    'build_decomp_model',
]


def build_decomp_model(
    problem: ProblemDefinition,
    mlp: SimpleMLP,
    traj_indices: List[int],
    data: InstanceData,
    slack_coef: float = 1.0,
    smoother_model: Optional[pyo.ConcreteModel] = None,
    unfix_io: bool = True,
) -> Tuple[pyo.ConcreteModel, NNGreyBoxModel]:
    """
    Build a multi-trajectory NLP with NNGreyBoxModel for the decomposition approach.

    Parameters
    ----------
    problem        : ProblemDefinition
    mlp            : SimpleMLP
    traj_indices   : List[int]
    data           : InstanceData
        Provides normalization statistics (input_mean/std, output_mean/std).
        Typically extracted from the solved smoother via
        ``extract_instance_data(problem, smoother_model)``.
    slack_coef     : float
    smoother_model : pyo.ConcreteModel, optional
        When provided the smoother structure is reused directly (no rebuild /
        re-discretisation); IPOPT warm-starts from the smoother solution.

    Returns
    -------
    m   : pyo.ConcreteModel
    gbm : NNGreyBoxModel

    Notes
    -----
    Two NLP variable roles (critical for the KKT gradient):
      ``norm_input[t, i]``  — NN evaluation point
      ``norm_obs[t, j]``    — observed variables in the objective
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

    # Auxiliary nn_z vars (GBM outputs, normalised space)
    for ii in range(num_traj):
        block = m.trajectories[ii]
        setattr(block, NN_Z_NAME, pyo.Var(block.t, block.nn_output_set, initialize=0.0))

    # GBM
    total_points = sum(len(ts) for ts in traj_t_sorted)
    gbm = NNGreyBoxModel(mlp, total_points)

    nn_inputs  = []
    nn_outputs = []
    for ii in range(num_traj):
        block    = m.trajectories[ii]
        norm_in  = getattr(block, NORM_INPUT_NAME)
        nn_z_var = getattr(block, NN_Z_NAME)
        for t in traj_t_sorted[ii]:
            for j in range(input_dim):
                nn_inputs.append(norm_in[t, j])
            for k in range(output_dim):
                nn_outputs.append(nn_z_var[t, k])

    m.nn_block = ExternalGreyBoxBlock()
    m.nn_block.set_external_model(gbm, inputs=nn_inputs, outputs=nn_outputs)

    # Slack variables + soft NN constraint
    for ii in range(num_traj):
        block = m.trajectories[ii]
        setattr(block, NN_SLACK_POS_NAME,
                pyo.Var(block.t, block.nn_output_set,
                        within=pyo.NonNegativeReals, initialize=0.0))
        setattr(block, NN_SLACK_NEG_NAME,
                pyo.Var(block.t, block.nn_output_set,
                        within=pyo.NonNegativeReals, initialize=0.0))

        @block.Constraint(block.t, block.nn_output_set)
        def nn_slack_constr(b, t, k):
            sp = getattr(b, NN_SLACK_POS_NAME)
            sn = getattr(b, NN_SLACK_NEG_NAME)
            return (getattr(b, NORM_OUTPUT_NAME)[t, k] - getattr(b, NN_Z_NAME)[t, k]
                    == sp[t, k] - sn[t, k])

    m.slack_coef = pyo.Param(initialize=slack_coef, mutable=True)

    @m.Objective()
    def obj(mo):
        data_fit    = build_data_fit_expr(mo, num_traj, traj_t_sorted, traj_norm_target, obs_dim)
        total_slack = 0.0
        for ii in range(num_traj):
            block  = mo.trajectories[ii]
            t_s    = traj_t_sorted[ii]
            sp_var = getattr(block, NN_SLACK_POS_NAME)
            sn_var = getattr(block, NN_SLACK_NEG_NAME)
            n_slack = len(t_s) * output_dim
            total_slack += pyo.quicksum(
                sp_var[t, k] + sn_var[t, k]
                for t in t_s for k in range(output_dim)
            ) / n_slack
        return data_fit + mo.slack_coef * total_slack

    return m, gbm
