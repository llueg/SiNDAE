"""
smoother.py

Smoother NLP: a raw (unnormalised) z_smooth variable tracks get_output_vars and
is penalised for smoothness in time via its DerivativeVar.  Normalisation of inputs
and outputs is NOT applied here — norm_input / norm_output vars are added later
(post-discretisation) by the decomp / simultaneous model builders.

Only observed-variable normalisation (norm_obs) is built here, using obs stats
computed directly from problem.obs_values.

API
---
  build_smoother_model(problem, mlp, traj_indices, smooth_coef=1.0) -> ConcreteModel

  solve_smoother(problem, mlp, traj_indices=None, smooth_coef=1.0,
                 ipopt_options=None) -> ConcreteModel
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.timing import HierarchicalTimer

from sindae.nn_utils import SimpleMLP
from sindae.problem import ProblemDefinition
from sindae.algorithms.timing_utils import tmp_log_path, parse_ipopt_log, set_output_file
from sindae.algorithms.model_builder_utils import (
    Z_SMOOTH_NAME,
    Z_SMOOTH_DERIV_NAME,
    add_obs_normalization_to_block,
    build_data_fit_expr,
    _compute_norm_targets,
    _unfix_nn_inputs_and_outputs,
    _add_dual_suffixes,
)

logger = logging.getLogger(__name__)


def build_smoother_model(
    problem: ProblemDefinition,
    mlp: SimpleMLP,
    traj_indices: List[int],
    smooth_coef: float = 1.0,
    unfix_io: bool = True,
) -> pyo.ConcreteModel:
    """
    Build a multi-trajectory smoother NLP.

    A raw auxiliary variable ``z_smooth[t, k]`` is linked to ``get_output_vars``
    and its time derivative is penalised for smoothness.  No ``norm_input`` or
    ``norm_output`` vars are created here; those are added post-discretisation when
    the decomp or simultaneous model is built from the solved smoother.

    Obs stats are computed from ``problem.obs_values`` so this can be called
    before ``problem.norm_stats`` is set.

    Parameters
    ----------
    problem      : ProblemDefinition  (obs_times and obs_values must be set)
    mlp          : SimpleMLP          (used for out_size only)
    traj_indices : List[int]
    smooth_coef  : float
        Weight on smoothness penalty: ``smooth_coef * mean((dz_smooth/dt)^2)``.

    Returns
    -------
    m : pyo.ConcreteModel
        Extra attributes:
          ``m._traj_t_sorted``    : List[List[float]]
          ``m._traj_norm_target`` : List[np.ndarray]  normalised obs targets
    """
    assert problem.obs_values is not None, \
        "problem.obs_values must be set before calling build_smoother_model"

    num_traj   = len(traj_indices)
    output_dim = mlp.out_size
    obs_dim    = len(problem.obs_mean)

    m = pyo.ConcreteModel()
    m.traj_set     = pyo.RangeSet(0, num_traj - 1)
    m.trajectories = pyo.Block(m.traj_set)

    # Pre-discretisation: base DAE + z_smooth + obs normalisation
    for ii, gi in enumerate(traj_indices):
        block = m.trajectories[ii]
        problem.build_trajectory(block, gi)

        # Raw (unnormalised) z variable for smoothness penalty.
        # nn_output_set is kept on the block so _add_norm_and_io_constr_post_disc
        # can reuse it when norm_output is added later.
        block.nn_output_set = pyo.RangeSet(0, output_dim - 1)
        setattr(block, Z_SMOOTH_NAME,
                pyo.Var(block.t, block.nn_output_set, initialize=0.0))
        setattr(block, Z_SMOOTH_DERIV_NAME,
                dae.DerivativeVar(getattr(block, Z_SMOOTH_NAME), wrt=block.t))

        @block.Constraint(block.t, block.nn_output_set)
        def z_smooth_constr(b, t, k):
            return getattr(b, Z_SMOOTH_NAME)[t, k] == problem.get_output_vars(b, t)[k]

        add_obs_normalization_to_block(
            block, problem.get_obs_vars,
            problem.obs_mean, problem.obs_std,
            block.t, obs_dim,
        )

    problem.discretize(m)
    traj_t_sorted    = [sorted(list(m.trajectories[ii].t)) for ii in range(num_traj)]
    traj_norm_target = _compute_norm_targets(traj_indices, traj_t_sorted, problem)
    m._traj_t_sorted    = traj_t_sorted
    m._traj_norm_target = traj_norm_target
    
    _add_dual_suffixes(m)
    
    if unfix_io:
        _unfix_nn_inputs_and_outputs(m, problem)

    @m.Objective()
    def obj(mo):
        data_fit     = build_data_fit_expr(mo, num_traj, traj_t_sorted, traj_norm_target, obs_dim)
        total_smooth = 0.0
        for ii in range(num_traj):
            block    = mo.trajectories[ii]
            t_s      = traj_t_sorted[ii]
            t_span = float(t_s[-1] - t_s[0])
            d_smooth = getattr(block, Z_SMOOTH_DERIV_NAME)
            n_smooth = len(t_s) * output_dim
            total_smooth += (pyo.quicksum(
                d_smooth[t, k] ** 2 for t in t_s for k in range(output_dim)
            ) / n_smooth ) * t_span
        return data_fit + smooth_coef * total_smooth

    return m


def solve_smoother(
    problem: ProblemDefinition,
    mlp: SimpleMLP,
    traj_indices: Optional[List[int]] = None,
    smooth_coef: float = 1.0,
    ipopt_options: Optional[dict] = None,
    timer: Optional[HierarchicalTimer] = None,
    unfix_io: bool = True,
) -> pyo.ConcreteModel:
    """
    Build and solve the smoother NLP, returning the solved model.

    Parameters
    ----------
    problem       : ProblemDefinition  (obs_times and obs_values must be set)
    mlp           : SimpleMLP          (used for out_size only)
    traj_indices  : List[int], optional  (default: all trajectories)
    smooth_coef   : float
    ipopt_options : dict, optional  (e.g. ``{'tol': 1e-6, 'max_iter': 500}``)

    Returns
    -------
    m : pyo.ConcreteModel  (solved)
    """
    if timer is None:
        timer = HierarchicalTimer()
    if traj_indices is None:
        traj_indices = list(range(problem.num_trajectories))

    logger.info(
        f"=== Building smoother for {len(traj_indices)} trajectories "
        f"(smooth_coef={smooth_coef}) ==="
    )
    timer.start('build')
    m = build_smoother_model(problem,
                             mlp,
                             traj_indices,
                             smooth_coef=smooth_coef,
                             unfix_io=unfix_io
                             )
    timer.stop('build')

    ipopt = pyo.SolverFactory('pounce')
    if ipopt_options:
        for k, v in ipopt_options.items():
            ipopt.options[k] = v
    _log = tmp_log_path()
    set_output_file(ipopt, _log)

    timer.start('solve')
    result = ipopt.solve(m, tee=False)
    timer.stop('solve')

    m._solver_result = result
    m._ipopt_timing  = parse_ipopt_log(_log)
    os.unlink(_log)
    logger.info(
        f"  Smoother: {result.solver.status} / {result.solver.termination_condition}"
    )
    return m
