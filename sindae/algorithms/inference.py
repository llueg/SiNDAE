"""
inference.py

Embed a trained MLP into a new problem as a hard GBM constraint and solve.

No data-fit objective.  When ``slack_coef=0`` (default) the NN equality is
enforced as a hard GBM output constraint — the system is square and POUNCE
finds the unique feasible trajectory.

When ``slack_coef > 0`` the constraint is relaxed via ℓ₁ slack variables
(same pattern as the decomp subproblem):

    norm_output[t,k] - nn_z[t,k] = sp[t,k] - sn[t,k],   sp,sn ≥ 0
    min  slack_coef * mean(sp + sn)

This adds degrees of freedom and can help convergence when the trained NN
does not fit the inference problem's dynamics exactly.

Normalization statistics are taken from an InstanceData extracted during
training so the NN is evaluated in the same normalised space it was trained in.

API
---
  make_inference_model(problem, mlp, traj_indices, data, slack_coef=0.0)
      -> ConcreteModel

  solve_inference(problem, mlp, data, traj_indices=None,
                  slack_coef=0.0, solver_options=None, backend='pounce', tee=False)
      -> ConcreteModel
"""
from __future__ import annotations

import logging
from typing import List, Optional

import pyomo.environ as pyo
from pyomo.common.timing import HierarchicalTimer

from sindae.solvers import make_nlp_solver
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock

from sindae.data_utils import InstanceData
from sindae.nn_utils import SimpleMLP
from sindae.problem import ProblemDefinition
from sindae.algorithms.decomp.nn_gbm import NNGreyBoxModel
from sindae.algorithms.decomp.kkt_utils import NN_SLACK_POS_NAME, NN_SLACK_NEG_NAME
from sindae.algorithms.model_builder_utils import (
    NORM_INPUT_NAME,
    NORM_OUTPUT_NAME,
    NN_Z_NAME,
    add_norm_vars_to_block,
    add_normalization_to_block,
)

logger = logging.getLogger(__name__)


def make_inference_model(
    problem: ProblemDefinition,
    mlp: SimpleMLP,
    traj_indices: List[int],
    data: InstanceData,
    slack_coef: float = 0.0,
) -> pyo.ConcreteModel:
    """
    Build an inference NLP: DAE + trained NN embedded as a GBM constraint.

    Parameters
    ----------
    problem      : ProblemDefinition
        Inference problem — may have different ICs / dynamics from training.
        Must implement build_trajectory / discretize / get_input_vars /
        get_output_vars.
    mlp          : SimpleMLP  (trained)
    traj_indices : List[int]
    data         : InstanceData
        Provides input_mean/std and output_mean/std from training.
    slack_coef   : float
        When 0 (default) the NN equality is a hard GBM output constraint
        (square system, no objective).
        When > 0 the constraint is relaxed with ℓ₁ slack variables and the
        objective is ``slack_coef * mean(sp + sn)``.  The model is then
        over-determined with a least-infeasibility flavour.

    Returns
    -------
    m : pyo.ConcreteModel
        Extra attribute: ``m._traj_t_sorted`` : List[List[float]]
    """
    num_traj   = len(traj_indices)
    input_dim  = mlp.in_size
    output_dim = mlp.out_size

    m = pyo.ConcreteModel()
    m.traj_set     = pyo.RangeSet(0, num_traj - 1)
    m.trajectories = pyo.Block(m.traj_set)

    # Pre-discretisation: base DAE + normalisation vars/constraints
    for ii, gi in enumerate(traj_indices):
        block = m.trajectories[ii]
        problem.build_trajectory(block, gi)
        add_norm_vars_to_block(block, block.t, input_dim, output_dim)
        add_normalization_to_block(
            block,
            problem.get_input_vars, problem.get_output_vars,
            data.input_mean, data.input_std,
            data.output_mean, data.output_std,
            block.t,
        )

    problem.discretize(m)
    traj_t_sorted = [sorted(list(m.trajectories[ii].t)) for ii in range(num_traj)]
    m._traj_t_sorted = traj_t_sorted

    total_points = sum(len(ts) for ts in traj_t_sorted)
    gbm = NNGreyBoxModel(mlp, total_points)

    nn_inputs  = []
    nn_outputs = []

    if slack_coef > 0.0:
        # ── Slack path: nn_z intermediate vars, relaxed constraint ────────────
        for ii in range(num_traj):
            block = m.trajectories[ii]
            setattr(block, NN_Z_NAME,
                    pyo.Var(block.t, block.nn_output_set, initialize=0.0))
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
                return (getattr(b, NORM_OUTPUT_NAME)[t, k]
                        - getattr(b, NN_Z_NAME)[t, k]
                        == sp[t, k] - sn[t, k])

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

        m.slack_coef = pyo.Param(initialize=slack_coef, mutable=True)

        @m.Objective()
        def obj(mo):
            total_slack = 0.0
            for ii in range(num_traj):
                block   = mo.trajectories[ii]
                t_s     = traj_t_sorted[ii]
                sp_var  = getattr(block, NN_SLACK_POS_NAME)
                sn_var  = getattr(block, NN_SLACK_NEG_NAME)
                n_slack = len(t_s) * output_dim
                total_slack += pyo.quicksum(
                    sp_var[t, k] + sn_var[t, k]
                    for t in t_s for k in range(output_dim)
                ) / n_slack
            return mo.slack_coef * total_slack

    else:
        # ── Hard constraint path: GBM outputs = norm_output directly ──────────
        for ii in range(num_traj):
            block   = m.trajectories[ii]
            norm_in = getattr(block, NORM_INPUT_NAME)
            norm_out = getattr(block, NORM_OUTPUT_NAME)
            for t in traj_t_sorted[ii]:
                for j in range(input_dim):
                    nn_inputs.append(norm_in[t, j])
                for k in range(output_dim):
                    nn_outputs.append(norm_out[t, k])

        m.nn_block = ExternalGreyBoxBlock()
        m.nn_block.set_external_model(gbm, inputs=nn_inputs, outputs=nn_outputs)

        @m.Objective()
        def obj(mo):
            return 0.0

    return m


def solve_inference(
    problem: ProblemDefinition,
    mlp: SimpleMLP,
    data: InstanceData,
    traj_indices: Optional[List[int]] = None,
    slack_coef: float = 0.0,
    solver_options: Optional[dict] = None,
    backend: str = 'pounce',
    tee: bool = False,
    timer: Optional[HierarchicalTimer] = None,
) -> pyo.ConcreteModel:
    """
    Build and solve the inference NLP, returning the solved model.

    Parameters
    ----------
    problem         : ProblemDefinition
    mlp             : SimpleMLP  (trained)
    data            : InstanceData  (from training, for norm stats)
    traj_indices    : List[int], optional  (default: all trajectories)
    slack_coef      : float  (0 = hard constraint; > 0 = ℓ₁-relaxed)
    solver_options  : dict, optional  e.g. ``{'max_iter': 500, 'tol': 1e-8}``
        Passed to the selected NLP backend.
    backend         : str  (default ``'pounce'``; ``'cyipopt'`` / ``'ipopt'``
        select alternative grey-box-capable backends)
    tee             : bool

    Returns
    -------
    m : pyo.ConcreteModel  (solved)
    """
    if timer is None:
        timer = HierarchicalTimer()
    if traj_indices is None:
        traj_indices = list(range(problem.num_trajectories))

    mode = f"slack_coef={slack_coef}" if slack_coef > 0.0 else "hard constraint"
    logger.info(
        f"=== Building inference model for {len(traj_indices)} trajectories "
        f"({mode}) ==="
    )
    timer.start('build')
    m = make_inference_model(problem, mlp, traj_indices, data, slack_coef=slack_coef)
    timer.stop('build')

    logger.info("=== Solving inference model ===")

    solver = make_nlp_solver(backend, solver_options)

    timer.start('solve')
    res = solver.solve(m, tee=tee)
    timer.stop('solve')

    m._solver_result = res.result
    m._pounce_timing = res.timing
    logger.info(
        f"  Inference: {res.result.solver.status} / {res.result.solver.termination_condition}"
    )
    return m
