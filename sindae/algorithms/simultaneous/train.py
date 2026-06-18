"""
train.py  (simultaneous approach)

solve_simultaneous — build and solve the simultaneous NLP with IPOPT.

The simultaneous approach places NN weights/biases directly inside the NLP
as decision variables.  IPOPT optimises states, NN outputs, and NN parameters
jointly in a single solve — no outer training loop required.

Two sub-approaches are supported (controlled by ``use_gbm``):
  False (default) : expression-writing — NNBlock, exact Hessian available
                    → solved with ``SolverFactory('pounce')`` (ASL interface)
  True            : grey-box (NNSimulGreyBoxModel) — requires L-BFGS
                    → solved with ``SolverFactory('cyipopt')`` (needed for
                       ExternalGreyBoxBlock; standard IPOPT cannot handle it)
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import pyomo.environ as pyo
from pyomo.common.timing import HierarchicalTimer

from sindae.algorithms.timing_utils import tmp_log_path, parse_pounce_log, set_output_file

from sindae.data_utils import InstanceData
from sindae.nn_utils import SimpleMLP
from sindae.problem import ProblemDefinition
from sindae.algorithms.simultaneous.model_builder import (
    build_simultaneous_model,
    build_simultaneous_model_gbm,
    extract_mlp,
)

logger = logging.getLogger(__name__)


def solve_simultaneous(
    problem: ProblemDefinition,
    mlp: SimpleMLP,
    data: InstanceData,
    traj_indices: Optional[List[int]] = None,
    smoother_model: Optional[pyo.ConcreteModel] = None,
    use_gbm: bool = False,
    reg_coef: float = 0.0,
    pounceoptions: Optional[dict] = None,
    tee: bool = False,
    timer: Optional[HierarchicalTimer] = None,
    unfix_io: bool = True,
) -> Tuple[pyo.ConcreteModel, SimpleMLP]:
    """
    Build and solve the simultaneous NLP, returning the solved model and
    the trained SimpleMLP.

    Parameters
    ----------
    problem        : ProblemDefinition
    mlp            : SimpleMLP          (architecture + initial weights)
    data           : InstanceData
        Provides normalization statistics (input_mean/std, output_mean/std).
    traj_indices   : List[int], optional  (default: all trajectories)
    smoother_model : pyo.ConcreteModel, optional
        Solved smoother model to reuse (warm-starts the simultaneous solve
        and avoids rebuilding / re-discretising the model).
    use_gbm        : bool
        False (default): expression-writing, exact Hessian available.
        True:            grey-box (NNSimulGreyBoxModel), requires L-BFGS.
    reg_coef       : float
        L2 regularisation coefficient on NN parameters.
    pounceoptions  : dict, optional
        Extra IPOPT options, e.g. ``{'max_iter': 500, 'tol': 1e-6,
        'hessian_approximation': 'limited-memory'}``.

    Returns
    -------
    m           : pyo.ConcreteModel  (solved; pass to ``problem.extract_arrays``)
    trained_mlp : SimpleMLP          (optimised weights extracted from the NLP)
    """
    if timer is None:
        timer = HierarchicalTimer()
    if traj_indices is None:
        traj_indices = list(range(problem.num_trajectories))

    # ── Build model ────────────────────────────────────────────────────────────
    timer.start('build')
    try:
        if use_gbm:
            logger.info(
                f"=== Building simultaneous GBM model for {len(traj_indices)} trajectories ==="
            )
            m = build_simultaneous_model_gbm(
                problem, mlp, traj_indices, data,
                smoother_model=smoother_model,
                reg_coef=reg_coef,
                unfix_io=unfix_io,
            )
        else:
            logger.info(
                f"=== Building simultaneous (expr-writing) model for "
                f"{len(traj_indices)} trajectories ==="
            )
            m = build_simultaneous_model(
                problem, mlp, traj_indices, data,
                smoother_model=smoother_model,
                reg_coef=reg_coef,
                unfix_io=unfix_io,
            )
    finally:
        timer.stop('build')

    # ── Helpers: configure and call solver ────────────────────────────────────
    def _solve_ipopt(extra_opts):
        """ASL-based POUNCE: expression-writing path (no ExternalGreyBoxBlock)."""
        ipopt = pyo.SolverFactory('pounce')
        if pounceoptions:
            for k, v in pounceoptions.items():
                ipopt.options[k] = v
        for k, v in extra_opts.items():
            ipopt.options[k] = v
        _log = tmp_log_path()
        set_output_file(ipopt, _log)
        result = ipopt.solve(m, tee=tee)
        timing = parse_pounce_log(_log)
        os.unlink(_log)
        logger.info(
            f"  POUNCE: {result.solver.status} / {result.solver.termination_condition}"
        )
        return result, timing

    def _solve_cyipopt(extra_opts):
        """cyipopt: GBM path (required for ExternalGreyBoxBlock)."""
        solver = pyo.SolverFactory('cyipopt')
        if pounceoptions:
            for k, v in pounceoptions.items():
                solver.config.options[k] = v
        for k, v in extra_opts.items():
            solver.config.options[k] = v
        _log = tmp_log_path()
        set_output_file(solver, _log, is_cyipopt=True)
        result = solver.solve(m, tee=tee)
        timing = parse_pounce_log(_log)
        os.unlink(_log)
        logger.info(
            f"  cyipopt: {result.solver.status} / {result.solver.termination_condition}"
        )
        return result, timing

    # ── Solve ──────────────────────────────────────────────────────────────────
    timer.start('solve')
    try:
        if use_gbm:
            logger.info("=== Solving simultaneous GBM model (cyipopt, L-BFGS) ===")
            result, pouncetiming = _solve_cyipopt({'hessian_approximation': 'limited-memory'})
        else:
            logger.info("=== Solving simultaneous model (POUNCE, expr-writing) ===")
            result, pouncetiming = _solve_ipopt({})
    finally:
        timer.stop('solve')

    m._solver_result = result
    m._pouncetiming  = pouncetiming

    # ── Extract trained MLP ────────────────────────────────────────────────────
    trained_mlp = extract_mlp(m)
    logger.info("=== Simultaneous solve complete ===")

    return m, trained_mlp
