"""
train.py  (simultaneous approach)

solve_simultaneous — build and solve the simultaneous NLP in a single solver call.

The simultaneous approach places NN weights/biases directly inside the NLP
as decision variables.  The solver optimises states, NN outputs, and NN
parameters jointly in a single solve — no outer training loop required.

Two sub-approaches are supported (controlled by ``SimultaneousConfig.use_gbm``):
  False (default) : expression-writing — NNBlock, exact Hessian available
                    → solved with POUNCE on the ASL interface
  True            : grey-box (NNSimulGreyBoxModel) — requires L-BFGS
                    → solved with POUNCE's cyipopt-style grey-box interface
                       (ExternalGreyBoxBlock; cyipopt selectable). Standard
                       ASL/IPOPT cannot consume grey-box callbacks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pyomo.environ as pyo
from pyomo.common.timing import HierarchicalTimer

from sindae.solvers import make_nlp_solver

from sindae.data_utils import InstanceData
from sindae.nn_utils import SimpleMLP
from sindae.problem import ProblemDefinition
from sindae.algorithms.simultaneous.model_builder import (
    build_simultaneous_model,
    build_simultaneous_model_gbm,
    extract_mlp,
)

logger = logging.getLogger(__name__)


@dataclass
class SimultaneousConfig:
    """Hyperparameters for the simultaneous (single-NLP) training approach."""
    use_gbm:  bool  = False   # grey-box (L-BFGS) vs expression-writing (exact Hessian)
    reg_coef: float = 0.0     # L2 regularization coefficient on NN parameters


def solve_simultaneous(
    problem: ProblemDefinition,
    mlp: SimpleMLP,
    cfg: SimultaneousConfig,
    data: InstanceData,
    smoother_model: Optional[pyo.ConcreteModel] = None,
    pounce_options: Optional[dict] = None,
    backend: Optional[str] = None,
    traj_indices: Optional[List[int]] = None,
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
    cfg            : SimultaneousConfig
        Algorithm hyperparameters (``use_gbm``, ``reg_coef``).
    data           : InstanceData
        Provides normalization statistics (input_mean/std, output_mean/std).
    smoother_model : pyo.ConcreteModel, optional
        Solved smoother model to reuse (warm-starts the simultaneous solve
        and avoids rebuilding / re-discretising the model).
    pounce_options : dict, optional
        Extra solver options, e.g. ``{'max_iter': 500, 'tol': 1e-6,
        'hessian_approximation': 'limited-memory'}``.  Passed to POUNCE
        (expression-writing) or cyipopt (GBM) depending on ``cfg.use_gbm``.
    backend        : str, optional
        NLP backend (``'pounce'`` default, ``'ipopt'`` / ``'cyipopt'``).
        Applies to both paths.  When ``cfg.use_gbm`` is True the backend must be
        grey-box-capable (POUNCE / cyipopt); ``'ipopt'`` is rejected there.
    traj_indices   : List[int], optional  (default: all trajectories)
    tee            : bool
        Stream solver output to stdout.
    timer          : HierarchicalTimer, optional
        Reuse an external timer; a fresh one is created when omitted.
    unfix_io       : bool
        Unfix the NN input/output variables before solving (default True).

    Returns
    -------
    m           : pyo.ConcreteModel  (solved; pass to ``extract_instance_data``)
    trained_mlp : SimpleMLP          (optimised weights extracted from the NLP)
    """
    if timer is None:
        timer = HierarchicalTimer()
    if traj_indices is None:
        traj_indices = list(range(problem.num_trajectories))

    use_gbm  = cfg.use_gbm
    reg_coef = cfg.reg_coef

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

    # ── Solve ──────────────────────────────────────────────────────────────────
    timer.start('solve')
    try:
        if use_gbm:
            # Grey-box (ExternalGreyBoxBlock) needs a grey-box-capable backend
            # (POUNCE default; cyipopt selectable) and L-BFGS — the GBM supplies
            # no Hessian.  POUNCE forces limited-memory itself; cyipopt needs it
            # passed in, so set it for both.
            solver = make_nlp_solver(backend or 'pounce', pounce_options)
            logger.info(
                f"=== Solving simultaneous GBM model ({solver.name}, L-BFGS) ==="
            )
            res = solver.solve(
                m, tee=tee,
                extra_options={'hessian_approximation': 'limited-memory'},
            )
        else:
            solver = make_nlp_solver(backend or 'pounce', pounce_options)
            logger.info(
                f"=== Solving simultaneous model ({solver.name}, expr-writing) ==="
            )
            res = solver.solve(m, tee=tee)
        logger.info(
            f"  {solver.name}: {res.result.solver.status} "
            f"/ {res.result.solver.termination_condition}"
        )
    finally:
        timer.stop('solve')

    m._solver_result = res.result
    m._pounce_timing = res.timing

    # ── Extract trained MLP ────────────────────────────────────────────────────
    trained_mlp = extract_mlp(m)
    logger.info("=== Simultaneous solve complete ===")

    return m, trained_mlp
