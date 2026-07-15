"""
pounce_interface.py

Drive a Pyomo model (including ``ExternalGreyBoxBlock`` grey-box models)
through POUNCE's cyipopt-style Python interface.

The ASL ``SolverFactory('pounce')`` path used elsewhere writes an NL file and
therefore cannot consume grey-box callbacks.  POUNCE also ships a
cyipopt-compatible ``pounce.Problem(n, m, problem_obj=..., lb, ub, cl, cu)``:
the same callback protocol (``objective`` / ``gradient`` / ``constraints`` /
``jacobian`` / ``jacobianstructure`` / ``hessian`` / ``hessianstructure``) that
cyipopt expects.  This module bridges a PyNumero NLP onto that protocol, so the
decomposition inner solve, inference, and the grey-box simultaneous solve all
run on POUNCE instead of cyipopt.

The construction mirrors ``PyomoCyIpoptSolver.solve``: build a
``PyomoNLPWithGreyBoxBlocks`` (or ``PyomoNLP``) directly from the model, drive
it through the solver, then load the solution back into the model.  Crucially it
imports cyipopt nowhere, so it works on the pip-only (POUNCE/FERAL) stack.

API
---
  PounceProblemInterface(nlp)
      cyipopt-style problem_obj wrapping a PyNumero NLP.

  solve_pyomo_with_pounce(model, options=None, tee=False, return_nlp=False)
      -> (results, nlp_or_None, timing)
"""
from __future__ import annotations

import io
import sys
from typing import Optional

import numpy as np

from pyomo.common.tee import capture_output
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import Objective, minimize
from pyomo.opt import SolverResults, TerminationCondition
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
    PyomoNLPWithGreyBoxBlocks,
)

from sindae.algorithms.timing_utils import parse_pounce_output

# POUNCE returns the Ipopt ApplicationReturnStatus enum string directly in
# info['status_msg'] (unlike cyipopt, which returns the long description).
_POUNCE_TERM_COND = {
    "Solve_Succeeded": TerminationCondition.optimal,
    "Solved_To_Acceptable_Level": TerminationCondition.feasible,
    "Infeasible_Problem_Detected": TerminationCondition.infeasible,
    "Search_Direction_Becomes_Too_Small": TerminationCondition.minStepLength,
    "Diverging_Iterates": TerminationCondition.unbounded,
    "User_Requested_Stop": TerminationCondition.userInterrupt,
    "Feasible_Point_Found": TerminationCondition.feasible,
    "Maximum_Iterations_Exceeded": TerminationCondition.maxIterations,
    "Restoration_Failed": TerminationCondition.noSolution,
    "Error_In_Step_Computation": TerminationCondition.solverFailure,
    "Maximum_CpuTime_Exceeded": TerminationCondition.maxTimeLimit,
    "Maximum_WallTime_Exceeded": TerminationCondition.maxTimeLimit,
    "Not_Enough_Degrees_Of_Freedom": TerminationCondition.invalidProblem,
    "Invalid_Problem_Definition": TerminationCondition.invalidProblem,
    "Invalid_Option": TerminationCondition.error,
    "Invalid_Number_Detected": TerminationCondition.internalSolverError,
    "Unrecoverable_Exception": TerminationCondition.internalSolverError,
    "NonIpopt_Exception_Thrown": TerminationCondition.error,
    "Insufficient_Memory": TerminationCondition.resourceInterrupt,
    "Internal_Error": TerminationCondition.internalSolverError,
}


class PounceProblemInterface:
    """cyipopt-style ``problem_obj`` wrapping a PyNumero NLP for POUNCE.

    Delegates the solver callbacks to ``nlp`` (an ``AslNLP`` /
    ``PyomoNLPWithGreyBoxBlocks`` providing numpy vectors and scipy COO
    matrices).  This mirrors pyomo's ``CyIpoptNLP`` but carries none of the
    cyipopt coupling (no ``cyipopt.Problem`` base class, no cyipopt import) —
    POUNCE owns the problem object and passes bounds via ``pounce.Problem``.

    When the NLP exposes no Lagrangian Hessian (a grey-box that only provides
    Jacobians, e.g. the simultaneous-GBM model), ``hessian_available`` is False:
    the structure is empty and ``hessian`` returns an empty array.  The caller
    is responsible for selecting ``hessian_approximation='limited-memory'`` in
    that case (``solve_pyomo_with_pounce`` does so automatically).
    """

    def __init__(self, nlp):
        self._nlp = nlp

        x = nlp.init_primals()
        y = nlp.init_duals()
        if np.any(np.isnan(y)):
            y = y.copy()
            y.fill(1.0)
        self._cached_x = x.copy()
        self._cached_y = y.copy()
        self._cached_obj_factor = 1.0
        nlp.set_primals(self._cached_x)
        nlp.set_duals(self._cached_y)

        self._jac_g = nlp.evaluate_jacobian()
        try:
            self._hess_lag = nlp.evaluate_hessian_lag()
            self._hess_lower_mask = self._hess_lag.row >= self._hess_lag.col
            self.hessian_available = True
        except (AttributeError, NotImplementedError):
            self._hess_lag = None
            self._hess_lower_mask = None
            self.hessian_available = False

    # -- primal/dual caching -------------------------------------------------

    def _set_primals_if_necessary(self, x):
        if not np.array_equal(x, self._cached_x):
            self._nlp.set_primals(x)
            self._cached_x = x.copy()

    def _set_duals_if_necessary(self, y):
        if not np.array_equal(y, self._cached_y):
            self._nlp.set_duals(y)
            self._cached_y = y.copy()

    # -- cyipopt-style callbacks ---------------------------------------------

    def objective(self, x):
        self._set_primals_if_necessary(x)
        return self._nlp.evaluate_objective()

    def gradient(self, x):
        self._set_primals_if_necessary(x)
        return self._nlp.evaluate_grad_objective()

    def constraints(self, x):
        self._set_primals_if_necessary(x)
        return self._nlp.evaluate_constraints()

    def jacobianstructure(self):
        return self._jac_g.row, self._jac_g.col

    def jacobian(self, x):
        self._set_primals_if_necessary(x)
        self._nlp.evaluate_jacobian(out=self._jac_g)
        return self._jac_g.data

    def hessianstructure(self):
        if not self.hessian_available:
            return np.zeros(0), np.zeros(0)
        row = np.compress(self._hess_lower_mask, self._hess_lag.row)
        col = np.compress(self._hess_lower_mask, self._hess_lag.col)
        return row, col

    def hessian(self, x, y, obj_factor):
        # POUNCE may still call hessian() under limited-memory; return an empty
        # array consistent with the empty structure rather than raising.
        if not self.hessian_available:
            return np.zeros(0)
        self._set_primals_if_necessary(x)
        self._set_duals_if_necessary(y)
        if obj_factor != self._cached_obj_factor:
            self._nlp.set_obj_factor(obj_factor)
            self._cached_obj_factor = obj_factor
        self._nlp.evaluate_hessian_lag(out=self._hess_lag)
        return np.compress(self._hess_lower_mask, self._hess_lag.data)


def _build_nlp(model):
    """Construct the PyNumero NLP for ``model`` (grey-box aware)."""
    has_grey_box = any(
        model.component_data_objects(ExternalGreyBoxBlock, active=True)
    )
    if has_grey_box:
        return PyomoNLPWithGreyBoxBlocks(model)
    return PyomoNLP(model)


def solve_pyomo_with_pounce(
    model,
    options: Optional[dict] = None,
    tee: bool = False,
    return_nlp: bool = False,
):
    """Solve ``model`` with POUNCE's cyipopt-style interface.

    Parameters
    ----------
    model : pyomo Block/ConcreteModel
        Must carry an active objective (the inference / decomp / simultaneous
        models always do).
    options : dict, optional
        POUNCE options (e.g. ``{'tol': 1e-8, 'max_iter': 500}``).
    tee : bool
        Stream POUNCE's iteration log to stdout.
    return_nlp : bool
        Also return the populated ``PyomoNLPWithGreyBoxBlocks`` (used by the
        decomposition inner solve for the KKT gradient back-solve).

    Returns
    -------
    (results, nlp_or_None, timing)
        ``results`` is a pyomo ``SolverResults``; ``nlp`` is the populated NLP
        when ``return_nlp`` else None; ``timing`` mirrors
        ``parse_pounce_output``'s keys — ``pounceonly`` / ``nlp_evals`` /
        ``n_iter`` come from POUNCE's info dict, ``last_lgrg`` is parsed from
        the captured iteration log ('-' when no inertia regularization was
        applied at the last iteration).
    """
    import pounce

    options = dict(options or {})
    nlp = _build_nlp(model)
    problem_obj = PounceProblemInterface(nlp)

    # A grey-box without a Hessian can only be solved with L-BFGS; force it so
    # POUNCE never relies on the (empty) Hessian we hand back.
    if not problem_obj.hessian_available:
        options["hessian_approximation"] = "limited-memory"

    prob = pounce.Problem(
        n=nlp.n_primals(),
        m=nlp.n_constraints(),
        problem_obj=problem_obj,
        lb=nlp.primals_lb(),
        ub=nlp.primals_ub(),
        cl=nlp.constraints_lb(),
        cu=nlp.constraints_ub(),
    )
    for key, value in options.items():
        prob.add_option(key, value)

    timer = TicTocTimer()
    # POUNCE prints its iteration table through the C-level stdout, so
    # capture_fd is required to grab it into ``solver_log`` (the lg(rg)
    # column is parsed below); tee=True additionally streams to the console.
    solver_log = io.StringIO()
    with capture_output(
        [solver_log, sys.stdout] if tee else solver_log, capture_fd=True
    ):
        x, info = prob.solve(x0=nlp.init_primals())
    wall_time = timer.toc(None)

    # Load the solution back into the NLP and the Pyomo model so downstream
    # `pyo.value(...)` reads (inference trajectory, decomp primals/obj) work.
    x = np.asarray(x, dtype=np.float64)
    nlp.set_primals(x)
    nlp.set_duals(np.asarray(info["mult_g"], dtype=np.float64))
    nlp.load_state_into_pyomo(
        bound_multipliers=(
            np.asarray(info["mult_x_L"], dtype=np.float64),
            np.asarray(info["mult_x_U"], dtype=np.float64),
        )
    )

    results = SolverResults()
    results.problem.name = model.name
    obj = next(model.component_data_objects(Objective, active=True))
    results.problem.sense = obj.sense
    if obj.sense == minimize:
        results.problem.upper_bound = info["obj_val"]
    else:
        results.problem.lower_bound = info["obj_val"]
    results.problem.number_of_constraints = nlp.n_constraints()
    results.problem.number_of_variables = nlp.n_primals()

    results.solver.name = "pounce"
    results.solver.return_code = info["status"]
    results.solver.message = info["status_msg"]
    results.solver.wallclock_time = wall_time
    tc = _POUNCE_TERM_COND.get(info["status_msg"], TerminationCondition.other)
    results.solver.termination_condition = tc
    results.solver.status = TerminationCondition.to_solver_status(tc)

    # POUNCE's info dict carries structured timing; the lg(rg) column only
    # appears in the printed iteration table, so parse it from the captured
    # log (None when POUNCE printed no table rows).
    parsed = parse_pounce_output(solver_log.getvalue())
    pounce_timing = info.get("timing") or {}
    n_iter = info.get("iter_count")
    if n_iter is None:
        n_iter = parsed["n_iter"]
    timing = {
        "pounceonly": pounce_timing.get("overall_alg"),
        "nlp_evals": pounce_timing.get("function_evaluations_total"),
        "n_iter": n_iter,
        "last_lgrg": parsed["last_lgrg"],
    }

    return results, (nlp if return_nlp else None), timing
