"""
Pyomo interior-point linear solver interface backed by FERAL.

FERAL (https://github.com/jkitchin/feral) is a pure-Rust sparse symmetric
indefinite LDL^T solver with certified inertia counts, distributed as
prebuilt wheels (``pip install feral-solver``).  It is an unlicensed,
pip-installable alternative to the HSL MA27 backend
(``InteriorPointMA27Interface``): unlike scipy's ``splu`` it exploits
symmetry and returns the inertia needed for inertia-correction schemes.

Usage matches the other interior-point linear solver interfaces::

    solver = FeralInterface()
    solver.do_symbolic_factorization(kkt)   # no-op; see note below
    solver.do_numeric_factorization(kkt)
    x, res = solver.do_back_solve(rhs)
    n_pos, n_neg, n_zero = solver.get_inertia()

Note on symbolic factorization: feral's ``Solver`` computes the symbolic
factorization inside the first ``factor()`` call and caches it across
subsequent calls on matrices with the same sparsity pattern (the IPM
use case), so ``do_symbolic_factorization`` only validates the matrix
and returns successfully — mirroring pyomo's ``ScipyLU``.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np
from scipy.sparse import isspmatrix_csc, spmatrix

from pyomo.contrib.interior_point.linalg.base_linear_solver_interface import (
    IPLinearSolverInterface,
)
from pyomo.contrib.pynumero.linalg.base import (
    LinearSolverResults,
    LinearSolverStatus,
)
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector

import feral


class FeralInterface(IPLinearSolverInterface):
    """Direct linear solver interface around ``feral.Solver``."""

    @classmethod
    def getLoggerName(cls):
        return 'feral'

    def __init__(
        self,
        iterative_refinement: bool = True,
        max_steps: Optional[int] = None,
        refine_tol: float = 1e-12,
        residual_tol: float = 1e-4,
    ):
        """
        Parameters
        ----------
        iterative_refinement
            If ``False``, ``do_back_solve`` performs a single unrefined direct
            solve. If ``True`` (default), iterative refinement is applied.
        max_steps
            Cap on the number of iterative-refinement steps in
            ``do_back_solve``. ``None`` (default) defers to feral's built-in
            ``solve_refined`` (its internal cap). An ``int`` instead runs a
            manual refinement loop of at most ``max_steps`` steps; ``0``
            disables refinement entirely. Ignored when
            ``iterative_refinement`` is ``False``.
        refine_tol
            Relative residual tolerance for the manual refinement loop (used
            only when ``max_steps`` is an ``int``). The loop stops early once
            ``||r|| <= refine_tol * (1 + ||b||)``. A non-positive value
            disables the early exit, so the loop always runs ``max_steps``
            steps.
        residual_tol
            Acceptance threshold on the final relative residual
            ``||b - A x|| / max(||b||, tiny)`` in ``do_back_solve``. feral
            factors some numerically rank-deficient matrices with
            ``FactorStatus.SUCCESS`` and a clean (zero-free) inertia, yet
            back-solves them to garbage; a non-finite or above-threshold
            residual therefore flags the solve as failed
            (``LinearSolverStatus.error``) instead of returning it as
            successful. Refined residuals scale roughly with
            ``cond(A) * machine_eps``, so the default ``1e-4`` tolerates
            legitimately solvable systems up to conditioning ~1e13 while
            garbage solves sit at O(1). A non-positive value disables the
            check.
        """
        self._solver = feral.Solver()
        self._csc: Optional[feral.CscMatrix] = None   # last factorized matrix
        self._matrix = None                           # last matrix as scipy csc
        self._inertia: Optional[Tuple[int, int, int]] = None
        self._iterative_refinement = iterative_refinement
        self._max_steps = max_steps
        self._refine_tol = refine_tol
        self._residual_tol = residual_tol
        # Number of refinement steps the manual loop ran in the last
        # do_back_solve; None when the loop was not used (solve_refined path).
        self.last_refine_steps: Optional[int] = None

        self.logger = logging.getLogger('feral')
        self.logger.propagate = False

    @staticmethod
    def _to_scipy_csc(matrix: Union[spmatrix, BlockMatrix]):
        return matrix if isspmatrix_csc(matrix) else matrix.tocsc()

    @staticmethod
    def _to_feral(matrix: Union[spmatrix, BlockMatrix]) -> feral.CscMatrix:
        # The primal-dual KKT matrix is symmetric and stored fully;
        # feral reads the lower triangle.
        return feral.from_scipy(
            FeralInterface._to_scipy_csc(matrix), symmetric='full'
        )

    def do_symbolic_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        res = LinearSolverResults()
        try:
            self._to_feral(matrix)  # validate square/symmetric structure
            res.status = LinearSolverStatus.successful
        except Exception:
            if raise_on_error:
                raise
            res.status = LinearSolverStatus.error
        return res

    def do_numeric_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        res = LinearSolverResults()
        try:
            a_scipy = self._to_scipy_csc(matrix)
            a = feral.from_scipy(a_scipy, symmetric='full')
            status, inertia = self._solver.factor(a)
        except Exception as err:
            self._csc = None
            self._matrix = None
            self._inertia = None
            if raise_on_error:
                raise
            self.logger.error('feral factorization raised: %s', err)
            res.status = LinearSolverStatus.error
            return res

        # feral factors some singular matrices with FactorStatus.SUCCESS and
        # records the rank deficiency only in the certified inertia (e.g. a
        # zero matrix factors "successfully" with inertia (0, 0, n)), so a
        # nonzero zero-eigenvalue count is treated as singular here.
        n_pos, n_neg, n_zero = inertia.as_tuple()

        if status == feral.FactorStatus.SUCCESS and n_zero == 0:
            self._csc = a
            self._matrix = a_scipy  # for residuals in do_back_solve
            self._inertia = (n_pos, n_neg, n_zero)
            res.status = LinearSolverStatus.successful
            return res

        # Not successful: clear the factorization state so a subsequent
        # do_back_solve fails its guard instead of reusing stale factors.
        self._csc = None
        self._matrix = None
        self._inertia = None
        if status == feral.FactorStatus.SINGULAR or (
            status == feral.FactorStatus.SUCCESS and n_zero > 0
        ):
            if raise_on_error:
                raise RuntimeError('feral: matrix is singular')
            res.status = LinearSolverStatus.singular
        else:
            if raise_on_error:
                raise RuntimeError(
                    'feral factorization failed with status: ' + str(status)
                )
            res.status = LinearSolverStatus.error
        return res

    def do_back_solve(
        self, rhs: Union[np.ndarray, BlockVector], raise_on_error: bool = True
    ) -> Tuple[Optional[Union[np.ndarray, BlockVector]], LinearSolverResults]:
        if self._csc is None:
            raise RuntimeError(
                'do_back_solve called before a successful do_numeric_factorization'
            )

        if isinstance(rhs, BlockVector):
            _rhs = rhs.flatten()
        else:
            _rhs = np.asarray(rhs, dtype=np.float64)

        try:
            if not self._iterative_refinement:
                result = self._solver.solve(_rhs)
                self.last_refine_steps = 0
            elif self._max_steps is None:
                result = self._solver.solve_refined(self._csc, _rhs)
                self.last_refine_steps = None
            else:
                result, self.last_refine_steps = self._refined_solve(_rhs)
        except Exception as err:
            if raise_on_error:
                raise
            self.logger.error('feral back solve raised: %s', err)
            return None, LinearSolverResults(LinearSolverStatus.error)

        if self._residual_tol > 0.0:
            # Guard against inaccurate solves feral does not flag itself:
            # numerically rank-deficient matrices can factor with a clean
            # status and inertia but back-solve to garbage (see __init__).
            rel_residual = np.linalg.norm(_rhs - self._matrix @ result) / max(
                np.linalg.norm(_rhs), np.finfo(np.float64).tiny
            )
            if not np.isfinite(rel_residual) or rel_residual > self._residual_tol:
                msg = (
                    'feral back solve failed the residual check: relative '
                    f'residual {rel_residual:g} exceeds residual_tol '
                    f'{self._residual_tol:g}'
                )
                if raise_on_error:
                    raise RuntimeError(msg)
                self.logger.error(msg)
                return None, LinearSolverResults(LinearSolverStatus.error)

        if isinstance(rhs, BlockVector):
            _result = rhs.copy_structure()
            _result.copyfrom(result)
            result = _result

        return result, LinearSolverResults(LinearSolverStatus.successful)

    def _refined_solve(self, b: np.ndarray) -> Tuple[np.ndarray, int]:
        """Iterative refinement capped at ``self._max_steps`` steps.

        Returns the refined solution and the number of refinement steps run.
        """
        x = self._solver.solve(b)
        b_norm = np.linalg.norm(b)
        steps = 0
        for _ in range(self._max_steps):
            r = b - self._matrix @ x
            if self._refine_tol > 0.0 and (
                np.linalg.norm(r) <= self._refine_tol * (1.0 + b_norm)
            ):
                break
            x = x + self._solver.solve(r)
            steps += 1
        return x, steps

    def get_inertia(self) -> Tuple[int, int, int]:
        if self._inertia is None:
            raise RuntimeError(
                'The inertia was not computed; call do_numeric_factorization first.'
            )
        return self._inertia
