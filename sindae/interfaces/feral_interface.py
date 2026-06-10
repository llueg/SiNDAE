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

    def __init__(self, iterative_refinement: bool = True):
        self._solver = feral.Solver()
        self._csc: Optional[feral.CscMatrix] = None   # last factorized matrix
        self._inertia: Optional[Tuple[int, int, int]] = None
        self._iterative_refinement = iterative_refinement

        self.logger = logging.getLogger('feral')
        self.logger.propagate = False

    @staticmethod
    def _to_feral(matrix: Union[spmatrix, BlockMatrix]) -> feral.CscMatrix:
        if not isspmatrix_csc(matrix):
            matrix = matrix.tocsc()
        # The primal-dual KKT matrix is symmetric and stored fully;
        # feral reads the lower triangle.
        return feral.from_scipy(matrix, symmetric='full')

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
            a = self._to_feral(matrix)
            status, inertia = self._solver.factor(a)
        except Exception as err:
            self._csc = None
            self._inertia = None
            if raise_on_error:
                raise
            self.logger.error('feral factorization raised: %s', err)
            res.status = LinearSolverStatus.error
            return res

        self._csc = a
        self._inertia = inertia.as_tuple()  # (n_pos, n_neg, n_zero)

        if status == feral.FactorStatus.SUCCESS:
            res.status = LinearSolverStatus.successful
        elif status == feral.FactorStatus.SINGULAR:
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
            if self._iterative_refinement:
                result = self._solver.solve_refined(self._csc, _rhs)
            else:
                result = self._solver.solve(_rhs)
        except Exception as err:
            if raise_on_error:
                raise
            self.logger.error('feral back solve raised: %s', err)
            return None, LinearSolverResults(LinearSolverStatus.error)

        if isinstance(rhs, BlockVector):
            _result = rhs.copy_structure()
            _result.copyfrom(result)
            result = _result

        return result, LinearSolverResults(LinearSolverStatus.successful)

    def get_inertia(self) -> Tuple[int, int, int]:
        if self._inertia is None:
            raise RuntimeError(
                'The inertia was not computed; call do_numeric_factorization first.'
            )
        return self._inertia
