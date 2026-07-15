"""Unit tests for the FERAL linear-solver interface (fast; no NLP solve).

Covers the configurable iterative-refinement cap (``max_steps``) in
``do_back_solve``. The reference solution comes from ``numpy.linalg.solve``
(an external reference solver); the step-count expectations come from the
``max_steps`` / ``refine_tol`` contract, not from the implementation.

Also covers the singularity guards: feral factors some rank-deficient
matrices with ``FactorStatus.SUCCESS``, so the interface must detect them
itself -- via the zero-eigenvalue count of the certified inertia at
factorization time and via the relative-residual check in ``do_back_solve``.
"""

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus

from sindae.interfaces.feral_interface import FeralInterface


def _spd_system():
    A = csc_matrix(
        np.array(
            [[4.0, 1.0, 0.0],
             [1.0, 3.0, 1.0],
             [0.0, 1.0, 2.0]]
        )
    )
    b = np.array([1.0, 2.0, 3.0])
    x_true = np.linalg.solve(A.toarray(), b)  # external reference oracle
    return A, b, x_true


def _factored(**kwargs):
    A, b, x_true = _spd_system()
    iface = FeralInterface(**kwargs)
    iface.do_symbolic_factorization(A)
    iface.do_numeric_factorization(A)
    return iface, A, b, x_true


def test_max_steps_zero_disables_refinement():
    iface, A, b, x_true = _factored(max_steps=0)
    x, res = iface.do_back_solve(b)
    assert iface.last_refine_steps == 0
    # With no refinement the result is exactly the unrefined direct solve.
    np.testing.assert_allclose(x, iface._solver.solve(b))


def test_max_steps_caps_refinement_iterations():
    # refine_tol=0 disables the convergence early-exit, so the loop runs
    # exactly max_steps times -> directly proves the cap is respected.
    iface, A, b, x_true = _factored(max_steps=3, refine_tol=0.0)
    x, res = iface.do_back_solve(b)
    assert iface.last_refine_steps == 3
    np.testing.assert_allclose(x, x_true, rtol=1e-8, atol=1e-10)


def test_default_preserves_solve_refined():
    iface, A, b, x_true = _factored()  # max_steps=None (default)
    x, res = iface.do_back_solve(b)
    assert iface.last_refine_steps is None  # manual refinement loop unused
    np.testing.assert_allclose(x, x_true, rtol=1e-8, atol=1e-10)


def test_zero_matrix_reports_singular():
    # feral factors the zero matrix with FactorStatus.SUCCESS and inertia
    # (0, 0, n); the interface must report singular from the inertia.
    A = csc_matrix(np.zeros((2, 2)))
    iface = FeralInterface()
    iface.do_symbolic_factorization(A)
    res = iface.do_numeric_factorization(A, raise_on_error=False)
    assert res.status == LinearSolverStatus.singular
    # The factorization state was cleared, so a back solve cannot silently
    # reuse the failed factors.
    with pytest.raises(RuntimeError):
        iface.do_back_solve(np.ones(2))


def test_zero_matrix_raises_by_default():
    A = csc_matrix(np.zeros((2, 2)))
    iface = FeralInterface()
    iface.do_symbolic_factorization(A)
    with pytest.raises(RuntimeError, match='singular'):
        iface.do_numeric_factorization(A)


def test_rank_deficient_back_solve_not_reported_successful():
    # feral factors this rank-1 matrix with FactorStatus.SUCCESS and a
    # *zero-free* (wrong) inertia, and back-solves it to ~1e15 garbage, so
    # only the residual check in do_back_solve can catch it.
    v = np.array([[1.0], [2.0], [3.0]])
    A = csc_matrix(v @ v.T)
    b = np.ones(3)
    iface = FeralInterface()
    iface.do_symbolic_factorization(A)
    fact_res = iface.do_numeric_factorization(A, raise_on_error=False)
    if fact_res.status == LinearSolverStatus.successful:
        x, res = iface.do_back_solve(b, raise_on_error=False)
        assert res.status != LinearSolverStatus.successful
        assert x is None
        with pytest.raises(RuntimeError, match='residual'):
            iface.do_back_solve(b)  # raise_on_error=True (default)
    else:
        assert fact_res.status == LinearSolverStatus.singular


def test_well_conditioned_solve_still_successful():
    A, b, x_true = _spd_system()
    iface = FeralInterface()
    sym_res = iface.do_symbolic_factorization(A)
    assert sym_res.status == LinearSolverStatus.successful
    num_res = iface.do_numeric_factorization(A)
    assert num_res.status == LinearSolverStatus.successful
    x, res = iface.do_back_solve(b)
    assert res.status == LinearSolverStatus.successful
    np.testing.assert_allclose(x, x_true, rtol=1e-8, atol=1e-10)
