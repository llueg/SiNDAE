"""Unit tests for the FERAL linear-solver interface (fast; no NLP solve).

Covers the configurable iterative-refinement cap (``max_steps``) in
``do_back_solve``. The reference solution comes from ``numpy.linalg.solve``
(an external reference solver); the step-count expectations come from the
``max_steps`` / ``refine_tol`` contract, not from the implementation.
"""

import numpy as np
from scipy.sparse import csc_matrix

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
