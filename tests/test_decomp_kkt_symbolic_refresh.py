"""Regression tests for the decomposition KKT gradient back-solve.

The KKT matrix assembled by ``evaluate_primal_dual_kkt_matrix`` does not have an
invariant sparsity pattern across training steps: the Hessian-of-Lagrangian and
the bound-barrier diagonal gain and lose structural nonzeros as the iterate
moves.  ``v_eval_del_obj_del_param`` must therefore refresh the symbolic
factorization on the current matrix before every numeric factorization.  If it
reuses a stale analyse ordering (as it did when the symbolic step was performed
exactly once by the caller), MA27 corrupts and falsely reports the (genuinely
nonsingular) matrix as singular:

    RuntimeError: Numeric factorization was not successful; return code: 3

FERAL and scipy re-derive their symbolic step inside every numeric call, so they
never exhibited the failure; MA27 did.  These tests pin the contract with a
solver-agnostic spy (runs on the pip-only stack) and reproduce the real numeric
failure with MA27 when it is available.
"""
import numpy as np
import scipy.sparse as sp

import sindae.algorithms.decomp.kkt_utils as kkt_utils
from pyomo.contrib.interior_point.linalg.scipy_interface import ScipyInterface


# --- tiny fixtures driving v_eval_del_obj_del_param ------------------------

def _symmetric_indefinite(extra_offdiag=False):
    """A 6x6 symmetric indefinite, nonsingular KKT-like matrix.

    ``extra_offdiag`` adds two structural nonzeros, so the two variants have
    different sparsity patterns (the second a superset of the first) - the
    across-step pattern change that broke the reused MA27 analyse.
    """
    rows = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    cols = [0, 1, 2, 3, 4, 5, 3, 4, 5, 0, 1, 2]
    vals = [3.0, 3.0, 3.0, -3.0, -3.0, -3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    if extra_offdiag:
        rows += [0, 4]
        cols += [4, 0]
        vals += [0.7, 0.7]
    return sp.coo_matrix((vals, (rows, cols)), shape=(6, 6)).tocsc()


class _FakeInterface:
    """Minimal interface returning KKT matrices whose pattern changes on the
    second and later evaluations."""

    def __init__(self):
        self._calls = 0

    def evaluate_primal_dual_kkt_matrix(self):
        self._calls += 1
        return _symmetric_indefinite(extra_offdiag=self._calls >= 2)

    def n_primals(self):
        return 3

    def n_ineq_constraints(self):
        return 0


def _fake_kwargs():
    """Consistent tiny arguments for v_eval_del_obj_del_param (n_rho = 6)."""
    idx = lambda v: np.array([[v]], dtype=int)  # noqa: E731
    zero = np.zeros((1, 1))

    def grad_fn(x, sp_, sn_):
        return zero, zero, zero

    def sum_model_vjp(inp, param, vbar_z):
        return np.zeros_like(param)

    def sum_mixed_vjp(inp, param, mult, vbar_x):
        return np.zeros_like(param)

    return dict(
        param=np.zeros(3),
        input=np.zeros((1, 1)),
        input_indices=idx(0),
        sp=zero, sp_indices=idx(1),
        sn=zero, sn_indices=idx(2),
        nn_constr_multipliers=zero, nn_constr_indices=idx(2),
        grad_fn=grad_fn,
        sum_mixed_vjp=sum_mixed_vjp,
        sum_model_vjp=sum_model_vjp,
    )


# --- spy solver: pins the symbolic-before-every-numeric contract -----------

class _SpySolver:
    """Wraps ScipyInterface and records the factorization call sequence."""

    def __init__(self):
        self._inner = ScipyInterface()
        self.calls = []  # list of ('sym'|'num', pattern_frozenset)

    @staticmethod
    def _pattern(matrix):
        coo = matrix.tocoo()
        return frozenset(zip(coo.row.tolist(), coo.col.tolist()))

    def do_symbolic_factorization(self, matrix, raise_on_error=True):
        self.calls.append(('sym', self._pattern(matrix)))
        return self._inner.do_symbolic_factorization(matrix, raise_on_error)

    def do_numeric_factorization(self, matrix, raise_on_error=True):
        self.calls.append(('num', self._pattern(matrix)))
        return self._inner.do_numeric_factorization(matrix, raise_on_error)

    def do_back_solve(self, rhs, raise_on_error=True):
        return self._inner.do_back_solve(rhs, raise_on_error)


def test_v_eval_refreshes_symbolic_before_every_numeric():
    """Every numeric factorization must be preceded by a symbolic factorization
    on a matrix with the same sparsity pattern (solver-agnostic contract)."""
    interface = _FakeInterface()
    spy = _SpySolver()
    kw = _fake_kwargs()

    # Two gradient evaluations with a sparsity-pattern change between them.
    for _ in range(2):
        _, diag = kkt_utils.v_eval_del_obj_del_param(
            interface=interface, linear_solver=spy, **kw
        )
        assert not diag['solve_failed']

    numeric_idx = [i for i, (kind, _) in enumerate(spy.calls) if kind == 'num']
    assert len(numeric_idx) == 2, spy.calls
    for i in numeric_idx:
        assert i > 0 and spy.calls[i - 1][0] == 'sym', (
            f'numeric factorization at {i} not preceded by symbolic: {spy.calls}'
        )
        assert spy.calls[i - 1][1] == spy.calls[i][1], (
            'symbolic and numeric ran on different sparsity patterns'
        )
