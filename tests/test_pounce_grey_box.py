"""Comparison tests: the POUNCE cyipopt-style grey-box solve vs cyipopt.

cyipopt is the ground-truth reference solver for the POUNCE grey-box migration
(backlog item 2b).  These build small NN grey-box NLPs and require POUNCE to
match cyipopt:

  * exact-Hessian path (decomp / inference): primals AND duals agree tightly;
  * limited-memory path (simultaneous-GBM): the OBJECTIVE agrees.  L-BFGS lets
    the two implementations follow different quasi-Newton paths and settle in
    different points along flat directions, so primals are not compared (same
    rationale as the migration_tests inference comparison).

The solves are tiny (a 6-wide MLP at a handful of points), so they belong in the
committed fast suite rather than the slow migration_tests harness.
"""
import numpy as np
import jax
import pytest
import scipy.sparse as sps
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxModel,
    ExternalGreyBoxBlock,
)
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
    PyomoNLPWithGreyBoxBlocks,
)

jax.config.update("jax_enable_x64", True)

# POUNCE is a hard requirement of the migrated default; cyipopt is the oracle.
pytest.importorskip("pounce")
pytest.importorskip("cyipopt")

# Grey-box / return_nlp solves build an in-memory PyomoNLP, which needs the
# compiled pynumero_ASL extension (`pyomo build-extensions`).  Skip cleanly on a
# pip-only machine that lacks it rather than erroring deep in PyomoNLP.__init__.
from pyomo.contrib.pynumero.asl import AmplInterface

if not AmplInterface.available():
    pytest.skip(
        "pynumero_ASL not available; grey-box NLPs require the compiled ASL "
        "interface (run `pyomo build-extensions`)",
        allow_module_level=True,
    )

from sindae.nn_utils import SimpleMLP
from sindae.algorithms.decomp.nn_gbm import NNGreyBoxModel, _eval_all, _jac_all
from sindae.solvers import make_nlp_solver
from sindae.interfaces.pounce_interface import solve_pyomo_with_pounce


# ── Model builders ────────────────────────────────────────────────────────────

def _targets(npts, out_size):
    return np.random.default_rng(0).normal(size=(npts, out_size))


def build_exact_hessian_model(npts=4):
    """NN grey-box least-squares NLP whose grey-box supplies an exact Hessian
    (the decomp / inference regime)."""
    in_size, out_size = 2, 1
    mlp = SimpleMLP(in_size, out_size, [6], [jax.nn.softplus],
                    key=jax.random.PRNGKey(3))
    gbm = NNGreyBoxModel(mlp, npts)
    tg = _targets(npts, out_size)

    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(npts), range(in_size), bounds=(-2.0, 2.0), initialize=0.1)
    m.z = pyo.Var(range(npts), range(out_size), initialize=0.0)
    nn_in = [m.x[i, j] for i in range(npts) for j in range(in_size)]
    nn_out = [m.z[i, k] for i in range(npts) for k in range(out_size)]
    m.nn_block = ExternalGreyBoxBlock()
    m.nn_block.set_external_model(gbm, inputs=nn_in, outputs=nn_out)
    m.obj = pyo.Objective(
        expr=sum((m.z[i, k] - float(tg[i, k])) ** 2
                 for i in range(npts) for k in range(out_size))
    )
    return m


class _NoHessianNN(ExternalGreyBoxModel):
    """NN grey-box that omits evaluate_hessian_outputs (L-BFGS only) — mirrors
    NNSimulGreyBoxModel's no-Hessian contract."""

    def __init__(self, mlp, npts):
        self._mlp, self._np = mlp, npts
        self._in, self._out = mlp.in_size, mlp.out_size
        self._inp = np.zeros((npts, mlp.in_size))
        r, c = [], []
        for i in range(npts):
            rr, cc = np.meshgrid(
                np.arange(i * self._out, (i + 1) * self._out),
                np.arange(i * self._in, (i + 1) * self._in), indexing="ij")
            r.append(rr.ravel()); c.append(cc.ravel())
        self._jr, self._jc = np.concatenate(r), np.concatenate(c)

    def input_names(self):
        return [f"x{i}_{j}" for i in range(self._np) for j in range(self._in)]

    def output_names(self):
        return [f"z{i}_{k}" for i in range(self._np) for k in range(self._out)]

    def set_input_values(self, v):
        self._inp[:] = v.reshape(self._np, self._in)

    def evaluate_outputs(self):
        return np.array(_eval_all(self._mlp, self._inp), dtype=float).ravel()

    def evaluate_jacobian_outputs(self):
        d = np.array(_jac_all(self._mlp, self._inp), dtype=float).ravel()
        return sps.coo_matrix((d, (self._jr, self._jc)),
                              shape=(self._np * self._out, self._np * self._in))


def build_no_hessian_model(npts=4):
    """Same least-squares NLP but with a no-Hessian grey-box (L-BFGS regime)."""
    mlp = SimpleMLP(2, 1, [6], [jax.nn.softplus], key=jax.random.PRNGKey(3))
    tg = _targets(npts, 1)
    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(npts), range(2), bounds=(-2.0, 2.0), initialize=0.1)
    m.z = pyo.Var(range(npts), range(1), initialize=0.0)
    nn_in = [m.x[i, j] for i in range(npts) for j in range(2)]
    nn_out = [m.z[i, 0] for i in range(npts)]
    m.nn_block = ExternalGreyBoxBlock()
    m.nn_block.set_external_model(_NoHessianNN(mlp, npts), inputs=nn_in, outputs=nn_out)
    m.obj = pyo.Objective(expr=sum((m.z[i, 0] - float(tg[i, 0])) ** 2
                                   for i in range(npts)))
    return m


# ── Oracle ────────────────────────────────────────────────────────────────────

def solve_cyipopt(model, options):
    from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
        PyomoCyIpoptSolver,
    )
    res, nlp = PyomoCyIpoptSolver().solve(model, return_nlp=True, options=options)
    return res, nlp


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_pounce_matches_cyipopt_exact_hessian():
    """Exact-Hessian grey-box: POUNCE matches cyipopt in primals and duals."""
    opts = {"tol": 1e-8}
    res_cy, nlp_cy = solve_cyipopt(build_exact_hessian_model(), opts)
    res_po, nlp_po, _ = solve_pyomo_with_pounce(
        build_exact_hessian_model(), options=opts, return_nlp=True)

    assert str(res_po.solver.termination_condition) == "optimal"

    dp = np.abs(nlp_po.get_primals() - nlp_cy.get_primals()).max()
    dd = np.abs(nlp_po.get_duals() - nlp_cy.get_duals()).max()
    assert dp < 1e-6, f"primal mismatch {dp:.2e}"
    assert dd < 1e-6, f"dual mismatch {dd:.2e}"


def test_pounce_solver_returns_nlp_for_grey_box():
    """make_nlp_solver('pounce').solve(return_nlp=True) returns the populated NLP
    and loads the solution back into the Pyomo model."""
    m = build_exact_hessian_model()
    res = make_nlp_solver("pounce", {"tol": 1e-8}).solve(m, return_nlp=True)

    assert isinstance(res.nlp, PyomoNLPWithGreyBoxBlocks)
    assert str(res.result.solver.termination_condition) == "optimal"
    # Solution loaded into the model (not still at the 0.0 initialisation).
    assert np.isfinite(pyo.value(m.z[0, 0]))
    assert abs(pyo.value(m.z[0, 0])) > 0.0


def test_pounce_supports_return_nlp_flag():
    assert make_nlp_solver("pounce").supports_return_nlp is True


def test_pounce_limited_memory_matches_cyipopt_objective():
    """No-Hessian grey-box: POUNCE (forced L-BFGS) matches cyipopt's objective.

    Primals are not compared: L-BFGS settles along flat directions differently.
    """
    opts = {"tol": 1e-8}
    res_cy, _ = solve_cyipopt(
        build_no_hessian_model(),
        {**opts, "hessian_approximation": "limited-memory"})
    obj_cy = res_cy.problem.upper_bound

    # The driver must auto-force limited-memory for a no-Hessian NLP.
    res_po, nlp_po, _ = solve_pyomo_with_pounce(
        build_no_hessian_model(), options=opts, return_nlp=True)

    assert str(res_po.solver.termination_condition) == "optimal"
    assert abs(nlp_po.evaluate_objective() - obj_cy) < 1e-6, (
        f"objective mismatch: pounce {nlp_po.evaluate_objective():.8f} "
        f"vs cyipopt {obj_cy:.8f}")


def test_pounce_non_grey_box_rejects_return_nlp():
    """The ASL POUNCE path cannot return an NLP; only grey-box solves can."""
    s = make_nlp_solver("pounce")
    with pytest.raises(ValueError):
        s.solve(pyo.ConcreteModel(), return_nlp=True)
