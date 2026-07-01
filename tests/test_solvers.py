"""Unit tests for the NLP / linear backend-selection layer (sindae.solvers).

These exercise the selection machinery only — construction, option routing, and
guards — not a full numerical solve (the smoke + migration suites cover solving).
Oracles come from the pyomo solver contract and the requested API spec.
"""
import pytest
import pyomo.environ as pyo

from sindae.solvers import (
    NLPSolver,
    PounceSolver,
    IpoptSolver,
    CyIpoptSolver,
    make_nlp_solver,
    make_linear_solver,
)
from sindae.interfaces.feral_interface import FeralInterface


# ── NLP solver selection ──────────────────────────────────────────────────────

def test_make_nlp_solver_returns_expected_types():
    assert isinstance(make_nlp_solver("pounce"), PounceSolver)
    assert isinstance(make_nlp_solver("ipopt"), IpoptSolver)
    assert isinstance(make_nlp_solver("cyipopt"), CyIpoptSolver)


def test_make_nlp_solver_default_is_pounce():
    assert isinstance(make_nlp_solver(), PounceSolver)


def test_make_nlp_solver_is_case_insensitive():
    assert isinstance(make_nlp_solver("POUNCE"), PounceSolver)
    assert isinstance(make_nlp_solver("CyIpopt"), CyIpoptSolver)


def test_make_nlp_solver_unknown_raises():
    with pytest.raises(ValueError):
        make_nlp_solver("nonesuch")


def test_make_nlp_solver_passes_through_instance():
    s = make_nlp_solver("pounce")
    assert make_nlp_solver(s) is s


def test_nlpsolver_is_abstract():
    with pytest.raises(TypeError):
        NLPSolver()


def test_solver_capability_flags():
    assert make_nlp_solver("pounce").is_cyipopt is False
    assert make_nlp_solver("ipopt").is_cyipopt is False
    assert make_nlp_solver("cyipopt").is_cyipopt is True

    # POUNCE returns the NLP for grey-box models (its cyipopt-style path).
    assert make_nlp_solver("pounce").supports_return_nlp is True
    assert make_nlp_solver("cyipopt").supports_return_nlp is True


def test_names():
    assert make_nlp_solver("pounce").name == "pounce"
    assert make_nlp_solver("ipopt").name == "ipopt"
    assert make_nlp_solver("cyipopt").name == "cyipopt"


def test_asl_options_route_to_solver_options():
    s = make_nlp_solver("pounce", options={"tol": 1e-7, "max_iter": 123})
    assert s.pyomo_solver.options["tol"] == 1e-7
    assert s.pyomo_solver.options["max_iter"] == 123


def test_cyipopt_options_route_to_config_options():
    s = make_nlp_solver("cyipopt", options={"tol": 1e-7, "max_iter": 123})
    assert s.pyomo_solver.config.options["tol"] == 1e-7
    assert s.pyomo_solver.config.options["max_iter"] == 123


def test_asl_solver_rejects_return_nlp():
    s = make_nlp_solver("pounce")
    with pytest.raises(ValueError):
        s.solve(pyo.ConcreteModel(), return_nlp=True)


# ── Linear (KKT) solver selection ─────────────────────────────────────────────

def test_make_linear_solver_default_is_feral():
    assert isinstance(make_linear_solver(), FeralInterface)


def test_make_linear_solver_feral_and_scipy():
    from pyomo.contrib.interior_point.linalg.scipy_interface import ScipyInterface

    assert isinstance(make_linear_solver("feral"), FeralInterface)
    assert isinstance(make_linear_solver("scipy"), ScipyInterface)


def test_make_linear_solver_is_case_insensitive():
    assert isinstance(make_linear_solver("Feral"), FeralInterface)


def test_make_linear_solver_forwards_kwargs_to_feral():
    s = make_linear_solver("feral", max_steps=4, refine_tol=1e-10)
    assert s._max_steps == 4
    assert s._refine_tol == 1e-10


def test_make_linear_solver_unknown_raises():
    with pytest.raises(ValueError):
        make_linear_solver("nonesuch")


def test_make_linear_solver_passes_through_instance():
    s = FeralInterface()
    assert make_linear_solver(s) is s
