"""
End-to-end POUNCE-vs-cyipopt comparison for the three grey-box sites
(backlog item 2b): the decomposition inner solve, inference, and the
simultaneous-GBM solve.

cyipopt is the ground-truth oracle.  Each site runs the full SiNDAE pipeline
(data -> smoother -> pretrain -> solve) once with the migrated POUNCE default
and once with backend='cyipopt', and the two are required to agree:

  * decomp     : the per-step objective history (the inner solve drives the KKT
                 gradient, so an agreeing history means the populated NLP and its
                 duals match across the loop);
  * inference  : the predicted NN outputs (a deterministic square/relaxed solve);
  * simul-GBM  : both reach optimal (L-BFGS lets the two settle differently, so
                 only convergence is asserted, matching the inference comparison
                 rationale in test_pounce_ipopt_swapins_inference.py).

Slow + solver-dependent; lives in migration_tests (run by the user).
"""
from __future__ import annotations

import logging

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
import pyomo.environ as pyo

from sindae import SimpleMLP, generate_data, extract_instance_data
from sindae.algorithms.smoother import solve_smoother
from sindae.algorithms.pretrain import PretrainConfig, pretrain_mlp
from sindae.algorithms.simultaneous.train import SimultaneousConfig, solve_simultaneous
from sindae.algorithms.decomp.train import DecompConfig, train_decomp
from sindae.algorithms.inference import solve_inference
from sindae.example_problems import LeslieGowerProblem

logging.getLogger("pyomo").setLevel(logging.ERROR)

DECOMP_OBJ_TOL = 1e-3   # per-step objective agreement over the training loop
INFERENCE_Z_TOL = 1e-6  # square/relaxed solve is deterministic -> tight


def _solver_available(name: str) -> bool:
    try:
        return bool(pyo.SolverFactory(name).available(exception_flag=False))
    except Exception:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _solver_available("pounce"),
                       reason="pounce not available"),
    pytest.mark.skipif(not _solver_available("cyipopt"),
                       reason="cyipopt (oracle) not available"),
]


def _pipeline_inputs():
    """Deterministic data + pretrained MLP shared by both solver branches."""
    problem = LeslieGowerProblem(nfe=12, ncp=2)
    mlp0 = SimpleMLP(2, 1, [8], [jax.nn.softplus], key=jax.random.PRNGKey(0))
    generate_data(problem, noise_std=np.array([0.02, 0.02]), obs_every=3, seed=0)
    sdata = extract_instance_data(
        problem, solve_smoother(problem, mlp0, smooth_coef=1.0))
    mlp = pretrain_mlp(
        mlp0, sdata, PretrainConfig(epochs=100, batch_size=16, reg_coef=1e-2))
    # A fresh smoother per build: the stage builders mutate the model they reuse.
    fresh_smoother = lambda: solve_smoother(problem, mlp0, smooth_coef=1.0)
    return problem, mlp0, mlp, sdata, fresh_smoother


def test_decomp_pounce_matches_cyipopt():
    problem, mlp0, mlp, sdata, fresh_smoother = _pipeline_inputs()

    def run(backend):
        cfg = DecompConfig(n_steps=8, lr=1e-2, init_slack_coef=1e2)
        _, _, hist = train_decomp(
            problem, mlp, cfg, data=sdata, smoother_model=fresh_smoother(),
            backend=backend, solver_options={"tol": 1e-7, "max_iter": 300})
        return np.array(hist["obj_history"])

    obj_cy = run("cyipopt")
    obj_po = run("pounce")
    max_diff = float(np.abs(obj_po - obj_cy).max())
    assert max_diff < DECOMP_OBJ_TOL, (
        f"decomp obj_history mismatch (max {max_diff:.2e}):\n"
        f"  cyipopt={obj_cy}\n  pounce ={obj_po}")


def test_inference_pounce_matches_cyipopt():
    problem, mlp0, mlp, sdata, _ = _pipeline_inputs()

    def infer(backend):
        m = solve_inference(problem, mlp, sdata, slack_coef=1e-5, backend=backend,
                            solver_options={"tol": 1e-8, "max_iter": 500})
        assert str(m._solver_result.solver.termination_condition) == "optimal"
        d = extract_instance_data(problem, m)
        return np.concatenate([t.nn_output for t in d._trajectories])

    z_diff = float(np.abs(infer("pounce") - infer("cyipopt")).max())
    assert z_diff < INFERENCE_Z_TOL, f"inference z mismatch {z_diff:.2e}"


def test_simultaneous_gbm_pounce_converges_like_cyipopt():
    problem, mlp0, mlp, sdata, fresh_smoother = _pipeline_inputs()

    def run(backend):
        m, _ = solve_simultaneous(
            problem, mlp, SimultaneousConfig(use_gbm=True, reg_coef=1e-3),
            data=sdata, smoother_model=fresh_smoother(), backend=backend,
            pounce_options={"tol": 1e-6, "max_iter": 300})
        return str(m._solver_result.solver.termination_condition)

    assert run("cyipopt") == "optimal"
    assert run("pounce") == "optimal"
