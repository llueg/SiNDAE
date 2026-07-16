"""
End-to-end POUNCE-vs-cyipopt comparison for the three grey-box sites
(backlog item 2b): the decomposition inner solve, inference, and the
simultaneous-GBM solve.

cyipopt is the ground-truth oracle.  Each site runs the full SiNDAE pipeline
(data -> smoother -> pretrain -> solve) once with the migrated POUNCE default
and once with nlp_solver='cyipopt', and the two are required to agree:

  * decomp     : the per-step objective history (the inner solve drives the KKT
                 gradient, so an agreeing history means the populated NLP and its
                 duals match across the loop);
  * inference  : the predicted NN outputs (a deterministic square/relaxed solve);
  * simul-GBM  : both reach optimal (L-BFGS lets the two settle differently, so
                 only convergence is asserted, matching the inference comparison
                 rationale in test_pounce_ipopt_swapins_inference.py).

test_pounce_matches_cyipopt_all_examples extends this to an exhaustive sweep
over every example problem for the two default-swap sites (decomp + simul-GBM),
reporting Z/X/weight/objective deltas like test_pounce_matches_ipopt_simultaneous.

Slow + solver-dependent; lives in migration_tests (run by the user).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

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
from sindae.example_problems import (
    FourTankProblem,
    FedBatchBioreactorProblem,
    LeslieGowerProblem,
)

logging.getLogger("pyomo").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

DECOMP_OBJ_TOL = 1e-3   # per-step objective agreement over the training loop
INFERENCE_Z_TOL = 1e-6  # square/relaxed solve is deterministic -> tight

# Cross-solver objective agreement for the exhaustive comparison test. cyipopt
# and POUNCE are different interior-point implementations; on a nonconvex
# training NLP they can settle at different weights, so the solver-agnostic
# invariant is that the two objective values agree (see the OBJ_REL_TOL comment
# in test_pounce_ipopt_swapin_train.py).
OBJ_REL_TOL = 1e-2
_CONVERGED_TCS = ("optimal", "locallyOptimal", "feasible")


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
            nlp_solver=backend, solver_options={"tol": 1e-7, "max_iter": 300})
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
        m = solve_inference(problem, mlp, sdata, slack_coef=1e-5, nlp_solver=backend,
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
            data=sdata, smoother_model=fresh_smoother(), nlp_solver=backend,
            solver_options={"tol": 1e-6, "max_iter": 300})
        return str(m._solver_result.solver.termination_condition)

    assert run("cyipopt") == "optimal"
    assert run("pounce") == "optimal"


# ---------------------------------------------------------------------------
# Exhaustive POUNCE-vs-cyipopt comparison across every example problem, for the
# two grey-box sites where POUNCE replaced cyipopt as the default backend.
# ---------------------------------------------------------------------------

@dataclass
class GreyBoxConfig:
    """One example problem plus the pipeline knobs for the comparison test."""
    name: str
    problem_factory: Callable        # (nfe, ncp) -> ProblemDefinition
    mlp_factory: Callable
    noise_std: np.ndarray
    obs_every: int
    nfe_gen: int                     # fine grid for ground-truth data generation
    ncp_gen: int
    nfe_train: int                   # coarser grid for the smoother / solves
    ncp_train: int
    smooth_coef: float
    pretrain_epochs: int
    pretrain_batch: int
    pretrain_reg: float
    decomp_steps: int
    decomp_lr: float
    decomp_slack: float
    gbm_reg: float


# Compact single-trajectory configs (short solves) so the full 3-problem x
# 2-method x 2-solver sweep stays tractable while still exercising each example's
# real dynamics.  Data is generated on a finer grid than the smoother/solve grid
# (four_tank / fedbatch are stiff and infeasible to integrate on a coarse grid);
# MLP shapes and noise levels mirror the proven configs in
# test_pounce_ipopt_swapin_train.py.
_GREYBOX_CONFIGS = [
    GreyBoxConfig(
        name="leslie_gower",
        problem_factory=lambda nfe, ncp: LeslieGowerProblem(nfe=nfe, ncp=ncp),
        mlp_factory=lambda: SimpleMLP(2, 1, [8], [jax.nn.softplus],
                                      key=jax.random.PRNGKey(0)),
        noise_std=np.array([0.02, 0.02]), obs_every=3,
        nfe_gen=12, ncp_gen=2, nfe_train=12, ncp_train=2, smooth_coef=1.0,
        pretrain_epochs=100, pretrain_batch=16, pretrain_reg=1e-2,
        decomp_steps=8, decomp_lr=1e-2, decomp_slack=1e2, gbm_reg=1e-3),
    GreyBoxConfig(
        name="four_tank",
        problem_factory=lambda nfe, ncp: FourTankProblem(nfe=nfe, ncp=ncp),
        mlp_factory=lambda: SimpleMLP(4, 2, [8], [jax.nn.tanh],
                                      key=jax.random.PRNGKey(0)),
        noise_std=np.array([0.05, 0.05, 0.05, 0.05]), obs_every=4,
        nfe_gen=40, ncp_gen=3, nfe_train=20, ncp_train=2, smooth_coef=1e1,
        pretrain_epochs=100, pretrain_batch=16, pretrain_reg=1e-2,
        decomp_steps=8, decomp_lr=1e-2, decomp_slack=1e2, gbm_reg=1e-3),
    GreyBoxConfig(
        name="fedbatch",
        problem_factory=lambda nfe, ncp: FedBatchBioreactorProblem(nfe=nfe, ncp=ncp),
        mlp_factory=lambda: SimpleMLP(4, 1, [8], [jax.nn.softplus],
                                      key=jax.random.PRNGKey(0)),
        noise_std=np.array([0.05, 0.05, 0.5, 0.1]), obs_every=4,
        nfe_gen=40, ncp_gen=3, nfe_train=20, ncp_train=3, smooth_coef=1e1,
        pretrain_epochs=100, pretrain_batch=16, pretrain_reg=1e-3,
        decomp_steps=8, decomp_lr=1e-2, decomp_slack=1e2, gbm_reg=1e-3),
]


def _flat_weights(mlp: SimpleMLP) -> np.ndarray:
    """All MLP parameters as one flat vector (for cross-solver comparison)."""
    leaves = jax.tree_util.tree_leaves(mlp)
    return np.concatenate([np.asarray(l).ravel()
                           for l in leaves if hasattr(l, "shape")])


def _relative_rmse(ref: np.ndarray, other: np.ndarray) -> float:
    return float(np.sqrt(np.mean((ref - other) ** 2))
                 / (np.sqrt(np.mean(ref ** 2)) + 1e-8))


def _xz(inst) -> tuple:
    """Stacked observed states (X) and NN outputs (Z) across all trajectories."""
    x = np.concatenate([t.obs for t in inst._trajectories])
    z = np.concatenate([t.nn_output for t in inst._trajectories])
    return x, z


def _pipeline_inputs_for(cfg: GreyBoxConfig):
    """Deterministic data + pretrained MLP shared by both solver branches."""
    # Generate ground-truth data on the fine grid, then coarsen for the solves.
    problem = cfg.problem_factory(cfg.nfe_gen, cfg.ncp_gen)
    mlp0 = cfg.mlp_factory()
    generate_data(problem, noise_std=cfg.noise_std, obs_every=cfg.obs_every, seed=0)
    problem.nfe = cfg.nfe_train
    problem.ncp = cfg.ncp_train
    sdata = extract_instance_data(
        problem, solve_smoother(problem, mlp0, smooth_coef=cfg.smooth_coef))
    mlp = pretrain_mlp(mlp0, sdata, PretrainConfig(
        epochs=cfg.pretrain_epochs, batch_size=cfg.pretrain_batch,
        reg_coef=cfg.pretrain_reg))
    # A fresh smoother per build: the stage builders mutate the model they reuse.
    fresh_smoother = lambda: solve_smoother(problem, mlp0, smooth_coef=cfg.smooth_coef)
    return problem, mlp0, mlp, sdata, fresh_smoother


def _compare(method: str, name: str,
             cy_mlp, cy_inst, cy_obj,
             po_mlp, po_inst, po_obj) -> dict:
    """Compute + log the pounce-vs-cyipopt deltas for one (method, problem)."""
    cy_x, cy_z = _xz(cy_inst)
    po_x, po_z = _xz(po_inst)
    z_rmse = _relative_rmse(cy_z, po_z)
    x_rmse = _relative_rmse(cy_x, po_x)
    w_rel = float(np.linalg.norm(_flat_weights(cy_mlp) - _flat_weights(po_mlp))
                  / (np.linalg.norm(_flat_weights(cy_mlp)) + 1e-12))
    obj_rel_gap = abs(cy_obj - po_obj) / (abs(cy_obj) + 1e-16)
    logger.info(
        "[%s/%s] Z rel-RMSE %.4e  X rel-RMSE %.4e  weight rel diff %.4e  "
        "cyipopt obj %.9e  pounce obj %.9e  obj rel gap %.4e",
        method, name, z_rmse, x_rmse, w_rel, cy_obj, po_obj, obj_rel_gap)
    return dict(method=method, name=name, z_rmse=z_rmse, x_rmse=x_rmse,
                w_rel=w_rel, cy_obj=cy_obj, po_obj=po_obj, obj_rel_gap=obj_rel_gap)


def _format_table(rows: list) -> str:
    header = (f"{'method':<10} {'problem':<13} {'Z rel-RMSE':>12} "
              f"{'X rel-RMSE':>12} {'weight rel':>12} {'obj rel gap':>12}")
    lines = ["POUNCE vs cyipopt (grey-box sites)", header, "-" * len(header)]
    for r in rows:
        lines.append(f"{r['method']:<10} {r['name']:<13} {r['z_rmse']:>12.3e} "
                     f"{r['x_rmse']:>12.3e} {r['w_rel']:>12.3e} "
                     f"{r['obj_rel_gap']:>12.3e}")
    return "\n".join(lines)


@pytest.mark.slow
def test_pounce_matches_cyipopt_all_examples():
    """
    Exhaustive POUNCE-vs-cyipopt comparison across every example problem for the
    two grey-box sites where POUNCE replaced cyipopt as the default backend: the
    decomposition inner solve (``train_decomp``) and the simultaneous-GBM solve
    (``solve_simultaneous(use_gbm=True)``).  cyipopt is the previously-working
    benchmark.

    For each (problem, method) the full pipeline runs once per backend and we
    report Z rel-RMSE, X rel-RMSE, trained-weight relative difference, both
    objectives, and their relative gap — mirroring
    ``test_pounce_matches_ipopt_simultaneous``.

    Assertions follow the same solver-agnostic invariant used there:
      * decomp    : the exact grey-box KKT solve drives the whole loop, so the
                    two final objectives must agree to ``OBJ_REL_TOL`` (this is
                    the relative-gap form of ``test_decomp_pounce_matches_cyipopt``);
      * simul-GBM : L-BFGS lets the two quasi-Newton paths settle in different
                    basins, so only convergence of both solvers is required
                    (matching ``test_simultaneous_gbm_pounce_converges_like_cyipopt``);
                    weights / trajectories / objective are reported, not asserted.
    """
    rows: list = []
    failures: list = []

    for cfg in _GREYBOX_CONFIGS:
        logger.info("\n%s\nComparing pounce vs cyipopt: %s\n%s",
                    "=" * 72, cfg.name, "=" * 72)
        problem, mlp0, mlp, sdata, fresh_smoother = _pipeline_inputs_for(cfg)

        # ── Decomposition inner solve ──────────────────────────────────────────
        def run_decomp(backend):
            dcfg = DecompConfig(n_steps=cfg.decomp_steps, lr=cfg.decomp_lr,
                                init_slack_coef=cfg.decomp_slack)
            m, tmlp, hist = train_decomp(
                problem, mlp, dcfg, data=sdata, smoother_model=fresh_smoother(),
                nlp_solver=backend, solver_options={"tol": 1e-7, "max_iter": 300})
            return (tmlp, extract_instance_data(problem, m),
                    float(hist["obj_history"][-1]))

        cy = run_decomp("cyipopt")
        po = run_decomp("pounce")
        m_decomp = _compare("decomp", cfg.name, *cy, *po)
        rows.append(m_decomp)
        if m_decomp["obj_rel_gap"] >= OBJ_REL_TOL:
            failures.append(
                f"[decomp/{cfg.name}] objective rel gap "
                f"{m_decomp['obj_rel_gap']:.4e} >= {OBJ_REL_TOL:.1e}")

        # ── Simultaneous grey-box (GBM, L-BFGS) solve ──────────────────────────
        def run_gbm(backend):
            m, tmlp = solve_simultaneous(
                problem, mlp, SimultaneousConfig(use_gbm=True, reg_coef=cfg.gbm_reg),
                data=sdata, smoother_model=fresh_smoother(), nlp_solver=backend,
                solver_options={"tol": 1e-6, "max_iter": 300})
            tc = str(m._solver_result.solver.termination_condition)
            return (tmlp, extract_instance_data(problem, m),
                    float(pyo.value(m.obj)), tc)

        cy_mlp, cy_inst, cy_obj, cy_tc = run_gbm("cyipopt")
        po_mlp, po_inst, po_obj, po_tc = run_gbm("pounce")
        rows.append(_compare("simul_gbm", cfg.name,
                             cy_mlp, cy_inst, cy_obj, po_mlp, po_inst, po_obj))
        if cy_tc not in _CONVERGED_TCS:
            failures.append(f"[simul_gbm/{cfg.name}] cyipopt did not converge: {cy_tc}")
        if po_tc not in _CONVERGED_TCS:
            failures.append(f"[simul_gbm/{cfg.name}] pounce did not converge: {po_tc}")

    logger.info("\n%s", _format_table(rows))
    assert not failures, ("POUNCE-vs-cyipopt grey-box comparison failures:\n  "
                          + "\n  ".join(failures))
