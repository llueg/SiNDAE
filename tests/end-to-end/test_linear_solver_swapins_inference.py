"""
KKT linear solver generalisation test (decomposition approach).

Train the MLP with the decomposition approach on the default four-tank ICs —
once per available KKT linear solver (MA27 / scipy / FERAL, selected via
``train_decomp(linear_solver=...)``) — then run inference on a validation problem
with different ICs and compare the inferred trajectories (x) and NN outputs
(z) across the linear solvers.

The linear solver only enters training through the implicit-differentiation
KKT back-solve (the hypergradient); the NLP subproblems and inference always
use the default POUNCE backend, so any difference between branches traces
back to round-off in the gradients amplified over the optax training loop.
Tolerances are therefore looser than the pounce/ipopt NLP-solver test:
this measures optimisation-path divergence, not a single solve's accuracy.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import optax
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyomo.environ as pyo

from sindae.nn_utils import SimpleMLP
from sindae import generate_data
from sindae.data_utils import extract_instance_data, InstanceData
from sindae.algorithms.smoother import solve_smoother
from sindae.algorithms.pretrain import PretrainConfig, pretrain_mlp
from sindae.algorithms.decomp.train import DecompConfig, train_decomp
from sindae.algorithms.inference import solve_inference
from sindae.algorithms.decomp.kkt_utils import NN_SLACK_POS_NAME, NN_SLACK_NEG_NAME
from sindae.example_problems import FourTankProblem
from sindae.plot_utils import plot_instance_data

from pyomo.contrib.interior_point.linalg.ma27_interface import InteriorPointMA27Interface
from pyomo.contrib.interior_point.linalg.scipy_interface import ScipyInterface

try:
    from sindae.interfaces.feral_interface import FeralInterface as _FeralInterface
    _FeralInterface()
    FeralInterface = _FeralInterface
    FERAL_AVAILABLE = True
except Exception:
    FeralInterface = None
    FERAL_AVAILABLE = False

try:
    InteriorPointMA27Interface()
    MA27_AVAILABLE = True
except Exception:
    MA27_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('pyomo').setLevel(logging.ERROR)
logging.getLogger('cyipopt').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Looser than the NLP-solver inference test: the linear solver only supplies
# training gradients, and tiny round-off differences diverge over the optax
# loop (same rationale as test_linear_solver_swapins).
X_TOL = 1e-2
Z_TOL = 1e-2

_PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')

_PLOT_NAMES = {
    'inputs':  ['x0', 'x1', 'x2', 'x3'],
    'outputs': ['z0', 'z1'],
}

_SOLVER_STYLES = {
    'ma27':  {'color': 'C0', 'ls': '-',  'lw': 2.0},
    'scipy': {'color': 'C1', 'ls': '--', 'lw': 2.0},
    'feral': {'color': 'C3', 'ls': '-.', 'lw': 2.0},
}


def _solver_available(name: str) -> bool:
    try:
        return bool(pyo.SolverFactory(name).available(exception_flag=False))
    except Exception:
        return False


# ── Training configuration (default ICs, decomp approach) ────────────────────

@dataclass
class TrainConfig:
    problem_cls:    type
    problem_kwargs: dict
    mlp_kwargs:     dict
    noise_std:      np.ndarray
    seed:           int
    nfe_train:      int
    ncp_train:      int
    smooth_coef:    float
    pretrain:       PretrainConfig
    decomp_cfg:     DecompConfig
    solver_options: dict


FOUR_TANK_TRAIN = TrainConfig(
    problem_cls    = FourTankProblem,
    problem_kwargs = dict(nfe=40, ncp=3),       # default ICs (_FT_DEFAULT_ICS)
    mlp_kwargs     = dict(in_size=4, out_size=2, widths=[32, 32],
                          activations=[jax.nn.tanh] * 2),
    noise_std      = np.array([0.05, 0.05, 0.05, 0.05]),
    seed           = 0,
    nfe_train      = 20,
    ncp_train      = 2,
    smooth_coef    = 1e1,
    pretrain       = PretrainConfig(epochs=200, batch_size=32, reg_coef=1e-2),
    decomp_cfg     = DecompConfig(
        n_steps=350, lr=5e-3, grad_clip_norm=np.inf,
        init_slack_coef=1e2, param_reg_coef=1e-2,
        lr_schedule=optax.warmup_cosine_decay_schedule(
            init_value=1e-4, peak_value=1e-2,
            warmup_steps=20, decay_steps=250, end_value=1e-4,
        ),
    ),
    solver_options = {},
)

# ── Validation problem (different ICs) ────────────────────────────────────────
# [x0, x1, x2, x3] per trajectory; x0 == x1 required for index-2 consistency.

VAL_ICS = np.array([
    [1.50, 1.50, 0.50, 2.60],
    [1.10, 1.10, 2.40, 0.90],
    [0.40, 0.40, 1.30, 0.10],
])
VAL_NFE = 25
VAL_NCP = 3

# ℓ₁-relaxed NN constraint for inference (see test_inference.py for the full
# rationale).  Mean slack is the gauge: ≈0 means the trained NN closes the
# validation dynamics exactly.
INFERENCE_SLACK_COEF = 1e3
MEAN_SLACK_TOL = 0.5


def make_validation_problem() -> FourTankProblem:
    return FourTankProblem(ics=VAL_ICS, nfe=VAL_NFE, ncp=VAL_NCP)


# ── Helpers ───────────────────────────────────────────────────────────────────

def relative_rmse(ref, other):
    return float(np.sqrt(np.mean((ref - other) ** 2))
                 / (np.sqrt(np.mean(ref ** 2)) + 1e-8))


def _flat_weights(mlp: SimpleMLP) -> np.ndarray:
    leaves = jax.tree_util.tree_leaves(mlp)
    return np.concatenate([np.asarray(l).ravel()
                           for l in leaves if hasattr(l, 'shape')])


def train_decomp_with(cfg: TrainConfig, solver_name: str):
    """
    Run the decomp training pipeline with `solver_name` as the KKT linear
    solver.  Data generation, the smoother, and pretraining are seeded, so
    branches differ only in the linear solver used for the
    implicit-differentiation gradient.

    Returns
    -------
    trained_mlp : SimpleMLP
    norm_data   : InstanceData   (smoother data — normalisation stats)
    """
    logger.info('Training (decomp) with linear solver %s', solver_name)

    problem = cfg.problem_cls(**cfg.problem_kwargs)
    mlp = SimpleMLP(**cfg.mlp_kwargs, key=jax.random.PRNGKey(cfg.seed))

    generate_data(problem=problem, noise_std=cfg.noise_std, obs_every=4, seed=cfg.seed)

    problem.nfe = cfg.nfe_train
    problem.ncp = cfg.ncp_train
    smoother_m = solve_smoother(problem, mlp, smooth_coef=cfg.smooth_coef)
    smoother_data: InstanceData = extract_instance_data(problem, smoother_m)

    mlp = pretrain_mlp(mlp, smoother_data, cfg.pretrain)

    # Select the KKT linear solver via the public API (no monkeypatching).
    _trained_m, trained_mlp, _info = train_decomp(
        problem=problem, mlp=mlp, cfg=cfg.decomp_cfg,
        data=smoother_data, smoother_model=smoother_m,
        solver_options=cfg.solver_options,
        linear_solver=solver_name,
    )

    logger.info('  ✓ trained with %s', solver_name)
    return trained_mlp, smoother_data


def _mean_abs_slack(inference_m) -> float:
    total, count = 0.0, 0
    for i in inference_m.traj_set:
        block = inference_m.trajectories[i]
        sp = getattr(block, NN_SLACK_POS_NAME)
        sn = getattr(block, NN_SLACK_NEG_NAME)
        for idx in sp:
            total += pyo.value(sp[idx]) + pyo.value(sn[idx])
            count += 1
    return total / max(count, 1)


def run_inference(trained_mlp: SimpleMLP, norm_data: InstanceData,
                  slack_coef: float = INFERENCE_SLACK_COEF) -> InstanceData:
    problem_val = make_validation_problem()
    inference_m = solve_inference(
        problem_val, trained_mlp, norm_data,
        slack_coef=slack_coef,
        solver_options={'tol': 1e-8, 'max_iter': 1000},
    )
    tc = str(inference_m._solver_result.solver.termination_condition)
    assert tc in ('optimal', 'locallyOptimal', 'feasible'), \
        f'inference solve did not converge: {tc}'

    if slack_coef > 0.0:
        mean_slack = _mean_abs_slack(inference_m)
        logger.info('  mean NN-constraint slack: %.3e (normalised z units)', mean_slack)
        assert mean_slack < MEAN_SLACK_TOL, (
            f'mean slack {mean_slack:.3e} > {MEAN_SLACK_TOL} — trained NN does '
            f'not close the validation dynamics at these ICs'
        )

    return extract_instance_data(problem_val, inference_m)


def make_validation_truth():
    """True dynamics from the validation ICs; None if the solve fails."""
    problem_val = make_validation_problem()
    truth = generate_data(problem=problem_val, noise_std=np.zeros(4),
                          obs_every=1, seed=0)
    if truth is None:
        logger.warning('Ground-truth solve at the validation ICs did not '
                       'reach optimality — proceeding without truth overlay.')
    return truth


def _save_comparison_plot(results: dict, truth=None) -> None:
    os.makedirs(_PLOTS_DIR, exist_ok=True)
    datasets = [
        (data, f'{name}-trained', dict(_SOLVER_STYLES[name]))
        for name, data in results.items()
    ]
    if truth is not None:
        datasets.insert(0, (truth, 'ground truth',
                            {'color': 'k', 'ls': ':', 'lw': 1.5}))
    fig, _ = plot_instance_data(
        datasets=datasets,
        nn_input_names=_PLOT_NAMES['inputs'],
        nn_output_names=_PLOT_NAMES['outputs'],
        groups=['inputs', 'outputs'],
        legend_placement='last',
        legend_kwargs={'fontsize': 10},
    )
    solver_names = ' vs '.join(results.keys())
    fig.suptitle(f'Four tank inference (val ICs): {solver_names} (decomp)',
                 y=1.01, fontsize=12)
    path = os.path.join(_PLOTS_DIR, 'linear_solver_inference_four_tank.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info('Plot saved → %s', path)


# ── Test ──────────────────────────────────────────────────────────────────────
@pytest.mark.slow
@pytest.mark.skipif(not _solver_available('pounce'),
                    reason='pounce binary required (data gen / smoother / '
                           'decomp subproblems / inference)')
def test_linear_solver_inference():
    """
    Train via decomp with each available KKT linear solver, infer on
    validation ICs, and require the inferred solutions to agree.
    """
    cfg = FOUR_TANK_TRAIN

    solvers = [('scipy', ScipyInterface)]
    if MA27_AVAILABLE:
        solvers = [('ma27', InteriorPointMA27Interface)] + solvers
    if FERAL_AVAILABLE:
        solvers.append(('feral', FeralInterface))
    ref_name = 'ma27' if MA27_AVAILABLE else 'scipy'

    # Ground truth at the validation ICs (informational reference only).
    val_truth = make_validation_truth()
    if val_truth is not None:
        truth_z = np.concatenate([t.nn_output for t in val_truth._trajectories])
        truth_x = np.concatenate([t.obs for t in val_truth._trajectories])

    results = {}
    weights = {}
    for solver_name, _solver_cls in solvers:
        trained_mlp, norm_data = train_decomp_with(cfg, solver_name)
        weights[solver_name] = _flat_weights(trained_mlp)
        logger.info('Inference on validation ICs (%s-trained MLP)…', solver_name)
        results[solver_name] = run_inference(trained_mlp, norm_data)

    ref = results[ref_name]
    ref_z = np.concatenate([t.nn_output for t in ref._trajectories])
    ref_x = np.concatenate([t.obs for t in ref._trajectories])

    failures = []
    for solver_name, data in results.items():
        if solver_name == ref_name:
            continue

        w_rel = (np.linalg.norm(weights[ref_name] - weights[solver_name])
                 / (np.linalg.norm(weights[ref_name]) + 1e-12))
        logger.info('trained-weight rel diff %s vs %s: %.4e',
                    ref_name, solver_name, w_rel)

        cmp_z = np.concatenate([t.nn_output for t in data._trajectories])
        cmp_x = np.concatenate([t.obs for t in data._trajectories])
        z_rmse = relative_rmse(ref_z, cmp_z)
        x_rmse = relative_rmse(ref_x, cmp_x)
        logger.info('inference %s vs %s — Z rel-RMSE: %.4e  X rel-RMSE: %.4e',
                    ref_name, solver_name, z_rmse, x_rmse)

        if z_rmse >= Z_TOL:
            failures.append(f'Z rel-RMSE {z_rmse:.4e} > {Z_TOL:.0e} '
                            f'({ref_name} vs {solver_name})')
        if x_rmse >= X_TOL:
            failures.append(f'X rel-RMSE {x_rmse:.4e} > {X_TOL:.0e} '
                            f'({ref_name} vs {solver_name})')

    # Generalisation vs truth (informational, not asserted).
    if val_truth is not None and truth_x.shape == ref_x.shape:
        for solver_name, data in results.items():
            d_z = np.concatenate([t.nn_output for t in data._trajectories])
            d_x = np.concatenate([t.obs for t in data._trajectories])
            logger.info('%s-trained vs truth — Z rel-RMSE: %.4e  X rel-RMSE: %.4e',
                        solver_name, relative_rmse(truth_z, d_z),
                        relative_rmse(truth_x, d_x))

    _save_comparison_plot(results, truth=val_truth)

    assert not failures, ('Inference mismatch between linear-solver-trained '
                          'models:\n' + '\n'.join(failures))