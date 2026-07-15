"""
Solver scalability tests.

Two visualisation tests, both training the four-tank UDE at three increasing
problem sizes (the training discretisation ``nfe`` is the size knob), timing
the training step at each size, and plotting wall-time vs size so the scaling
behaviour of the compared approaches can be read off a graph.

  test_scalability_pounce_vs_ipopt
      Simultaneous (expr-writing) approach — the whole NLP is handed to one
      solver.  Times POUNCE vs IPOPT as the NLP grows.

  test_scalability_linear_solvers
      Decomposition (GBM + KKT-gradient) approach — the linear solver only
      enters through the implicit-differentiation KKT back-solve.  Times the
      MA27 / scipy / FERAL back-solvers selected via
      ``train_decomp(linear_solver=...)`` as the per-step KKT system grows.

Only the *training* call is timed (``time.perf_counter`` around
``solve_simultaneous`` / ``train_decomp``); data generation, the smoother and
pretraining are identical setup across the compared branches and run outside
the timed region.  These are timing/scaling visualisations rather than
accuracy checks, so the assertions only require that each branch produced at
least one valid measurement and that the figure was written — the artefacts
(PNG + CSV) under ``tests/plots`` are the real deliverable.
"""
from __future__ import annotations

import logging
import os
import time

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyomo.environ as pyo

from sindae.nn_utils import SimpleMLP
from sindae import generate_data
from sindae.data_utils import extract_instance_data
from sindae.algorithms.smoother import solve_smoother
from sindae.algorithms.pretrain import PretrainConfig, pretrain_mlp
from sindae.algorithms.decomp.train import DecompConfig, train_decomp
from sindae.algorithms.simultaneous.train import SimultaneousConfig, solve_simultaneous
from sindae.example_problems import FourTankProblem

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

_PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')

# ── Sweep / training configuration ───────────────────────────────────────────
# The three problems of increasing size are the same four-tank UDE solved at
# an increasing training discretisation.  Doubling nfe roughly doubles the
# number of NLP variables, so the per-size training cost should grow with it.
SIZES_NFE   = [10, 20, 40]      # training finite elements — the size knob
NCP_TRAIN   = 2                 # training collocation points (held fixed)
DATA_NFE    = 40                # data-generation discretisation (held fixed)
DATA_NCP    = 3
SEED        = 0
NOISE_STD   = np.array([0.05, 0.05, 0.05, 0.05])
SMOOTH_COEF = 1e1
MLP_KWARGS  = dict(in_size=4, out_size=2, widths=[32, 32],
                   activations=[jax.nn.tanh] * 2)
PRETRAIN    = PretrainConfig(epochs=200, batch_size=32, reg_coef=1e-2)

# Simultaneous (pounce vs ipopt) ──
REG_COEF       = 1e-2
HESS_APPROX    = 'exact'        # both solvers take exact-Newton steps → fair race

# Decomp (linear-solver swap) ──
# Fewer steps than the accuracy tests (this measures per-step solve time, not
# convergence) so the 3 sizes × N solvers sweep stays tractable.
DECOMP_N_STEPS = 30

_PV_STYLES = {
    'ipopt':  {'color': 'C0', 'ls': '-',  'lw': 2.0, 'marker': 'o'},
    'pounce': {'color': 'C1', 'ls': '--', 'lw': 2.0, 'marker': 's'},
}
_SOLVER_STYLES = {
    'ma27':  {'color': 'C0', 'ls': '-',  'lw': 2.0, 'marker': 'o'},
    'scipy': {'color': 'C1', 'ls': '--', 'lw': 2.0, 'marker': 's'},
    'feral': {'color': 'C3', 'ls': '-.', 'lw': 2.0, 'marker': '^'},
}


def _solver_available(name: str) -> bool:
    try:
        return bool(pyo.SolverFactory(name).available(exception_flag=False))
    except Exception:
        return False


def _decomp_cfg() -> DecompConfig:
    """Fresh decomp config for one training run (size-independent settings)."""
    return DecompConfig(
        n_steps=DECOMP_N_STEPS, lr=5e-3, grad_clip_norm=np.inf,
        init_slack_coef=1e2, param_reg_coef=1e-2,
    )


# ── Setup (untimed) ───────────────────────────────────────────────────────────

def _make_problem_and_mlp(nfe_train: int):
    """
    Run the seeded, solver-agnostic setup for one problem size: generate data,
    solve the smoother and pretrain the MLP.  Returns everything the timed
    training step needs.  Rebuilt per (size, solver) so reusing/​mutating a
    warm-started smoother model never contaminates a later branch.
    """
    problem = FourTankProblem(nfe=DATA_NFE, ncp=DATA_NCP)
    mlp = SimpleMLP(**MLP_KWARGS, key=jax.random.PRNGKey(SEED))

    generate_data(problem=problem, noise_std=NOISE_STD, obs_every=4, seed=SEED)

    problem.nfe = nfe_train
    problem.ncp = NCP_TRAIN
    smoother_m = solve_smoother(problem, mlp, smooth_coef=SMOOTH_COEF)
    smoother_data = extract_instance_data(problem, smoother_m)

    mlp = pretrain_mlp(mlp, smoother_data, PRETRAIN)
    return problem, mlp, smoother_data, smoother_m


# ── Timed training steps ──────────────────────────────────────────────────────

def _time_simultaneous(nfe_train: int, solver_name: str) -> float:
    """Wall-time of one simultaneous solve with the NLP routed to `solver_name`."""
    problem, mlp, data, smoother_m = _make_problem_and_mlp(nfe_train)

    # Select the NLP backend via the public API (no monkeypatching).
    t0 = time.perf_counter()
    solve_simultaneous(
        problem=problem, mlp=mlp,
        cfg=SimultaneousConfig(use_gbm=False, reg_coef=REG_COEF),
        data=data, smoother_model=smoother_m,
        pounce_options={'hessian_approximation': HESS_APPROX},
        backend=solver_name,
        tee=False,
    )
    return time.perf_counter() - t0


def _time_decomp(nfe_train: int, solver_name: str) -> float:
    """Wall-time of one decomp training run with `solver_name` as KKT back-solver."""
    problem, mlp, data, smoother_m = _make_problem_and_mlp(nfe_train)

    # Select the KKT linear solver via the public API (no monkeypatching).
    t0 = time.perf_counter()
    train_decomp(
        problem=problem, mlp=mlp, cfg=_decomp_cfg(),
        data=data, smoother_model=smoother_m, solver_options={},
        linear_solver=solver_name,
    )
    return time.perf_counter() - t0


# ── Output helpers ────────────────────────────────────────────────────────────

def _save_scaling_csv(sizes, timings: dict, fname: str) -> str:
    os.makedirs(_PLOTS_DIR, exist_ok=True)
    path = os.path.join(_PLOTS_DIR, fname)
    names = list(timings.keys())
    with open(path, 'w') as f:
        f.write('nfe,' + ','.join(names) + '\n')
        for i, s in enumerate(sizes):
            f.write(f'{s},' + ','.join(f'{timings[n][i]:.6f}' for n in names) + '\n')
    logger.info('Scaling timings saved → %s', path)
    return path


def _save_scaling_plot(sizes, timings: dict, styles: dict,
                       title: str, fname: str) -> str:
    os.makedirs(_PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, ts in timings.items():
        ax.plot(sizes, ts, label=name, **styles.get(name, {}))
    ax.set_xlabel('training finite elements (nfe)')
    ax.set_ylabel('training wall-time (s)')
    ax.set_title(title)
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    path = os.path.join(_PLOTS_DIR, fname)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info('Scaling plot saved → %s', path)
    return path


def _log_timing_table(sizes, timings: dict) -> None:
    names = list(timings.keys())
    col_w = max(12, *(len(n) + 2 for n in names))
    header = f"{'nfe':>6}" + ''.join(f'{n:>{col_w}}' for n in names)
    logger.info(header)
    logger.info('-' * len(header))
    for i, s in enumerate(sizes):
        row = f'{s:>6}' + ''.join(f'{timings[n][i]:>{col_w}.3f}' for n in names)
        logger.info(row)


def _has_valid_measurement(ts) -> bool:
    arr = np.asarray(ts, dtype=float)
    return bool(np.any(np.isfinite(arr) & (arr > 0.0)))


# ── Tests ─────────────────────────────────────────────────────────────────────
@pytest.mark.slow
@pytest.mark.skipif(not _solver_available('pounce'),
                    reason='pounce binary required (data gen / smoother / branch)')
@pytest.mark.skipif(not _solver_available('ipopt'),
                    reason='ipopt binary required (comparison branch)')
def test_scalability_pounce_vs_ipopt():
    """
    Time the simultaneous solve with POUNCE and with IPOPT across three
    increasing problem sizes and plot wall-time vs size.
    """
    approaches = ['ipopt', 'pounce']
    timings = {a: [] for a in approaches}

    for nfe in SIZES_NFE:
        logger.info('\n%s\nSize: nfe=%d (ncp=%d, %d trajectories)\n%s',
                    '=' * 60, nfe, NCP_TRAIN, len(FourTankProblem.DEFAULT_ICS), '=' * 60)
        for a in approaches:
            try:
                dt = _time_simultaneous(nfe, a)
            except Exception as e:                       # keep the sweep going
                logger.warning('  simultaneous %s @ nfe=%d failed: %s', a, nfe, e)
                dt = float('nan')
            timings[a].append(dt)
            logger.info('  [nfe=%d] %-6s simultaneous train time: %.3f s', nfe, a, dt)

    _log_timing_table(SIZES_NFE, timings)
    _save_scaling_csv(SIZES_NFE, timings, 'scalability_pounce_vs_ipopt.csv')
    plot_path = _save_scaling_plot(
        SIZES_NFE, timings, _PV_STYLES,
        'Simultaneous training scalability: POUNCE vs IPOPT (four tank)',
        'scalability_pounce_vs_ipopt.png',
    )

    for a in approaches:
        assert _has_valid_measurement(timings[a]), \
            f'no valid training-time measurement for {a}'
    assert os.path.exists(plot_path), 'scalability plot was not written'


@pytest.mark.slow
@pytest.mark.skipif(not _solver_available('pounce'),
                    reason='pounce binary required (data gen / smoother / '
                           'decomp subproblems)')
def test_scalability_linear_solvers():
    """
    Time the decomp (KKT-gradient) training with each available linear solver
    swapped into the back-solve across three increasing problem sizes and plot
    wall-time vs size.
    """
    solvers = [('scipy', ScipyInterface)]
    if MA27_AVAILABLE:
        solvers = [('ma27', InteriorPointMA27Interface)] + solvers
    if FERAL_AVAILABLE:
        solvers.append(('feral', FeralInterface))

    timings = {name: [] for name, _ in solvers}

    for nfe in SIZES_NFE:
        logger.info('\n%s\nSize: nfe=%d (ncp=%d, %d steps)\n%s',
                    '=' * 60, nfe, NCP_TRAIN, DECOMP_N_STEPS, '=' * 60)
        for name, _cls in solvers:
            try:
                dt = _time_decomp(nfe, name)
            except Exception as e:                       # keep the sweep going
                logger.warning('  decomp %s @ nfe=%d failed: %s', name, nfe, e)
                dt = float('nan')
            timings[name].append(dt)
            logger.info('  [nfe=%d] %-6s decomp train time (%d steps): %.3f s',
                        nfe, name, DECOMP_N_STEPS, dt)

    _log_timing_table(SIZES_NFE, timings)
    _save_scaling_csv(SIZES_NFE, timings, 'scalability_linear_solvers.csv')
    plot_path = _save_scaling_plot(
        SIZES_NFE, timings, _SOLVER_STYLES,
        f'Decomp (KKT) training scalability across linear solvers '
        f'(four tank, {DECOMP_N_STEPS} steps)',
        'scalability_linear_solvers.png',
    )

    for name, _ in solvers:
        assert _has_valid_measurement(timings[name]), \
            f'no valid training-time measurement for {name}'
    assert os.path.exists(plot_path), 'scalability plot was not written'
