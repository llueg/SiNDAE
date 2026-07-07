from __future__ import annotations
import pytest

import logging
import os
from dataclasses import dataclass
from enum import Enum

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import optax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp

from sindae.nn_utils import SimpleMLP
from sindae import generate_data
from sindae.data_utils import extract_instance_data, InstanceData
from sindae.algorithms.smoother import solve_smoother
from sindae.algorithms.pretrain import PretrainConfig, pretrain_mlp
from sindae.algorithms.decomp.train import DecompConfig, train_decomp
from sindae.example_problems import FourTankProblem, LeslieGowerProblem, FedBatchBioreactorProblem
from sindae.plot_utils import plot_instance_data

from pyomo.contrib.interior_point.linalg.ma27_interface import InteriorPointMA27Interface
from pyomo.contrib.interior_point.linalg.scipy_interface import ScipyInterface

try:
    InteriorPointMA27Interface()
    MA27_AVAILABLE = True
except Exception:
    MA27_AVAILABLE = False

try:
    from sindae.interfaces.feral_interface import FeralInterface as _FeralInterface
    _FeralInterface()
    FeralInterface = _FeralInterface
    FERAL_AVAILABLE = True
except Exception:
    FeralInterface = None
    FERAL_AVAILABLE = False

logger = logging.getLogger(__name__)

X_TOL = 1e-4
Z_TOL = 1e-4

_PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')

_PLOT_NAMES = {
    'four_tank':    {'inputs': ['x0', 'x1', 'x2', 'x3'], 'outputs': ['z0', 'z1']},
    'leslie_gower': {'inputs': ['x0', 'x1'],              'outputs': ['z0']},
    'fedbatch':     {'inputs': ['x0', 'x1', 'x2', 'x3'], 'outputs': ['z0']},
}

_SOLVER_STYLES = {
    'ma27':  {'color': 'C0', 'ls': '-',  'lw': 2.0},
    'scipy': {'color': 'C1', 'ls': '--', 'lw': 2.0},
    'feral': {'color': 'C3', 'ls': '-.', 'lw': 2.0},
}


@dataclass
class TestConfig:
    problem_cls:     type
    problem_kwargs:  dict
    mlp_kwargs:      dict
    noise_std:       np.ndarray
    seed:            int
    nfe_train:       int
    ncp_train:       int
    smooth_coef:     float
    pretrain:        PretrainConfig
    decomp_cfg:      DecompConfig
    solver_options: dict


class Configs(Enum):
    four_tank = TestConfig(
        problem_cls    = FourTankProblem,
        problem_kwargs = dict(nfe=40, ncp=3),
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

    leslie_gower = TestConfig(
        problem_cls    = LeslieGowerProblem,
        problem_kwargs = dict(nfe=60, ncp=3),
        mlp_kwargs     = dict(in_size=2, out_size=1, widths=[16, 16],
                              activations=[jax.nn.softplus] * 2),
        noise_std      = np.array([0.05, 0.05]),
        seed           = 0,
        nfe_train      = 40,
        ncp_train      = 3,
        smooth_coef    = 1.0,
        pretrain       = PretrainConfig(epochs=200, batch_size=32, reg_coef=1e-3),
        decomp_cfg     = DecompConfig(
            n_steps=300, lr=5e-3, grad_clip_norm=np.inf,
            init_slack_coef=1e1, param_reg_coef=1e-3,
        ),
        solver_options = dict(tol=1e-6, max_iter=300),
    )

    fedbatch = TestConfig(
        problem_cls    = FedBatchBioreactorProblem,
        problem_kwargs = dict(nfe=40, ncp=3),
        mlp_kwargs     = dict(in_size=4, out_size=1, widths=[20, 20],
                              activations=[jax.nn.softplus] * 2),
        noise_std      = np.array([0.05, 0.05, 0.5, 0.1]),
        seed           = 0,
        nfe_train      = 20,
        ncp_train      = 3,
        smooth_coef    = 1e1,
        pretrain       = PretrainConfig(epochs=200, batch_size=32, reg_coef=1e-3),
        decomp_cfg     = DecompConfig(
            n_steps=300, lr=5e-3, grad_clip_norm=np.inf,
            init_slack_coef=1e2, param_reg_coef=1e-3,
        ),
        solver_options = dict(tol=1e-6, max_iter=500, mu_init=1e-4),
    )


def relative_rmse(ref, other):
    return float(np.sqrt(np.mean((ref - other) ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-8))


def run_decomp_with_solver(cfg: TestConfig, solver_name: str) -> InstanceData:
    problem = cfg.problem_cls(**cfg.problem_kwargs)
    mlp = SimpleMLP(**cfg.mlp_kwargs, key=jax.random.PRNGKey(cfg.seed))

    generate_data(problem=problem, noise_std=cfg.noise_std, obs_every=4, seed=cfg.seed)

    problem.nfe = cfg.nfe_train
    problem.ncp = cfg.ncp_train
    smoother_m = solve_smoother(problem, mlp, smooth_coef=cfg.smooth_coef)
    smoother_data = extract_instance_data(problem, smoother_m)

    mlp = pretrain_mlp(mlp, smoother_data, cfg.pretrain)

    # Select the KKT linear solver via the public API (no monkeypatching).
    train_decomp(
        problem=problem, mlp=mlp, cfg=cfg.decomp_cfg,
        data=smoother_data, smoother_model=smoother_m,
        solver_options=cfg.solver_options,
        linear_solver=solver_name,
    )

    return extract_instance_data(problem, smoother_m)


def _save_comparison_plots(config_name: str, results: dict) -> None:
    os.makedirs(_PLOTS_DIR, exist_ok=True)
    names = _PLOT_NAMES[config_name]

    datasets = [
        (data, solver_name, dict(_SOLVER_STYLES[solver_name]))
        for solver_name, data in results.items()
        if data is not None
    ]

    fig, _ = plot_instance_data(
        datasets=datasets,
        nn_input_names=names['inputs'],
        nn_output_names=names['outputs'],
        groups=['inputs', 'outputs'],
        legend_placement='last',
        legend_kwargs={'fontsize': 10},
    )
    solver_names = ' vs '.join(results.keys())
    fig.suptitle(f"{config_name.replace('_', ' ').title()}: {solver_names} (decomp)", y=1.01, fontsize=12)
    path = os.path.join(_PLOTS_DIR, f'linear_solver_{config_name}.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info('Plot saved → %s', path)


def test_back_solve_consistency():
    """
    Direct check that every available interface solves the same symmetric
    indefinite (KKT-like) system to the same answer through the
    do_symbolic / do_numeric / do_back_solve protocol used in kkt_utils.

    This catches a wrong or inaccurate solve immediately, whereas the
    end-to-end test below only sees its diluted effect after hundreds of
    optimizer steps.
    """

    rng = np.random.default_rng(0)
    n = 500
    H = sp.random(n, n, density=0.02, random_state=0)
    kkt = (H + H.T).tolil()
    kkt.setdiag(np.concatenate([np.full(n // 2, 4.0), np.full(n - n // 2, -1.0)]))
    kkt = kkt.tocsc()
    rhs = rng.standard_normal(n)

    solvers = [('scipy', ScipyInterface)]
    
    solvers.append(('ma27', InteriorPointMA27Interface))
    
    solvers.append(('feral', FeralInterface))

    solutions = {}
    for name, cls in solvers:
        solver = cls()
        solver.do_symbolic_factorization(kkt)
        solver.do_numeric_factorization(kkt)
        x, res = solver.do_back_solve(rhs)
        assert res.status.name == 'successful', f'{name}: back solve failed ({res.status})'
        resid = np.linalg.norm(kkt @ x - rhs) / np.linalg.norm(rhs)
        assert resid < 1e-10, f'{name}: residual {resid:.2e}'
        solutions[name] = x

    ref = solutions['scipy']
    for name, x in solutions.items():
        rel = np.linalg.norm(x - ref) / np.linalg.norm(ref)
        assert rel < 1e-8, f'{name} vs scipy: rel diff {rel:.2e}'


def _format_rmse_table(rows: list, ref_name: str) -> str:
    """
    rows: list of (config_name, solver_name, z_rmse, x_rmse).
    Fixed-width table — problems as rows, solvers as columns, each cell
    'Z rel-RMSE / X rel-RMSE' of the final trained result vs ref_name.
    """
    configs = list(dict.fromkeys(r[0] for r in rows))
    solvers = list(dict.fromkeys(r[1] for r in rows))
    cell = {(c, s): (z, x) for c, s, z, x in rows}

    col_w = max(24, *(len(s) + 2 for s in solvers))
    name_w = max(len('problem'), *(len(c) for c in configs)) + 2

    header = f"{'problem':<{name_w}}" + ''.join(f'{s:>{col_w}}' for s in solvers)
    lines = [
        f'Final-result rel-RMSE vs reference solver ({ref_name})  [Z / X]',
        '-' * len(header), header, '-' * len(header),
    ]
    for c in configs:
        row = f'{c:<{name_w}}'
        for s in solvers:
            z, x = cell.get((c, s), (None, None))
            entry = f'{z:.2e} / {x:.2e}' if z is not None else '—'
            row += f'{entry:>{col_w}}'
        lines.append(row)
    lines.append('-' * len(header))
    return '\n'.join(lines)


@pytest.mark.slow
def test_linear_solver_swapins():
    solver_names = ['ma27', 'scipy', 'feral']
    ref_name = 'ma27' if MA27_AVAILABLE else 'scipy'

    table_rows = []   # (config_name, solver_name, z_rmse, x_rmse)
    failures = []

    for config_enum in Configs:
        config_name = config_enum.name
        cfg = config_enum.value

        logger.info('\n%s\nConfig: %s\n%s', '='*72, config_name, '='*72)

        results = {}
        for solver_name in solver_names:
            logger.info('Running %s …', solver_name)
            results[solver_name] = run_decomp_with_solver(cfg, solver_name)

        ref_data = results[ref_name]
        ref_z = np.concatenate([t.nn_output for t in ref_data._trajectories])
        ref_x = np.concatenate([t.obs for t in ref_data._trajectories])

        for solver_name, data in results.items():
            if solver_name == ref_name:
                continue
            cmp_z = np.concatenate([t.nn_output for t in data._trajectories])
            cmp_x = np.concatenate([t.obs for t in data._trajectories])

            z_rmse = relative_rmse(ref_z, cmp_z)
            x_rmse = relative_rmse(ref_x, cmp_x)
            table_rows.append((config_name, solver_name, z_rmse, x_rmse))

            if z_rmse >= Z_TOL:
                failures.append(f'[{config_name}] Z rel-RMSE {z_rmse:.4e} > {Z_TOL:.0e} ({ref_name} vs {solver_name})')
            if x_rmse >= X_TOL:
                failures.append(f'[{config_name}] X rel-RMSE {x_rmse:.4e} > {X_TOL:.0e} ({ref_name} vs {solver_name})')

        _save_comparison_plots(config_name, results)

    # Comparison table — all problems × all solvers (printed even on failure)
    table = _format_rmse_table(table_rows, ref_name)
    logger.info('\n%s', table)
    print('\n' + table)

    os.makedirs(_PLOTS_DIR, exist_ok=True)
    csv_path = os.path.join(_PLOTS_DIR, 'linear_solver_rmse.csv')
    with open(csv_path, 'w') as f:
        f.write('problem,solver,reference,z_rel_rmse,x_rel_rmse\n')
        for c, s, z, x in table_rows:
            f.write(f'{c},{s},{ref_name},{z:.6e},{x:.6e}\n')
    logger.info('RMSE table saved → %s', csv_path)

    assert not failures, 'Solver swap-in mismatches:\n' + '\n'.join(failures)
