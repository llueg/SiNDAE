"""
four_tank.py — Four-tank hydraulic network UDE example.

Demonstrates SiNDAE on an index-2 DAE system.  The NN replaces two nonlinear
functions (pump characteristic and discharge rate) from noisy state observations.

Set METHOD at the top and run:

    python four_tank.py
"""

import logging
import os
import jax
import numpy as np
import optax
import matplotlib
matplotlib.use('Agg')

from sindae.nn_utils import SimpleMLP
from sindae import generate_data
from sindae.data_utils import extract_instance_data, InstanceData
from sindae.algorithms.smoother import solve_smoother
from sindae.algorithms.pretrain import PretrainConfig, pretrain_mlp
from sindae.algorithms.decomp.train import DecompConfig, train_decomp
from sindae.algorithms.simultaneous.train import SimultaneousConfig, solve_simultaneous

from sindae.example_problems import FourTankProblem
from sindae.plot_utils import plot_instance_data, plot_training_history

jax.config.update('jax_enable_x64', True)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('pyomo').setLevel(logging.ERROR)
logging.getLogger('cyipopt').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

METHOD   = 'decomp'   # 'decomp' | 'simul'
# simul options
USE_GBM  = False       # simul only — whether to use GBM in simultaneous training;
HESS_APPROX = 'limited-memory' # alternatively, 'exact'
if USE_GBM:
    HESS_APPROX = 'limited-memory' # GBM only implemented for jacobian
SEED     = 0
PLOTTING = True

# discretization for data generation with true model
NFE_DATA  = 40
NCP_DATA  = 3
NOISE_STD   = np.array([0.05, 0.05, 0.05, 0.05])
# discretization for training
NFE_TRAIN = 20
NCP_TRAIN = 2

# regularization coefficients
SMOOTH_COEF = 1e1
REG_COEF    = 1e-2

lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4, peak_value=1e-2, warmup_steps=20, decay_steps=250, end_value=1e-4)
decomp_cfg = DecompConfig(
    n_steps               = 350,
    lr                    = 5e-3,
    grad_clip_norm        = np.inf,
    init_slack_coef       = 1e2,
    param_reg_coef        = REG_COEF,
    lr_schedule=lr_schedule,
)
# NLP solver options for subproblem solutions
decomp_solver_opts = {}

simul_cfg = SimultaneousConfig(
    use_gbm  = USE_GBM,
    reg_coef = REG_COEF,
)
# if exact Hessian desired, disable USE_GBM and set hessian_approximation to 'exact' in simul_pounce options
# POUNCE options for simultaneous training
simul_pounce  = dict(tol=1e-6, max_iter=1000, hessian_approximation=HESS_APPROX)

_STATE_NAMES  = ['$x_0$', '$x_1$', '$x_2$', '$x_3$']
_OUTPUT_NAMES = ['$z_0$', '$z_1$']

plot_folder = os.path.join(os.path.dirname(__file__), 'plots', 'four_tank')
os.makedirs(plot_folder, exist_ok=True)

# ── Problem and MLP ───────────────────────────────────────────────────────────

problem = FourTankProblem(nfe=NFE_DATA, ncp=NCP_DATA)

mlp = SimpleMLP(
    in_size=problem.input_dim,
    out_size=problem.z_dim,
    widths=[32, 32],
    activations=[jax.nn.tanh] * 2,
    key=jax.random.PRNGKey(SEED),
)

# ── 1. Generate data ──────────────────────────────────────────────────────────

logger.info('=== 1. Generating data ===')
true_data: InstanceData = generate_data(
    problem=problem, noise_std=NOISE_STD, obs_every=4, seed=SEED,
)

# ── 2. Solve smoother ─────────────────────────────────────────────────────────

logger.info('=== 2. Solving smoother ===')
problem.nfe = NFE_TRAIN
problem.ncp = NCP_TRAIN
smoother_m = solve_smoother(problem, mlp, smooth_coef=SMOOTH_COEF)
smoother_data: InstanceData = extract_instance_data(problem, smoother_m)

# ── 3. Pretrain MLP ───────────────────────────────────────────────────────────

logger.info('=== 3. Pretraining MLP ===')
mlp = pretrain_mlp(mlp, smoother_data, PretrainConfig(epochs=200, batch_size=32, reg_coef=REG_COEF))

# ── 4. Train ──────────────────────────────────────────────────────────────────

if METHOD == 'decomp':
    logger.info('=== 4. Training (decomposition) ===')
    trained_m, mlp, history = train_decomp(
        problem=problem, mlp=mlp, cfg=decomp_cfg,
        data=smoother_data, smoother_model=smoother_m,
        solver_options=decomp_solver_opts,
    )

elif METHOD == 'simul':
    logger.info('=== 4. Solving simultaneously ===')
    trained_m, mlp = solve_simultaneous(
        problem=problem, mlp=mlp, cfg=simul_cfg,
        data=smoother_data, smoother_model=smoother_m,
        solver_options=simul_pounce, tee=True,
    )

else:
    raise ValueError(f"Unknown METHOD: {METHOD!r}")

# ── 5. Extract and plot ───────────────────────────────────────────────────────

trained_data: InstanceData = extract_instance_data(problem, trained_m)

if PLOTTING:
    datasets = [
        (true_data,     'ground truth',  {'color': 'black', 'ls': '-'}),
        (smoother_data, 'smoother init', {'color': 'C2',    'ls': '--'}),
        (trained_data,  METHOD,          {'color': 'C0',    'ls': '-.'}),
    ]

    fig_x, _ = plot_instance_data(
        datasets=datasets,
        nn_input_names=_STATE_NAMES, nn_output_names=_OUTPUT_NAMES,
        obs_times=problem.obs_times, obs_values=problem.obs_values,
        obs_names=_STATE_NAMES, groups=['inputs'], legend_placement='last',
    )
    fig_z, _ = plot_instance_data(
        datasets=datasets,
        nn_input_names=_STATE_NAMES, nn_output_names=_OUTPUT_NAMES,
        groups=['outputs'], legend_placement='last',
    )
    fig_x.savefig(os.path.join(plot_folder, f'four_tank_{METHOD}_states.pdf'))
    fig_z.savefig(os.path.join(plot_folder, f'four_tank_{METHOD}_output.pdf'))

    if METHOD == 'decomp':
        fig_h, _ = plot_training_history(history)
        fig_h.savefig(os.path.join(plot_folder, f'four_tank_{METHOD}_history.pdf'))

    logger.info('Plots saved to %s', plot_folder)
