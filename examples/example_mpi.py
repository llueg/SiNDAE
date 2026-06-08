"""
example_mpi.py — SiNDAE decomposition training with MPI parallelism.

Demonstrates the four-tank hydraulic network UDE trained across multiple MPI
ranks.  Each rank solves a disjoint subset of smoother subproblems concurrently;
the root rank aggregates smoother data, pretrains the NN, and then all ranks
participate in the decomposition training loop where KKT gradients are
Allreduced every step.

Usage
-----
    mpirun -n <N> python example_mpi.py
    # or, equivalently:
    python -m mpi4py.run example_mpi.py

Notes
-----
- Requires mpi4py: ``conda install mpi4py`` or ``pip install mpi4py``.
- IPOPT/HSL must be available on every node.
- The decomposition approach is the only algorithm that benefits from MPI;
  the simultaneous approach solves a single NLP and is not parallelised here.
"""

import logging

import jax
import numpy as np
from mpi4py import MPI

from sindae.nn_utils import SimpleMLP
from sindae.data_utils import extract_instance_data, InstanceData
from sindae.algorithms.smoother import solve_smoother
from sindae.algorithms.pretrain import PretrainConfig, pretrain_mlp
from sindae.algorithms.decomp.train import DecompConfig, train_decomp

from sindae.example_problems import FourTankProblem

jax.config.update('jax_enable_x64', True)

# ── MPI setup ─────────────────────────────────────────────────────────────────

comm    = MPI.COMM_WORLD
rank    = comm.Get_rank()
size    = comm.Get_size()
is_root = rank == 0

logging.basicConfig(
    level=logging.INFO if is_root else logging.WARNING,
    format=f'[rank {rank}] %(message)s',
)
logging.getLogger('pyomo').setLevel(logging.ERROR)
logging.getLogger('cyipopt').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

SEED      = 0
NUM_TRAJ  = 4      # total trajectories distributed across ranks
MLP_WIDTH = 10     # hidden layer width (two layers)

NFE_DATA  = 40
NCP_DATA  = 3
NFE_TRAIN = 20
NCP_TRAIN = 2

SMOOTH_COEF = 1e1
REG_COEF    = 1e-2
OBS_NOISE_STD = np.array([0.05, 0.05, 0.05, 0.05])
OBS_EVERY     = 4

decomp_cfg = DecompConfig(
    n_steps               = 500,
    lr                    = 5e-3,
    grad_clip_norm        = np.inf,
    init_slack_coef       = 1e2,
    slack_scale           = 2.0,
    slack_update_interval = 500,
    max_slack_coef        = 1e3,
    param_reg_coef        = REG_COEF,
    subsample_frac        = 1.0,
)
decomp_cyipopt = dict(tol=1e-6, max_iter=500, mu_init=1e-4)

# ── Initial conditions ─────────────────────────────────────────────────────────
# Cycle through the three default ICs and add uniform noise.
# Column 0 is set equal to column 1 to satisfy the algebraic
# height-equality constraint (x₀ = x₁) at t₀.

_BASE_ICS = np.array([
    [0.75, 0.75, 2.50, 0.60],
    [3.10, 3.10, 1.50, 0.50],
    [0.90, 0.90, 1.80, 1.10],
], dtype=float)
_IC_NOISE = [0.5, 0.0, 1.0, 0.3]   # per-state uniform half-width

rng  = np.random.default_rng(SEED)
n_base = len(_BASE_ICS)
ics    = np.array([_BASE_ICS[i % n_base] for i in range(NUM_TRAJ)], dtype=float)
ics   += rng.uniform(-1.0, 1.0, (NUM_TRAJ, 4)) * _IC_NOISE
ics[:, 0] = ics[:, 1]   # enforce x₀ = x₁ at t₀

# ── Problem and MLP ────────────────────────────────────────────────────────────

problem = FourTankProblem(ics=ics, nfe=NFE_DATA, ncp=NCP_DATA)

mlp = SimpleMLP(
    in_size=problem.input_dim,
    out_size=problem.z_dim,
    widths=[MLP_WIDTH, MLP_WIDTH],
    activations=[jax.nn.tanh, jax.nn.tanh],
    key=jax.random.PRNGKey(SEED),
)

# ── 1. Generate data (root only, then broadcast) ───────────────────────────────

if is_root:
    logger.info('=== 1. Generating data (%d trajectories) ===', NUM_TRAJ)
    from sindae import generate_data
    true_data: InstanceData = generate_data(
        problem=problem, noise_std=OBS_NOISE_STD, obs_every=OBS_EVERY, seed=SEED,
    )
    obs_times  = problem.obs_times
    obs_values = problem.obs_values
else:
    obs_times = obs_values = None

obs_times  = comm.bcast(obs_times,  root=0)
obs_values = comm.bcast(obs_values, root=0)
problem.obs_times  = obs_times
problem.obs_values = obs_values

# ── 2. Solve smoother in parallel ──────────────────────────────────────────────
# Each rank handles its own subset of trajectories concurrently.

problem.nfe = NFE_TRAIN
problem.ncp = NCP_TRAIN

local_idxs = [i for i in range(NUM_TRAJ) if i % size == rank]
logger.info('=== 2. Solving smoother for trajectories %s ===', local_idxs)

smoother_m         = solve_smoother(problem, mlp,
                                    traj_indices=local_idxs,
                                    smooth_coef=SMOOTH_COEF)
local_smoother_data = extract_instance_data(problem, smoother_m)

# ── 3. Gather global smoother data ─────────────────────────────────────────────
# All ranks need the full dataset for consistent normalisation stats.

local_pairs  = list(zip(local_idxs, list(local_smoother_data)))
all_pairs    = [p for rank_data in comm.allgather(local_pairs) for p in rank_data]
all_pairs.sort(key=lambda p: p[0])
smoother_data = InstanceData([td for _, td in all_pairs])

# ── 4. Pretrain MLP (root only; train_decomp broadcasts params at startup) ─────

comm.Barrier()
if is_root:
    logger.info('=== 3. Pretraining MLP ===')
    mlp = pretrain_mlp(mlp, smoother_data, PretrainConfig(epochs=200, batch_size=32,
                                                           reg_coef=REG_COEF))

# ── 5. Train decomposition (all ranks participate) ─────────────────────────────

if is_root:
    logger.info('=== 4. Training decomposition (%d ranks) ===', size)

comm.Barrier()
mlp, history = train_decomp(
    problem=problem,
    mlp=mlp,
    cfg=decomp_cfg,
    data=smoother_data,
    smoother_model=smoother_m,
    mpi_comm=comm,
    cyipopt_options=decomp_cyipopt,
)

if is_root:
    logger.info('Training complete.')
    logger.info('Final objective: %.4e', history['obj_history'][-1])
