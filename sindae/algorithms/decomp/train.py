"""
train.py  (decomposition approach)

train_decomp: main training function for the decomposition (GBM + KKT gradient) approach.

Workflow
--------
1. Assert norm_stats is set on problem.
2. Partition trajectories across MPI ranks.
3. Each rank builds one TrajectoryBatchSubproblem.
4. Training loop: step -> Allreduce -> Adam update.
5. Return trained SimpleMLP.

Pretraining (optional) and smoother solving are handled externally:
  see ``sindae.algorithms.pretrain`` and ``sindae.algorithms.smoother``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import pyomo.environ as pyo
from pyomo.common.timing import HierarchicalTimer

import sindae.algorithms.decomp.kkt_utils as dutils
from sindae.data_utils import InstanceData
from sindae.nn_utils import SimpleMLP, flatten_fn, make_unflatten_fn
from sindae.problem import ProblemDefinition
from sindae.algorithms.decomp.model_builder import build_decomp_model
from sindae.algorithms.decomp.subproblem import TrajectoryBatchSubproblem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class DecompConfig:
    """Hyperparameters for the decomposition (Adam + KKT gradient) training loop."""
    n_steps:               int   = 100
    lr:                    float = 1e-2
    grad_clip_norm:        float = np.inf   # set to np.inf to disable clipping
    init_slack_coef:       float = 1e2
    slack_scale:           float = 2.0
    slack_update_interval: int   = np.inf
    max_slack_coef:        float = 1e3
    mu_target:             float = 1e-10
    param_reg_coef:        float = 0.0
    subsample_frac:        float = 1.0
    # Early stopping: stop when the inner NLP is feasible (slack < slack_tol)
    # AND the data-fit component has not improved for `patience` consecutive steps.
    # Set patience=0 to disable early stopping.
    patience:              int   = 0
    slack_tol:             float = 1e-6
    # Optional learning-rate schedule: any callable (step: int) -> float.
    # step is 1-indexed to match the training loop counter.
    # When None, lr is used as a constant.
    # Compatible with optax schedules (pass the schedule object directly;
    # optax schedules accept plain Python ints).
    # Example:
    #   import optax
    #   cfg = DecompConfig(lr=1e-2,
    #       lr_schedule=optax.cosine_decay_schedule(1e-2, decay_steps=200))
    lr_schedule: Optional[Callable[[int], float]] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def adam_step(params, m, v, grad, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return params - lr * m_hat / (np.sqrt(v_hat) + eps), m, v


def make_batch_obj_fn(m: pyo.ConcreteModel):
    """
    Create a JAX objective matching the Pyomo objective in build_decomp_model.

      f = mean((norm_obs - norm_obs_target)^2) + slack_coef * mean(|sp| + |sn|)

    Returns
    -------
    obj_fn : callable  (norm_obs, sp, sn, slack_coef) -> scalar
    """
    norm_obs_target_jax = jnp.array(np.vstack(m._traj_norm_target))

    def obj_fn(norm_obs, sp, sn, slack_coef):
        data_fit   = jnp.mean((norm_obs - norm_obs_target_jax) ** 2)
        slack_term = jnp.mean(jnp.abs(sp) + jnp.abs(sn))
        return data_fit + slack_coef * slack_term

    return obj_fn


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_decomp(
    problem: ProblemDefinition,
    mlp: SimpleMLP,
    cfg: DecompConfig,
    data: InstanceData,
    smoother_model: Optional[pyo.ConcreteModel] = None,
    mpi_comm=None,
    solver_options: Optional[dict] = None,
    nlp_solver: str = 'pounce',
    linear_solver: str = 'feral',
    unfix_io: bool = True,
) -> Tuple[pyo.ConcreteModel, SimpleMLP, dict]:
    """
    Train a neural network via the decomposition (GBM + KKT gradient) approach.

    Pretraining is NOT handled here — call ``pretrain_mlp`` from
    ``sindae.algorithms.pretrain`` before this function if desired.

    Parameters
    ----------
    problem         : ProblemDefinition
    mlp             : SimpleMLP
    cfg             : DecompConfig
    data            : InstanceData
        Provides normalization statistics.  Typically extracted from the
        solved smoother via ``extract_instance_data(problem, smoother_model)``.
    smoother_model  : pyo.ConcreteModel, optional
        When provided, reused as the decomp NLP base (no rebuild /
        re-discretisation); IPOPT warm-starts from the smoother solution.
    mpi_comm        : mpi4py.MPI.Comm, optional
    solver_options  : dict, optional
        Options passed to the NLP backend, e.g. ``{'max_iter': 200, 'tol': 1e-6}``.
    nlp_solver      : str  (default ``'pounce'``; ``'cyipopt'`` / ``'ipopt'``)
        NLP solver for the inner grey-box solve.  Must be grey-box-capable.
    linear_solver   : str  (default ``'feral'``; ``'ma27'`` / ``'scipy'``)
        KKT/linear solver for the decomposition gradient back-solve.
    unfix_io        : bool  (default True)
        Unfix the NN input/output variables in the decomposition model.
        Set False for partially observed problems.

    Returns
    -------
    trained_m : pyo.ConcreteModel
        The solved decomposition NLP (the rank-local model under MPI).  Holds
        the final training iterate's trajectory; pass to
        ``extract_instance_data`` to recover states/outputs.  The trajectory
        reflects the last solve, which may differ slightly from the returned
        best-weights ``mlp``; for a trajectory strictly consistent with ``mlp``
        use ``solve_inference``.
    mlp       : SimpleMLP  (trained; rank-0 parameters are authoritative)
    history   : dict with keys obj_history, data_fit_history,
        grad_norm_history, diag_history, pouncetiming_history
    """
    jax.config.update("jax_enable_x64", True)

    # MPI setup
    if mpi_comm is not None:
        rank = mpi_comm.Get_rank()
        size = mpi_comm.Get_size()
    else:
        rank = 0
        size = 1
    is_root = rank == 0

    # Flatten initial params + broadcast to all ranks
    unflatten_fn = make_unflatten_fn(mlp)
    flat_params  = np.array(flatten_fn(mlp))
    if mpi_comm is not None:
        if not is_root:
            flat_params = np.zeros_like(flat_params)
        mpi_comm.Bcast(flat_params, root=0)
    mlp = unflatten_fn(flat_params)

    # Build VJP functions
    full_eval_fn  = dutils.make_full_eval_fn(unflatten_fn)
    model_vjp_fn  = dutils.make_model_vjp_fn(full_eval_fn)
    mixed_vjp_fn  = dutils.make_mixed_vjp_fn(full_eval_fn)
    sum_model_vjp = dutils.make_summed_model_vjp_fn(model_vjp_fn)
    sum_mixed_vjp = dutils.make_summed_mixed_vjp_fn(mixed_vjp_fn)

    # Build local batch model + subproblem
    local_traj_idxs = [i for i in range(problem.num_trajectories) if i % size == rank]

    if is_root:
        logger.info(
            f"=== Building decomp model for {len(local_traj_idxs)} trajectories "
            f"(rank {rank}) ==="
        )

    m, gbm = build_decomp_model(
        problem=problem,
        mlp=mlp,
        traj_indices=local_traj_idxs,
        data=data,
        slack_coef=cfg.init_slack_coef,
        smoother_model=smoother_model,
        unfix_io=unfix_io,
    )

    obj_fn = make_batch_obj_fn(m)

    sub = TrajectoryBatchSubproblem(
        model=m,
        gbm=gbm,
        obj_fun_jax=obj_fn,
        unflatten_fn=unflatten_fn,
        sum_model_vjp=sum_model_vjp,
        sum_mixed_vjp=sum_mixed_vjp,
        mu_target=cfg.mu_target,
        slack_coef=cfg.init_slack_coef,
        subsample_frac=cfg.subsample_frac,
        solver_options=solver_options,
        nlp_solver=nlp_solver,
        linear_solver=linear_solver,
    )

    # Training loop
    if is_root:
        logger.info(f"=== Training ({cfg.n_steps} steps) ===")

    slack_coef = cfg.init_slack_coef
    adam_m     = np.zeros_like(flat_params)
    adam_v     = np.zeros_like(flat_params)

    obj_history          = []
    data_fit_history     = []
    grad_norm_history    = []
    diag_history         = []
    pouncetiming_history = []

    timer = HierarchicalTimer()
    timer.start('training')

    # best_params: params with lowest data-fit among feasible steps (slack < tol).
    # If no feasible step occurs, fall back to lowest full objective.
    best_params      = np.zeros_like(flat_params)
    best_data_fit    = np.inf   # tracks min data-fit (feasible steps only)
    best_obj         = np.inf   # fallback: min full objective
    stagnation_count = 0        # consecutive feasible steps with no data-fit improvement

    _prev_step_t  = 0.0
    _prev_solve_t = 0.0

    for step in range(1, cfg.n_steps + 1):
        timer.start('step')
        key = jax.random.PRNGKey(step)

        try:
            obj_i, grad_i = sub.step(
                flat_params, slack_coef=slack_coef, key=key, timer=timer
            )
        except Exception as e:
            logger.error(f"[rank {rank}] step {step}: EXCEPTION: {e}")
            raise

        # Compute data-fit component: obj = data_fit + slack_coef * slack_penalty
        diag_i      = sub.get_diagnostics()
        slack_mean_i = diag_i.get('slack_mean', np.inf)
        data_fit_i   = float(obj_i) - slack_coef * float(slack_mean_i)

        # MPI Allreduce (no-op if serial)
        if mpi_comm is not None:
            from mpi4py import MPI
            global_grad      = np.zeros_like(flat_params)
            global_obj       = np.zeros(1)
            global_data_fit  = np.zeros(1)
            mpi_comm.Allreduce(grad_i,                   global_grad,     op=MPI.SUM)
            mpi_comm.Allreduce(np.array([obj_i]),        global_obj,      op=MPI.SUM)
            mpi_comm.Allreduce(np.array([data_fit_i]),   global_data_fit, op=MPI.SUM)
            # The JAX gradient is computed via jnp.mean over local time points,
            # making it ~size× larger than the 1-rank gradient; divide to correct.
            # The Pyomo objective is a sum of per-trajectory means, so the
            # Allreduce SUM already gives the correct global aggregate.
            global_grad     /= size
            global_obj       = float(global_obj[0])
            global_data_fit  = float(global_data_fit[0])
        else:
            global_grad     = grad_i
            global_obj      = obj_i
            global_data_fit = data_fit_i

        global_grad += cfg.param_reg_coef * flat_params  # L2 regularization
        # Gradient clipping
        grad_norm = float(np.linalg.norm(global_grad))
        if np.isfinite(cfg.grad_clip_norm) and grad_norm > cfg.grad_clip_norm:
            global_grad = global_grad * (cfg.grad_clip_norm / grad_norm)

        # Adam update — lr from schedule (if provided) or constant
        lr_t = float(cfg.lr_schedule(step)) if cfg.lr_schedule is not None else cfg.lr
        flat_params, adam_m, adam_v = adam_step(
            flat_params, adam_m, adam_v, global_grad, step, lr=lr_t
        )

        # Slack schedule
        if step % cfg.slack_update_interval == 0:
            slack_coef = min(slack_coef * cfg.slack_scale, cfg.max_slack_coef)

        timer.stop('step')

        _curr_step_t  = timer.get_total_time('training.step')
        _curr_solve_t = timer.get_total_time('training.step.solve')
        _wall_step    = _curr_step_t  - _prev_step_t
        _wall_solve   = _curr_solve_t - _prev_solve_t
        _prev_step_t  = _curr_step_t
        _prev_solve_t = _curr_solve_t

        obj_history.append(global_obj)
        data_fit_history.append(global_data_fit)
        grad_norm_history.append(grad_norm)

        # best_params: prefer feasible steps (slack ≈ 0), tracked by data-fit.
        # Fallback to full-objective minimum if no feasible step yet.
        feasible = slack_mean_i < cfg.slack_tol
        if feasible:
            if global_data_fit < best_data_fit:
                best_data_fit = global_data_fit
                best_params   = flat_params.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
        else:
            stagnation_count = 0   # reset: can't stop while infeasible
            if global_obj < best_obj:
                best_obj    = global_obj
                if best_data_fit == np.inf:   # no feasible step yet
                    best_params = flat_params.copy()

        if is_root:
            diag_history.append(diag_i)
            _entry = dict(sub._last_solve_info)
            _entry['wall_step']  = _wall_step
            _entry['wall_solve'] = _wall_solve
            pouncetiming_history.append(_entry)

        if is_root and step % 10 == 0:
            raw_norm = grad_norm / max(len(flat_params), 1)
            logger.info(
                f"  step {step:4d}  obj={global_obj:.4e}  data_fit={global_data_fit:.4e}"
                f"  slack={slack_mean_i:.2e}  |grad|={raw_norm:.2e}"
                f"  lr={lr_t:.2e}  slack_coef={slack_coef:.1e}"
                f"  step_s={_wall_step:.2f}"
            )

        # Early stopping: feasible + stagnation
        if cfg.patience > 0 and stagnation_count >= cfg.patience:
            if is_root:
                logger.info(
                    f"  Early stopping at step {step}: data_fit stagnated for "
                    f"{cfg.patience} consecutive feasible steps."
                )
            break

    timer.stop('training')

    if is_root:
        logger.info("=== Training complete ===")
        logger.info(str(timer))

    return m, unflatten_fn(best_params), {
        'obj_history':          obj_history,
        'data_fit_history':     data_fit_history,
        'grad_norm_history':    grad_norm_history,
        'diag_history':         diag_history,
        'pouncetiming_history': pouncetiming_history,
    }
