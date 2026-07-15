"""
model_builder_utils.py

Shared building blocks used by both the decomposition and simultaneous model
builders.  Import from here rather than from either approach-specific module.

Exports
-------
Constants (variable/constraint names on trajectory blocks):
    NORM_INPUT_NAME, NORM_OUTPUT_NAME, NN_Z_NAME, NORM_OBS_NAME,
    Z_SMOOTH_NAME, Z_SMOOTH_DERIV_NAME, Z_SMOOTH_CONSTR_NAME

Block builders (call BEFORE discretisation):
    add_norm_vars_to_block
    add_normalization_to_block(block, get_input_vars, get_output_vars,
                               input_mean, input_std, output_mean, output_std, t_set)
    add_obs_normalization_to_block(block, get_obs_vars, obs_mean, obs_std, t_set, obs_dim)

Objective helper:
    build_data_fit_expr     — returns the normalised data-fit Pyomo expression

Internal helpers (used by approach model_builders):
    _compute_norm_targets
    _remove_smoother_components
    _build_fresh_base(problem, mlp, traj_indices, data)
    _add_norm_and_io_constr_post_disc  — adds norm_input/norm_output post-discretisation
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np
import pyomo.environ as pyo

from sindae.data_utils import InstanceData
from sindae.nn_utils import SimpleMLP
from sindae.problem import ProblemDefinition


# ---------------------------------------------------------------------------
# Variable / constraint name constants
# Referenced by subproblem.py and both model_builders.
# ---------------------------------------------------------------------------

NORM_INPUT_NAME        = 'norm_input'
NORM_OUTPUT_NAME       = 'norm_output'
NORM_OUTPUT_DERIV_NAME = 'd_norm_output_dt'   # kept for back-compat; no longer used by smoother
NN_Z_NAME              = 'nn_z'               # auxiliary GBM output vars (decomp)
NORM_OBS_NAME          = 'norm_obs'           # normalised observed vars (objective)

# Smoother-specific: raw (unnormalised) z variable for the smoothness penalty
Z_SMOOTH_NAME        = 'z_smooth'
Z_SMOOTH_DERIV_NAME  = 'dz_smooth_dt'
Z_SMOOTH_CONSTR_NAME = 'z_smooth_constr'


# ---------------------------------------------------------------------------
# Block builders (pre-discretisation)
# ---------------------------------------------------------------------------

def add_norm_vars_to_block(
    block,
    t_set,          # ContinuousSet (block.t, before discretisation)
    input_dim: int,
    output_dim: int,
) -> None:
    """
    Add normalised input/output Vars and their index sets to a trajectory block.

    Call BEFORE discretisation — Pyomo DAE extends these Vars automatically
    when the ContinuousSet is discretised.

    Creates on ``block``:
      ``nn_input_set``  / ``nn_output_set``         (RangeSets)
      ``norm_input[t, i]`` / ``norm_output[t, k]``  (Vars)
    """
    block.nn_input_set  = pyo.RangeSet(0, input_dim  - 1)
    block.nn_output_set = pyo.RangeSet(0, output_dim - 1)
    setattr(block, NORM_INPUT_NAME,  pyo.Var(t_set, block.nn_input_set,  initialize=0.0))
    setattr(block, NORM_OUTPUT_NAME, pyo.Var(t_set, block.nn_output_set, initialize=0.0))


def add_normalization_to_block(
    block,
    get_input_vars: Callable,   # (block, t) -> List[pyo.Var]
    get_output_vars: Callable,  # (block, t) -> List[pyo.Var]
    input_mean,                 # array-like (input_dim,)
    input_std,                  # array-like (input_dim,)
    output_mean,                # array-like (output_dim,)
    output_std,                 # array-like (output_dim,)
    t_set,                      # ContinuousSet or iterable of time points
) -> None:
    """
    Add linking constraints between raw user vars and the normalised vars.

    Can be called BEFORE discretisation with ``t_set = block.t``; Pyomo DAE
    expands the constraints to all collocation points automatically.
    Requires ``add_norm_vars_to_block`` to have been called first.

    Creates on ``block``:
      ``norm_input_constr[t,i]``  : norm_input[t,i]  == (var_i[t] - μ_i) / σ_i
      ``norm_output_constr[t,k]`` : var_k[t]         == μ_k + σ_k * norm_output[t,k]
    """
    input_mean  = np.asarray(input_mean).tolist()
    input_std   = np.asarray(input_std).tolist()
    output_mean = np.asarray(output_mean).tolist()
    output_std  = np.asarray(output_std).tolist()

    @block.Constraint(t_set, block.nn_input_set)
    def norm_input_constr(b, t, i):
        raw = get_input_vars(b, t)
        return getattr(b, NORM_INPUT_NAME)[t, i] == (raw[i] - input_mean[i]) / input_std[i]

    @block.Constraint(t_set, block.nn_output_set)
    def norm_output_constr(b, t, k):
        raw = get_output_vars(b, t)
        return raw[k] == output_mean[k] + output_std[k] * getattr(b, NORM_OUTPUT_NAME)[t, k]


def add_obs_normalization_to_block(
    block,
    get_obs_vars: Callable,   # (block, t) -> List[pyo.Var]
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    t_set,                    # ContinuousSet or iterable of time points
    obs_dim: int,
) -> None:
    """
    Add normalised observed-variable vars and linking constraints.

    Can be called BEFORE discretisation (same expansion semantics as
    ``add_normalization_to_block``).

    Creates on ``block``:
      ``norm_obs[t, i]`` + constraint: norm_obs[t,i] == (obs_i[t] - μ_i) / σ_i
    """
    obs_mean = np.asarray(obs_mean).tolist()
    obs_std  = np.asarray(obs_std).tolist()

    block.nn_obs_set = pyo.RangeSet(0, obs_dim - 1)
    setattr(block, NORM_OBS_NAME, pyo.Var(t_set, block.nn_obs_set, initialize=0.0))

    @block.Constraint(t_set, block.nn_obs_set)
    def norm_obs_constr(b, t, i):
        raw = get_obs_vars(b, t)
        return getattr(b, NORM_OBS_NAME)[t, i] == (raw[i] - obs_mean[i]) / obs_std[i]


# ---------------------------------------------------------------------------
# Objective helper
# ---------------------------------------------------------------------------

def build_data_fit_expr(
    m,
    num_traj: int,
    traj_t_sorted: List[List[float]],
    traj_norm_target,   # List[np.ndarray], shape (n_t, obs_dim) each
    obs_dim: int,
):
    """
    Return the normalised data-fit Pyomo expression:

        sum_traj  ( sum_{t,j} (norm_obs[t,j] - target[t,j])^2 / (n_t * obs_dim) )

    Usable in any objective that needs a data-fit term (smoother, decomp, simultaneous).
    """
    total = 0.0
    for ii in range(num_traj):
        block    = m.trajectories[ii]
        t_s      = traj_t_sorted[ii]
        ntgt     = traj_norm_target[ii]
        norm_obs = getattr(block, NORM_OBS_NAME)
        n_pts    = len(t_s) * obs_dim
        total += pyo.quicksum(
            (norm_obs[t, j] - float(ntgt[ti, j])) ** 2
            for ti, t in enumerate(t_s)
            for j in range(obs_dim)
        ) / n_pts
    return total


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_norm_targets(
    traj_indices: List[int],
    traj_t_sorted: List[List[float]],
    problem: ProblemDefinition,
) -> List[np.ndarray]:
    """Interpolate observed data to model times and normalise.

    Obs stats come from problem.obs_mean / problem.obs_std (computed from
    problem.obs_values), so this can be called before any InstanceData exists.
    """
    obs_mean = problem.obs_mean
    obs_std  = problem.obs_std
    obs_dim  = len(obs_mean)
    traj_norm_target = []
    for ii, gi in enumerate(traj_indices):
        t_model    = np.array(traj_t_sorted[ii])
        t_obs      = problem.obs_times[gi]
        obs_data   = problem.obs_values[gi]
        obs_interp = np.column_stack([
            np.interp(t_model, t_obs, obs_data[:, s]) for s in range(obs_dim)
        ])
        traj_norm_target.append((obs_interp - obs_mean) / obs_std)
    return traj_norm_target


def _remove_smoother_components(m: pyo.ConcreteModel, num_traj: int) -> None:
    """
    Remove smoother-specific components from ``m`` so it can be reused as a
    decomp or simultaneous model.

    Deletes:
      * The smoother objective (``m.obj``).
      * For each trajectory block: the ``z_smooth`` DerivativeVar and its
        Pyomo-generated disc_eq, the ``z_smooth_constr`` linking constraint,
        and the ``z_smooth`` Var itself.

    ``norm_obs`` and ``nn_output_set`` are preserved — they are reused by
    the decomp/simultaneous objective and post-disc norm-var addition.
    """
    m.del_component('obj')

    for ii in range(num_traj):
        block = m.trajectories[ii]
        # disc_eq — Pyomo names it after the state var (z_smooth_disc_eq) but
        # may also use the derivative name; try both.
        for cname in (Z_SMOOTH_NAME + '_disc_eq', Z_SMOOTH_DERIV_NAME + '_disc_eq'):
            if block.component(cname) is not None:
                block.del_component(cname)
                break
        for cname in (Z_SMOOTH_DERIV_NAME, Z_SMOOTH_CONSTR_NAME, Z_SMOOTH_NAME):
            if block.component(cname) is not None:
                block.del_component(cname)


def _add_norm_and_io_constr_post_disc(
    block,
    input_dim: int,
    output_dim: int,
    input_mean,         # array-like (input_dim,)
    input_std,          # array-like (input_dim,)
    output_mean,        # array-like (output_dim,)
    output_std,         # array-like (output_dim,)
    get_input_vars: Callable,
    get_output_vars: Callable,
) -> None:
    """
    Add ``norm_input`` / ``norm_output`` Vars and normalization constraints to an
    *already-discretised* trajectory block.

    Called from the smoother-reuse path of build_decomp_model /
    build_simultaneous_model after ``_remove_smoother_components``.
    ``nn_output_set`` must already exist on the block (created by the smoother).

    Initialises the new Vars to normalised values of the current raw Var solution
    so that IPOPT warm-starts correctly.
    """
    input_mean  = np.asarray(input_mean)
    input_std   = np.asarray(input_std)
    output_mean = np.asarray(output_mean)
    output_std  = np.asarray(output_std)

    # nn_input_set is new; nn_output_set already exists from smoother build.
    block.nn_input_set = pyo.RangeSet(0, input_dim - 1)
    setattr(block, NORM_INPUT_NAME,  pyo.Var(block.t, block.nn_input_set,  initialize=0.0))
    setattr(block, NORM_OUTPUT_NAME, pyo.Var(block.t, block.nn_output_set, initialize=0.0))

    # Add normalisation constraints (block.t is a finite Set post-disc).
    add_normalization_to_block(
        block, get_input_vars, get_output_vars,
        input_mean, input_std, output_mean, output_std,
        block.t,
    )

    # Warm-start: initialise from current raw solution values.
    norm_in  = getattr(block, NORM_INPUT_NAME)
    norm_out = getattr(block, NORM_OUTPUT_NAME)
    for t in block.t:
        for i, v in enumerate(get_input_vars(block, t)):
            norm_in[t, i].set_value((pyo.value(v) - input_mean[i]) / input_std[i])
            #v.set_value(1.0)
        for k, v in enumerate(get_output_vars(block, t)):
            norm_out[t, k].set_value((pyo.value(v) - output_mean[k]) / output_std[k])
            #v.set_value(0.0)


def _build_fresh_base(
    problem: ProblemDefinition,
    mlp: SimpleMLP,
    traj_indices: List[int],
    data: InstanceData,
) -> tuple:
    """
    Build a fresh (un-discretised → discretised) base model without any NN
    connectivity or objective.  Returns ``(m, traj_t_sorted, traj_norm_target)``.

    Used by the fresh-build paths of ``build_decomp_model``,
    ``build_simultaneous_model``, and ``build_simultaneous_model_gbm``.

    ``data`` provides normalization statistics (input_mean/std, output_mean/std).
    """
    num_traj   = len(traj_indices)
    input_dim  = mlp.in_size
    output_dim = mlp.out_size
    obs_dim    = len(problem.obs_mean)

    m = pyo.ConcreteModel()
    m.traj_set     = pyo.RangeSet(0, num_traj - 1)
    m.trajectories = pyo.Block(m.traj_set)

    for ii, gi in enumerate(traj_indices):
        block = m.trajectories[ii]
        problem.build_trajectory(block, gi)
        add_norm_vars_to_block(block, block.t, input_dim, output_dim)
        add_normalization_to_block(
            block, problem.get_input_vars, problem.get_output_vars,
            data.input_mean, data.input_std, data.output_mean, data.output_std,
            block.t,
        )
        add_obs_normalization_to_block(
            block, problem.get_obs_vars, problem.obs_mean, problem.obs_std, block.t, obs_dim
        )

    problem.discretize(m)
    traj_t_sorted    = [sorted(list(m.trajectories[ii].t)) for ii in range(num_traj)]
    traj_norm_target = _compute_norm_targets(traj_indices, traj_t_sorted, problem)
    m._traj_t_sorted    = traj_t_sorted
    m._traj_norm_target = traj_norm_target

    return m, traj_t_sorted, traj_norm_target


def _unfix_nn_inputs_and_outputs(m: pyo.ConcreteModel, problem: ProblemDefinition) -> None:

    for tblock in m.trajectories.values():
        for t in tblock.t:
            for v in problem.get_input_vars(tblock, t):
                v.fixed = False
            for v in problem.get_output_vars(tblock, t):
                v.fixed = False

def _add_dual_suffixes(model):
    model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zL_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    model.ipopt_zU_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
