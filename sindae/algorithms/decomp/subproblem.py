"""
subproblem.py

TrajectoryBatchSubproblem: manages one multi-trajectory NLP in the training loop.

Generalises TrajectorySubproblem (sindae/algorithms/decomp/subproblem.py) to:
  - A batch of trajectories in a single NLP (m.traj_set / m.trajectories[i])
  - Normalised NN inputs: uses standard var names norm_input[t,i] / norm_output[t,k]
  - Separate observed vars: norm_obs[t,j] for objective / KKT RHS
  - KKT gradient via v_eval_del_obj_del_param — same formula as the single-traj case,
    but with obs/obs_indices distinct from input/input_indices when obs ≠ NN inputs.

Expected model structure (produced by build_decomp_model)
---------------------------------------------------------
  m.traj_set                   : RangeSet(0, num_traj-1)
  m.trajectories[i].t          : discretized ContinuousSet
  m.trajectories[i].norm_input[t, j]   (NORM_INPUT_NAME) — NN evaluation point
  m.trajectories[i].norm_output[t, k]  (NORM_OUTPUT_NAME)
  m.trajectories[i].norm_obs[t, j]     (NORM_OBS_NAME)   — objective / KKT RHS
  m.trajectories[i].nn_z[t, k]         (NN_Z_NAME)
  m.trajectories[i].nn_slack_pos[t, k] (dutils.NN_SLACK_POS_NAME)
  m.trajectories[i].nn_slack_neg[t, k] (dutils.NN_SLACK_NEG_NAME)
  m.trajectories[i].nn_slack_constr    : norm_output - nn_z == sp - sn
  m.nn_block                   : ExternalGreyBoxBlock with NNGreyBoxModel
  m.obj                        : data-fit (norm_obs) + slack_coef * slack_term
  m.slack_coef                 : mutable Pyomo Param
  m._traj_t_sorted             : List[List[float]]
  m._traj_norm_target          : List[np.ndarray]  — normalised obs targets (obs_dim)
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pyomo.environ as pyo
from pyomo.common.timing import HierarchicalTimer

import sindae.algorithms.decomp.kkt_utils as dutils
from sindae.algorithms.model_builder_utils import NORM_INPUT_NAME, NORM_OBS_NAME
from sindae.solvers import make_linear_solver, make_nlp_solver
from sindae.interfaces.interior_point_compat import InteriorPointInterface
from sindae.interfaces.pyomo_grey_box_nlp_extended import PyomoNLPWithGreyBoxBlocksExtended

_logger = logging.getLogger(__name__)


class TrajectoryBatchSubproblem:
    """
    One batch subproblem for Neural DAE training.

    Handles a batch of trajectories in a single NLP; computes the KKT gradient
    dL/dθ via implicit differentiation after each IPOPT solve.

    The KKT gradient uses two separate sets of NLP variables:
      norm_input : NN evaluation point → sum_model_vjp / sum_mixed_vjp + v̄_x extraction
      norm_obs   : observed variables  → ∂f/∂norm_obs scattered in KKT RHS

    When obs == NN inputs, norm_obs and norm_input are linked to the same raw Pyomo vars
    but have their own NLP variables (separate norm_obs[t,j] vars + constraints).

    Parameters
    ----------
    model : pyo.ConcreteModel
        Fully-built multi-trajectory model from build_decomp_model().
    gbm : NNGreyBoxModel
    obj_fun_jax : callable  (norm_obs, sp, sn, slack_coef) -> scalar
        JAX-differentiable objective.  First argument is norm_obs (not norm_input).
    unflatten_fn : callable  flat_params -> SimpleMLP
    sum_model_vjp, sum_mixed_vjp : from decomp_utils
    mu_target : float
    slack_coef : float
    param_reg_coef : float
    subsample_frac : float
    backend : str or NLPSolver
        NLP solver for the inner grey-box solve ('pounce' default; 'cyipopt' /
        'ipopt' select alternatives).  The KKT gradient needs the populated NLP,
        so the backend must be grey-box-capable (POUNCE / cyipopt).
    linear_solver : str or IPLinearSolverInterface
        KKT/linear solver for the gradient back-solve ('feral' default, 'ma27',
        'scipy', or a pre-built interface).
    """

    def __init__(
        self,
        model,
        gbm,
        obj_fun_jax,
        unflatten_fn,
        sum_model_vjp,
        sum_mixed_vjp,
        mu_target=1e-10,
        slack_coef=1.0,
        subsample_frac=1.0,
        solver_options=None,
        backend='pounce',
        linear_solver='feral',
    ):
        self._model         = model
        self._gbm           = gbm
        self._obj_fun_jax   = obj_fun_jax
        self._unflatten_fn  = unflatten_fn
        self._sum_model_vjp = sum_model_vjp
        self._sum_mixed_vjp = sum_mixed_vjp
        self._mu_target     = mu_target
        self._slack_coef    = slack_coef
        self._subsample_frac = subsample_frac

        # Precompile gradient; obj_fun_jax first arg is norm_obs
        self._grad_obj_jit = jax.jit(jax.grad(obj_fun_jax, argnums=(0, 1, 2)))

        # POUNCE's default (monotone) mu strategy stalls on the cold index-2
        # DAE inner NLP; the adaptive strategy converges robustly and matches
        # the cyipopt reference.  Default it in for POUNCE only (cyipopt/ipopt
        # keep their defaults); a user-supplied mu_strategy always wins.
        opts = dict(solver_options or {})
        if isinstance(backend, str) and backend.lower() == 'pounce':
            opts.setdefault('mu_strategy', 'adaptive')
        self._nlp_solver    = make_nlp_solver(backend, opts)
        self._linear_solver = make_linear_solver(linear_solver)

        self._extended_nlp = None
        self._interface    = None

        # NLP indices — computed once after first solve
        self._norm_input_idx = None   # (total_points, input_dim)  — for VJPs + v̄_x
        self._norm_obs_idx   = None   # (total_points, obs_dim)    — for KKT RHS
        self._sp_idx         = None   # (total_points, output_dim)
        self._sn_idx         = None   # (total_points, output_dim)
        self._nn_constr_idx  = None   # (total_points, output_dim)

        self._last_diag   = {}
        self._last_solve_info = {}

        # Problem dimensions from model metadata
        m = model
        self._traj_list     = list(m.traj_set)
        self._num_traj      = len(self._traj_list)
        self._traj_t_sorted = m._traj_t_sorted
        self._total_points  = sum(len(ts) for ts in self._traj_t_sorted)

        block0 = m.trajectories[self._traj_list[0]]
        self._input_dim  = len(list(block0.nn_input_set))
        self._output_dim = len(list(block0.nn_output_set))
        self._obs_dim    = len(list(block0.nn_obs_set))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, flat_params, slack_coef=None, key=None, timer: HierarchicalTimer = None):
        """
        One training step: update MLP → solve → compute gradient.

        Returns
        -------
        obj  : float
        grad : np.ndarray  shape (n_params,)
        """
        def _t(name): return timer.start(name) if timer else None
        def _s(name): return timer.stop(name)  if timer else None

        if key is None:
            key = jax.random.PRNGKey(0)
        if slack_coef is None:
            slack_coef = self._slack_coef

        # 1. Update GBM with new MLP weights
        _t('update_mlp')
        self._gbm.update_mlp(self._unflatten_fn(flat_params))
        if hasattr(self._model, 'slack_coef'):
            self._model.slack_coef.set_value(slack_coef)
        _s('update_mlp')

        # 2. Solve the inner NLP; return_nlp=True gives us the populated NLP
        _t('solve')
        _res = self._nlp_solver.solve(self._model, return_nlp=True)
        solved_nlp = _res.nlp
        self._last_solve_info = _res.timing
        _lgrg = self._last_solve_info.get('last_lgrg')
        if _lgrg is not None and _lgrg != '-':
            _logger.warning(
                "Subproblem step: IPOPT last iteration has inertia regularization "
                "lg(rg)=%s — KKT gradient may be inaccurate.",
                _lgrg,
            )
        obj = float(pyo.value(self._model.obj))
        _s('solve')

        # 3. Update NLP wrapper + InteriorPointInterface
        self._update_nlp(solved_nlp, timer=timer)

        # 4. Extract current primal values
        _t('get_values')
        norm_input, norm_obs, sp, sn = self._get_current_values()
        nn_constr_mult = -self._interface.get_duals_eq()[self._nn_constr_idx]
        _s('get_values')

        self._last_diag = {
            'norm_obs': norm_obs,
            'sp': sp, 'sn': sn,
            'nn_mult': nn_constr_mult,
        }

        # 5. Gradient function — receives norm_obs as first arg
        _sc = jnp.array(slack_coef)

        def grad_fn(obs_, sp_, sn_):
            return self._grad_obj_jit(obs_, sp_, sn_, _sc)

        # 6. Compute gradient via KKT implicit differentiation
        #    input / input_indices → norm_input (NN evaluation point, VJPs, v̄_x)
        #    obs   / obs_indices   → norm_obs   (observed vars, KKT RHS)
        _t('grad_eval')
        grad, grad_diag = dutils.v_eval_del_obj_del_param(
            interface=self._interface,
            linear_solver=self._linear_solver,
            param=flat_params,
            input=norm_input,
            input_indices=self._norm_input_idx,
            obs=norm_obs,
            obs_indices=self._norm_obs_idx,
            sp=sp,
            sp_indices=self._sp_idx,
            sn=sn,
            sn_indices=self._sn_idx,
            nn_constr_multipliers=nn_constr_mult,
            nn_constr_indices=self._nn_constr_idx,
            grad_fn=grad_fn,
            sum_mixed_vjp=self._sum_mixed_vjp,
            sum_model_vjp=self._sum_model_vjp,
            subsample_frac=self._subsample_frac,
            key=key,
            timer=timer,
        )
        _s('grad_eval')

        self._last_diag.update(grad_diag)
        return obj, np.array(grad)

    def get_diagnostics(self):
        """
        Return a flat dict of scalar diagnostics from the last step().

        Keys
        ----
        nn_mult_mean_abs, nn_mult_max_abs : mean/max |λ_nn|
        slack_mean, slack_max             : mean/max (sp + sn)
        model_vjp_norm, mixed_vjp_norm    : gradient component norms
        v_bar_z_norm, v_bar_x_norm        : back-solve vector norms
        solve_failed                      : True if KKT back-solve failed
        """
        d = self._last_diag
        if not d:
            return {}
        out = {}
        if 'nn_mult' in d:
            out['nn_mult_mean_abs'] = float(np.mean(np.abs(d['nn_mult'])))
            out['nn_mult_max_abs']  = float(np.max(np.abs(d['nn_mult'])))
        if 'sp' in d and 'sn' in d:
            slack = np.abs(d['sp']) + np.abs(d['sn'])
            out['slack_mean'] = float(np.mean(slack))
            out['slack_max']  = float(np.max(slack))
        for key in ('model_vjp_norm', 'mixed_vjp_norm',
                    'v_bar_z_norm', 'v_bar_x_norm', 'solve_failed'):
            if key in d:
                out[key] = d[key]
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_nlp(self, solved_nlp, timer: HierarchicalTimer = None):
        """
        Update the extended NLP wrapper and rebuild InteriorPointInterface.

        First call: builds PyomoNLPWithGreyBoxBlocksExtended (computes sparsity).
        Subsequent calls: swap inner NLP only (reuse sparsity maps).

        Symbolic factorization of the KKT linear solver is refreshed on each
        gradient evaluation inside ``v_eval_del_obj_del_param`` (the KKT sparsity
        pattern is not invariant across steps), so it is not done here.
        """
        def _t(name): return timer.start(name) if timer else None
        def _s(name): return timer.stop(name)  if timer else None

        _t('update_nlp')
        if self._extended_nlp is None:
            _t('build_extended_nlp')
            self._extended_nlp = PyomoNLPWithGreyBoxBlocksExtended(solved_nlp)
            _s('build_extended_nlp')
        else:
            self._extended_nlp._nlp = solved_nlp

        _t('build_interface')
        self._interface = InteriorPointInterface(self._extended_nlp)
        self._interface.set_barrier_parameter(self._mu_target)
        _s('build_interface')

        if self._norm_input_idx is None:
            self._compute_indices()

        _s('update_nlp')

    def _compute_indices(self):
        """
        Compute and cache NLP primal and constraint indices.  Called once.

        Collects norm_input, norm_obs, sp, sn var references across all (traj, t)
        pairs in the same order as the GBM input/output construction in build_decomp_model.
        """
        m   = self._model
        ext = self._extended_nlp

        norm_in_vars  = []
        norm_obs_vars = []
        sp_vars       = []
        sn_vars       = []
        for i in self._traj_list:
            block    = m.trajectories[i]
            norm_in  = getattr(block, NORM_INPUT_NAME)
            norm_obs = getattr(block, NORM_OBS_NAME)
            sp_var   = getattr(block, dutils.NN_SLACK_POS_NAME)
            sn_var   = getattr(block, dutils.NN_SLACK_NEG_NAME)
            for t in self._traj_t_sorted[i]:
                for j in range(self._input_dim):
                    norm_in_vars.append(norm_in[t, j])
                for j in range(self._obs_dim):
                    norm_obs_vars.append(norm_obs[t, j])
                for k in range(self._output_dim):
                    sp_vars.append(sp_var[t, k])
                    sn_vars.append(sn_var[t, k])

        self._norm_input_idx = np.array(
            ext.get_primal_indices(norm_in_vars), dtype=int
        ).reshape(self._total_points, self._input_dim)

        self._norm_obs_idx = np.array(
            ext.get_primal_indices(norm_obs_vars), dtype=int
        ).reshape(self._total_points, self._obs_dim)

        self._sp_idx = np.array(
            ext.get_primal_indices(sp_vars), dtype=int
        ).reshape(self._total_points, self._output_dim)

        self._sn_idx = np.array(
            ext.get_primal_indices(sn_vars), dtype=int
        ).reshape(self._total_points, self._output_dim)

        self._nn_constr_idx = np.array(
            ext.get_grey_box_output_constraint_indices(m.nn_block), dtype=int
        ).reshape(self._total_points, self._output_dim)

    def _get_current_values(self):
        """Read norm_input, norm_obs, sp, sn from the current Pyomo model state."""
        m = self._model

        rows_ni, rows_obs, rows_sp, rows_sn = [], [], [], []
        for i in self._traj_list:
            block    = m.trajectories[i]
            norm_in  = getattr(block, NORM_INPUT_NAME)
            norm_obs = getattr(block, NORM_OBS_NAME)
            sp_var   = getattr(block, dutils.NN_SLACK_POS_NAME)
            sn_var   = getattr(block, dutils.NN_SLACK_NEG_NAME)
            for t in self._traj_t_sorted[i]:
                rows_ni.append([pyo.value(norm_in[t, j])  for j in range(self._input_dim)])
                rows_obs.append([pyo.value(norm_obs[t, j]) for j in range(self._obs_dim)])
                rows_sp.append([pyo.value(sp_var[t, k])   for k in range(self._output_dim)])
                rows_sn.append([pyo.value(sn_var[t, k])   for k in range(self._output_dim)])

        return (np.array(rows_ni), np.array(rows_obs),
                np.array(rows_sp), np.array(rows_sn))
