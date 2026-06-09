"""
kkt_utils.py  (decomp approach)

KKT gradient utilities for the decomposition training loop.

Functions
---------
  make_full_eval_fn, make_model_vjp_fn, make_mixed_vjp_fn,
  make_summed_model_vjp_fn, make_summed_mixed_vjp_fn
      Build JAX VJP callables used in the KKT implicit-differentiation formula.

  init_linear_solver(linear_solver, interface)
      Symbolic factorization of the KKT matrix.

  eval_del_obj_del_rho(x, sp, sn, ...) -> np.ndarray
      Build the ∂L/∂ρ RHS for the KKT back-solve.

  v_eval_del_obj_del_param(interface, linear_solver, ...) -> (grad, diag)
      Full KKT gradient  dL/dθ  via implicit differentiation.

Constants
---------
  NN_CONSTR_NAME, NN_SLACK_POS_NAME, NN_SLACK_NEG_NAME
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

import pyomo.environ as pyo
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.interior_point.linalg.scipy_interface import ScipyInterface
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
from pyomo.common.timing import HierarchicalTimer


NN_CONSTR_NAME    = 'nn_constraints'
NN_SLACK_POS_NAME = 'nn_slack_pos'
NN_SLACK_NEG_NAME = 'nn_slack_neg'


# ---------------------------------------------------------------------------
# VJP builders
# ---------------------------------------------------------------------------

def make_full_eval_fn(unflatten_fn):
    def _full_eval_fn(_x, _flat_params):
        new_mlp = unflatten_fn(_flat_params)
        return new_mlp(_x).reshape((-1,))
    return _full_eval_fn


def make_eval_with_param_fn(x, unflatten_fn):
    return functools.partial(make_full_eval_fn(unflatten_fn), _x=x)


def make_eval_with_x_fn(flat_params, unflatten_fn):
    return functools.partial(make_full_eval_fn(unflatten_fn), _flat_params=flat_params)


def make_mixed_vjp_fn(full_eval_fn):
    def mixed_vjp(_inputs, _flat_params, _cotangent_out, _cotangent_in):
        def _vjp_wrt_x(_params):
            _, vjp_fn_inner = jax.vjp(lambda _x: full_eval_fn(_x, _params), _inputs)
            return vjp_fn_inner(_cotangent_out)[0]
        _, vjp_fn_outer = jax.vjp(_vjp_wrt_x, _flat_params)
        return vjp_fn_outer(_cotangent_in)[0]
    return mixed_vjp


def make_model_vjp_fn(full_eval_fn):
    def model_vjp(_inputs, _params, _cotangent_out):
        _, vjp_fn = jax.vjp(lambda _p: full_eval_fn(_inputs, _p), _params)
        return vjp_fn(_cotangent_out)[0]
    return model_vjp


def make_summed_model_vjp_fn(model_vjp_fn):
    v_model_vjp = jax.vmap(model_vjp_fn, in_axes=(0, None, 0), out_axes=0)
    def summed_model_vjp(_inputs, _params, _cotangent_out):
        return jnp.sum(v_model_vjp(_inputs, _params, _cotangent_out), axis=0)
    return jax.jit(summed_model_vjp)


def make_summed_mixed_vjp_fn(mixed_vjp_fn):
    v_mixed_vjp = jax.vmap(mixed_vjp_fn, in_axes=(0, None, 0, 0), out_axes=0)
    def summed_mixed_vjp(_inputs, _flat_params, _cotangent_out, _cotangent_in):
        return jnp.sum(v_mixed_vjp(_inputs, _flat_params, _cotangent_out, _cotangent_in), axis=0)
    return jax.jit(summed_mixed_vjp)


# ---------------------------------------------------------------------------
# Linear solver helpers
# ---------------------------------------------------------------------------

def init_linear_solver(linear_solver: ScipyInterface,
                       interface: InteriorPointInterface):
    kkt_matrix = interface.evaluate_primal_dual_kkt_matrix()
    res = linear_solver.do_symbolic_factorization(kkt_matrix)
    if res.status != LinearSolverStatus.successful:
        raise RuntimeError('Symbolic factorization failed with status: ' + str(res.status))


# ---------------------------------------------------------------------------
# NLP variable / constraint extractors  (internal helpers)
# ---------------------------------------------------------------------------

def _get_nn_input_vars(inputs, _interface, get_input_vars_at_t, t_set=None):
    num_points, input_dim = inputs.shape
    _m = _interface.pyomo_model()
    if t_set is None:
        assert hasattr(_m, 't')
        t_set = _m.t
    assert len(t_set) == num_points
    for i, ti in enumerate(t_set):
        t_input_vars = get_input_vars_at_t(_m, ti)
        inputs[i, :] = np.array([pyo.value(v) for v in t_input_vars])


def _get_nn_input_vars_indices(inputs_primal_indices, _interface, get_input_vars_at_t, t_set=None):
    num_points, input_dim = inputs_primal_indices.shape
    _m = _interface.pyomo_model()
    if t_set is None:
        assert hasattr(_m, 't')
        t_set = _m.t
    assert len(t_set) == num_points
    for i, ti in enumerate(t_set):
        t_input_vars = get_input_vars_at_t(_m, ti)
        inputs_primal_indices[i, :] = np.array(
            _interface.get_primal_indices(pyomo_variables=t_input_vars), dtype=int
        ).reshape((input_dim,))


def _get_nn_constraint_slack_vars(slack_pos, slack_neg, interface):
    num_points, output_dim = slack_pos.shape
    assert slack_neg.shape == (num_points, output_dim)
    _m = interface.pyomo_model()
    spv = getattr(_m, NN_SLACK_POS_NAME)
    snv = getattr(_m, NN_SLACK_NEG_NAME)
    for i in range(output_dim):
        slack_pos[:, i] = np.array(pyo.value(spv[:, i]))
        slack_neg[:, i] = np.array(pyo.value(snv[:, i]))


def _get_nn_constraint_slack_vars_indices(slack_pos_primal_indices,
                                          slack_neg_primal_indices,
                                          interface):
    num_points, output_dim = slack_pos_primal_indices.shape
    assert slack_neg_primal_indices.shape == (num_points, output_dim)
    _m = interface.pyomo_model()
    spv = getattr(_m, NN_SLACK_POS_NAME)
    snv = getattr(_m, NN_SLACK_NEG_NAME)
    slack_pos_primal_indices[:] = np.array(
        interface.get_primal_indices(pyomo_variables=[spv]), dtype=int
    ).reshape((num_points, output_dim))
    slack_neg_primal_indices[:] = np.array(
        interface.get_primal_indices(pyomo_variables=[snv]), dtype=int
    ).reshape((num_points, output_dim))


def _get_nn_constr_multipliers(_interface, multipliers, constr_indices):
    assert multipliers.shape == constr_indices.shape
    multipliers[:] = -_interface.get_duals_eq()[constr_indices]


def _get_nn_constr_indices(_interface, constr_indices):
    _m = _interface.pyomo_model()
    nn_constr = getattr(_m, NN_CONSTR_NAME)
    constr_indices[:] = np.array(
        _interface.get_constraint_indices(pyomo_constraints=[nn_constr])
    ).reshape(constr_indices.shape)


# ---------------------------------------------------------------------------
# KKT gradient
# ---------------------------------------------------------------------------

def eval_del_obj_del_rho(x, sp, sn,
                         _x_primal_indices, _sp_primal_indices, _sn_primal_indices,
                         grad_fn, n_rho,
                         subsample_frac=1.0,
                         key=jax.random.PRNGKey(0)):
    """
    Build the ∂L/∂ρ vector for the KKT back-solve.

    Parameters
    ----------
    grad_fn : callable  (x, sp, sn) -> (grad_x, grad_sp, grad_sn)
        Should be a **precompiled** (jax.jit) gradient function.
    """
    del_obj_del_rho = np.zeros(n_rho)
    grad_obj_x, grad_obj_sp, grad_obj_sn = grad_fn(x, sp, sn)

    if subsample_frac < 1.0:
        n_pts = x.shape[0]
        n_sel = int(n_pts * subsample_frac)
        selected = jax.random.choice(key, n_pts, shape=(n_sel,), replace=False)
        mask = jnp.zeros(n_pts, dtype=bool).at[selected].set(True)
        grad_obj_x  = jnp.where(mask[:, None], grad_obj_x,  0.0)
        grad_obj_sp = jnp.where(mask[:, None], grad_obj_sp, 0.0)
        grad_obj_sn = jnp.where(mask[:, None], grad_obj_sn, 0.0)

    del_obj_del_rho[_x_primal_indices.flatten()]  = np.array(grad_obj_x).flatten()
    del_obj_del_rho[_sp_primal_indices.flatten()] = np.array(grad_obj_sp).flatten()
    del_obj_del_rho[_sn_primal_indices.flatten()] = np.array(grad_obj_sn).flatten()
    return del_obj_del_rho


def v_eval_del_obj_del_param(interface, linear_solver, param,
                              input, input_indices,
                              sp, sp_indices,
                              sn, sn_indices,
                              nn_constr_multipliers, nn_constr_indices,
                              grad_fn, sum_mixed_vjp, sum_model_vjp,
                              subsample_frac=1.0,
                              key=jax.random.PRNGKey(0),
                              timer: HierarchicalTimer=None,
                              obs=None,
                              obs_indices=None):
    """
    Compute dL/dθ via KKT implicit differentiation.

    Parameters
    ----------
    input / input_indices : norm_input values + NLP indices
    obs / obs_indices     : norm_obs values + NLP indices (default: same as input)
    """
    _obs     = obs         if obs         is not None else input
    _obs_idx = obs_indices if obs_indices is not None else input_indices

    def _t(name):
        if timer is not None:
            timer.start(name)

    def _s(name):
        if timer is not None:
            timer.stop(name)

    _t('kkt_matrix')
    kkt   = interface.evaluate_primal_dual_kkt_matrix()
    n_rho = kkt.shape[0]
    _s('kkt_matrix')

    _t('del_obj_del_rho')
    del_obj_del_rho = eval_del_obj_del_rho(
        x=jnp.array(_obs), sp=jnp.array(sp), sn=jnp.array(sn),
        _x_primal_indices=_obs_idx,
        _sp_primal_indices=sp_indices, _sn_primal_indices=sn_indices,
        grad_fn=grad_fn, n_rho=n_rho,
        subsample_frac=subsample_frac, key=key,
    )
    _s('del_obj_del_rho')

    _t('numeric_fact')
    linear_solver.do_numeric_factorization(kkt)
    _s('numeric_fact')

    _t('back_solve')
    v_bar, status = linear_solver.do_back_solve(del_obj_del_rho)
    _s('back_solve')

    if status.status != LinearSolverStatus.successful:
        print('Warning: back solve failed with status: ' + str(status.status))
        diag = {'model_vjp_norm': 0.0, 'mixed_vjp_norm': 0.0,
                'v_bar_z_norm': 0.0, 'v_bar_x_norm': 0.0, 'solve_failed': True}
        return np.zeros_like(param), diag

    v_bar_input_components = v_bar[input_indices]
    z_constr_indices_in_vbar = (nn_constr_indices
                                + interface.n_primals()
                                + interface.n_ineq_constraints())
    v_bar_z_constr_components = v_bar[z_constr_indices_in_vbar]

    del_obj_del_param = np.zeros_like(param)

    _t('vjp')
    model_term = np.array(sum_model_vjp(input, param, v_bar_z_constr_components))
    mixed_term = np.array(sum_mixed_vjp(input, param, nn_constr_multipliers, v_bar_input_components))
    del_obj_del_param -= model_term
    del_obj_del_param -= mixed_term
    _s('vjp')

    diag = {
        'model_vjp_norm': float(np.linalg.norm(model_term)),
        'mixed_vjp_norm': float(np.linalg.norm(mixed_term)),
        'v_bar_z_norm':   float(np.linalg.norm(v_bar_z_constr_components)),
        'v_bar_x_norm':   float(np.linalg.norm(v_bar_input_components)),
        'solve_failed':   False,
    }
    return del_obj_del_param, diag
