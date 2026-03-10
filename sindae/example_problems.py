"""
problems.py — Example ProblemDefinition subclasses for SiNDAE.

Each class demonstrates how to interface a specific ODE/DAE system with the
SiNDAE training framework.  The NN replaces an unknown nonlinear term ``z(t)``
that appears in the dynamics.

Classes
-------
FourTankProblem
    Four-tank hydraulic network (index-2 DAE, z_dim=2).
LeslieGowerProblem
    Leslie-Gower predator-prey ODE (z_dim=1).
FedBatchBioreactorProblem
    Fed-batch bioreactor ODE with Monod growth kinetics (z_dim=1).
"""
from __future__ import annotations

import numpy as np
import pyomo.dae as dae
import pyomo.environ as pyo

from sindae.problem import ProblemDefinition


# ── Four-Tank Problem ─────────────────────────────────────────────────────────

_FT_PARAMS = {
    'a1':        0.1,
    'a2':        0.1,
    'pump_coef': 0.2,
    'phi':       [0.1, 0.5, 2.0, 10.0],
}

_FT_DEFAULT_ICS = np.array([
    [0.75, 0.75, 2.50, 0.60],
    [3.10, 3.10, 1.50, 0.50],
    [0.90, 0.90, 1.80, 1.10],
])

_FT_STATE_DIM = 4
_FT_Z_DIM     = 2
_FT_ALG_DIM   = 5


class FourTankProblem(ProblemDefinition):
    """
    Four-tank hydraulic network UDE (index-2 DAE).

    Four liquid-level states x₀…x₃; five algebraic flow variables u₀…u₄.
    The NN replaces two nonlinear functions:
        z₀ = pump_coef · x₀ · x₃   (pump characteristic)
        z₁ = a1 · √x₀               (discharge from tank 0)

    The height-equality constraint (x₀ = x₁) makes this an index-2 DAE.

    Parameters
    ----------
    ics : array-like, shape (n_traj, 4), optional
        Initial conditions [x₀, x₁, x₂, x₃] per trajectory.
    params : dict, optional
        Physical parameters (a1, a2, pump_coef, phi). Defaults to _FT_PARAMS.
    t_span : (float, float)
    nfe, ncp : int
        Finite elements and collocation points for Pyomo DAE discretisation.
    """

    DEFAULT_PARAMS = _FT_PARAMS
    DEFAULT_ICS    = _FT_DEFAULT_ICS

    def __init__(
        self,
        ics=None,
        params=None,
        t_span=(0.0, 400.0),
        nfe=40,
        ncp=3,
        obs_times=None,
        obs_values=None,
    ):
        ics = np.asarray(ics) if ics is not None else self.DEFAULT_ICS.copy()
        super().__init__(
            ics,
            input_dim=_FT_STATE_DIM,
            z_dim=_FT_Z_DIM,
            t_span=t_span,
            nfe=nfe, ncp=ncp,
            obs_times=obs_times, obs_values=obs_values,
            obs_dim=_FT_STATE_DIM,
            aux_vars_dim=_FT_ALG_DIM,
        )
        self.params = params if params is not None else dict(self.DEFAULT_PARAMS)

    def build_trajectory(self, block: pyo.Block, traj_idx: int) -> None:
        t0, _ = self.t_span
        x0    = self.ics[traj_idx]
        p     = self.params
        phi   = p['phi']

        block.t    = dae.ContinuousSet(bounds=self.t_span)
        block.x    = pyo.Var(block.t, range(_FT_STATE_DIM), initialize=10.0)
        block.z    = pyo.Var(block.t, range(_FT_Z_DIM))
        block.u    = pyo.Var(block.t, range(_FT_ALG_DIM))
        block.dxdt = dae.DerivativeVar(block.x, wrt=block.t)

        @block.Constraint(block.t, range(_FT_STATE_DIM))
        def diffeq(b, t, s):
            u = [b.u[t, j] for j in range(_FT_ALG_DIM)]
            if s == 0:
                return b.dxdt[t, 0] == (1.0 / phi[0]) * (u[1] - u[3])
            elif s == 1:
                return b.dxdt[t, 1] == (1.0 / phi[1]) * u[2]
            elif s == 2:
                return b.dxdt[t, 2] == (1.0 / phi[2]) * (u[3] - u[4])
            else:
                return b.dxdt[t, 3] == (1.0 / phi[3]) * (u[4] - u[0])

        # Algebraic constraints
        @block.Constraint(block.t, range(_FT_ALG_DIM))
        def flow_lb(b, t, j):
            if j == 2:
                return pyo.Constraint.Skip
            return b.u[t, j] >= 0

        @block.Constraint(block.t)
        def flow_balance(b, t): return b.u[t, 0] == b.u[t, 1] + b.u[t, 2]

        @block.Constraint(block.t)
        def pump(b, t):        return b.u[t, 0] == b.z[t, 0]

        @block.Constraint(block.t)
        def height_req(b, t):  return b.x[t, 0] == b.x[t, 1]

        @block.Constraint(block.t)
        def dis3(b, t):        return b.u[t, 3] == b.z[t, 1]

        @block.Constraint(block.t)
        def dis4(b, t):        return b.u[t, 4] == p['a2'] * pyo.sqrt(b.x[t, 2])

        # Initial conditions (x₀ fixed implicitly by height_req)
        for j in range(1, _FT_STATE_DIM):
            block.x[t0, j].fix(float(x0[j]))

        # Clamp u₁ at t₀ to avoid under-determined algebraic system at the
        # initial collocation point (Radau does not enforce algebraic constraints there).
        @block.Constraint()
        def clamp_u1(b): return b.u[t0, 1] == b.u[t0, 0]

    def get_input_vars(self, block, t):  return [block.x[t, j] for j in range(_FT_STATE_DIM)]
    def get_output_vars(self, block, t): return [block.z[t, k] for k in range(_FT_Z_DIM)]
    def get_aux_vars(self, block, t):    return [block.u[t, k] for k in range(_FT_ALG_DIM)]

    def add_true_output_constraints(self, block: pyo.Block) -> None:
        a1 = self.params['a1']
        cp = self.params['pump_coef']

        @block.Constraint(block.t, range(_FT_Z_DIM))
        def true_z(b, t, k):
            if k == 0:
                return b.z[t, 0] == cp * b.x[t, 0] * b.x[t, 3]
            else:
                return b.z[t, 1] == a1 * pyo.sqrt(b.x[t, 0])


# ── Leslie-Gower Problem ──────────────────────────────────────────────────────

_LG_PARAMS = {
    'a1': 0.2,
    'a2': 0.01,
    'r1': 0.2,
    'r2': 0.2,
    'b1': 0.1,
}

_LG_DEFAULT_ICS = np.array([[1.0, 0.1]])


class LeslieGowerProblem(ProblemDefinition):
    """
    Leslie-Gower predator-prey UDE (ODE, z_dim=1).

    States: x₀ (prey), x₁ (predator)
    ODE:
        dx₀/dt = x₀ · (r1 - a1·x₁ - b1·x₀)
        dx₁/dt = x₁ · z₀
    True z:
        z₀ = r2 - a2·x₁/x₀   (modified Holling type II response)

    Parameters
    ----------
    ics : array-like, shape (n_traj, 2), optional
    params : dict, optional
    t_span, nfe, ncp : discretisation settings
    lyap_descent : bool
        If True, adds a Lyapunov descent inequality to the model.
    """

    DEFAULT_PARAMS = _LG_PARAMS
    DEFAULT_ICS    = _LG_DEFAULT_ICS

    def __init__(
        self,
        ics=None,
        params=None,
        t_span=(0.0, 80.0),
        nfe=40,
        ncp=3,
        obs_times=None,
        obs_values=None,
        lyap_descent: bool = False,
    ):
        ics = np.asarray(ics) if ics is not None else self.DEFAULT_ICS.copy()
        super().__init__(
            ics, input_dim=2, z_dim=1, t_span=t_span,
            nfe=nfe, ncp=ncp, obs_times=obs_times, obs_values=obs_values,
            obs_dim=2,
        )
        self.params       = params if params is not None else dict(self.DEFAULT_PARAMS)
        self.lyap_descent = lyap_descent

    def build_trajectory(self, block: pyo.Block, traj_idx: int) -> None:
        t0, _ = self.t_span
        p     = self.params
        x0_ic = self.ics[traj_idx]

        denom = p['a1'] * p['r2'] + p['a2'] * p['b1']
        x0_ss = p['r1'] * p['a2'] / denom
        x1_ss = p['r1'] * p['r2'] / denom

        block.t    = dae.ContinuousSet(bounds=self.t_span)
        block.x    = pyo.Var(block.t, range(2), initialize=1.0, within=pyo.NonNegativeReals)
        block.z    = pyo.Var(block.t, range(1), initialize=0.0)
        block.dxdt = dae.DerivativeVar(block.x, wrt=block.t)

        block.lyap_var = pyo.Var(block.t, initialize=1.0)
        block.dlyap_dt = dae.DerivativeVar(block.lyap_var, wrt=block.t)

        @block.Constraint(block.t)
        def lyap_constr(b, t):
            return b.lyap_var[t] == (
                pyo.log(b.x[t, 0] / x0_ss) + x0_ss / b.x[t, 0]
                + (p['a1'] * x0_ss / p['a2'])
                * (pyo.log(b.x[t, 1] / x1_ss) + x1_ss / b.x[t, 1])
            )

        if self.lyap_descent:
            @block.Constraint(block.t)
            def lyap_descent_constr(b, t):
                if t == t0:
                    return pyo.Constraint.Skip
                return b.dlyap_dt[t] <= 0

        @block.Constraint(block.t, range(2))
        def diffeq(b, t, s):
            if s == 0:
                return b.dxdt[t, 0] == b.x[t, 0] * (p['r1'] - p['a1'] * b.x[t, 1] - p['b1'] * b.x[t, 0])
            else:
                return b.dxdt[t, 1] == b.x[t, 1] * b.z[t, 0]

        for j in range(2):
            block.x[t0, j].fix(float(x0_ic[j]))

    def get_input_vars(self, block, t):  return [block.x[t, j] for j in range(2)]
    def get_output_vars(self, block, t): return [block.z[t, 0]]
    def get_aux_vars(self, block, t):    return [block.lyap_var[t]]

    def add_true_output_constraints(self, block: pyo.Block) -> None:
        p = self.params

        @block.Constraint(block.t)
        def true_z(b, t):
            return b.z[t, 0] == p['r2'] - p['a2'] * b.x[t, 1] / b.x[t, 0]


# ── Fed-Batch Bioreactor Problem ──────────────────────────────────────────────

_FB_PARAMS = {
    'Ks':     1.0,
    'Feed':   0.05,
    'Ypx':    0.2,
    'Yxs':    0.5,
    'mu_max': 0.2,
}

_FB_DEFAULT_ICS = np.array([
    [0.05,  0, 10,  1.0],
    [0.025, 0,  5,  0.8],
    [0.5,   0, 7.5, 0.95],
])


class FedBatchBioreactorProblem(ProblemDefinition):
    """
    Fed-batch bioreactor UDE (ODE, z_dim=1).

    States: x₀=X (biomass), x₁=P (product), x₂=S (substrate), x₃=V (volume)
    NN output: z₀ = μ (specific growth rate)

    ODE:
        dX/dt =  μ·X  - Feed·X/V
        dP/dt =  Ypx·μ·X - Feed·P/X
        dS/dt =  Feed·(Sf - S)/V - μ·X/Yxs
        dV/dt =  Feed

    True μ: Monod kinetics  μ = mu_max·S / (Ks + S)

    The feed concentration Sf is set per trajectory as the initial substrate
    value ics[traj_idx, 2].

    Parameters
    ----------
    ics : array-like, shape (n_traj, 4), optional
        Initial [X, P, S, V] per trajectory.  S₀ is also used as Sf.
    params : dict, optional
    t_span, nfe, ncp : discretisation settings
    """

    DEFAULT_PARAMS = _FB_PARAMS
    DEFAULT_ICS    = _FB_DEFAULT_ICS

    def __init__(
        self,
        ics=None,
        params=None,
        t_span=(0.0, 40.0),
        nfe=40,
        ncp=3,
        obs_times=None,
        obs_values=None,
    ):
        ics = np.asarray(ics) if ics is not None else self.DEFAULT_ICS.copy()
        super().__init__(
            ics, input_dim=4, z_dim=1, t_span=t_span,
            nfe=nfe, ncp=ncp, obs_times=obs_times, obs_values=obs_values,
            obs_dim=4,
        )
        self.params = params if params is not None else dict(self.DEFAULT_PARAMS)

    def build_trajectory(self, block: pyo.Block, traj_idx: int) -> None:
        p  = self.params
        t0 = self.t_span[0]
        x0 = self.ics[traj_idx]
        Sf = float(x0[2])

        block.t    = dae.ContinuousSet(bounds=self.t_span)
        block.x    = pyo.Var(block.t, range(4), domain=pyo.NonNegativeReals, initialize=1.0)
        block.z    = pyo.Var(block.t, range(1), initialize=0.1)
        block.dxdt = dae.DerivativeVar(block.x, wrt=block.t)

        Feed, Ypx, Yxs = p['Feed'], p['Ypx'], p['Yxs']

        @block.Constraint(block.t, range(4))
        def diffeq(b, t, s):
            mu = b.z[t, 0]
            X, P, S, V = b.x[t, 0], b.x[t, 1], b.x[t, 2], b.x[t, 3]
            if s == 0:
                return b.dxdt[t, 0] == -Feed * (X / V) + mu * X
            elif s == 1:
                return b.dxdt[t, 1] == -Feed * (P / X) + Ypx * mu * X
            elif s == 2:
                return b.dxdt[t, 2] == Feed * (Sf - S) / V - mu * (X / Yxs)
            else:
                return b.dxdt[t, 3] == Feed

        for j in range(4):
            block.x[t0, j].fix(float(x0[j]))

    def get_input_vars(self, block, t):  return [block.x[t, j] for j in range(4)]
    def get_output_vars(self, block, t): return [block.z[t, 0]]

    def add_true_output_constraints(self, block: pyo.Block) -> None:
        Ks, mu_max = self.params['Ks'], self.params['mu_max']

        @block.Constraint(block.t)
        def true_z(b, t):
            return b.z[t, 0] == mu_max * b.x[t, 2] / (Ks + b.x[t, 2])
