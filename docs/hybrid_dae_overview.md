# Hybrid DAE Overview

SiNDAE targets *hybrid* differential-algebraic equation (DAE) systems, also called
**Universal Differential Equations** (UDEs), in which one or more constitutive relations
are replaced by a trainable neural network.

---

## Problem Formulation

Consider a DAE system on $[t_0, t_f]$:

$$
\dot{x}(t) = f\bigl(x(t),\, z(t),\, u(t)\bigr), \qquad 0 = g\bigl(x(t),\, z(t)\bigr),
$$

where $x \in \mathbb{R}^{n_x}$ are differential states, $z \in \mathbb{R}^{n_z}$ is an unknown
nonlinear term (the *NN output*), and $u$ are known inputs or algebraic variables.

The NN approximates $z$:

$$
z(t) \approx \phi_\theta\bigl(\xi(t)\bigr),
$$

where $\xi(t)$ is a subset of the DAE variables (the *NN input*, defined by
`get_input_vars`) and $\theta$ are the network parameters.

Given noisy measurements $\{(\tau_k,\, \hat{x}_k)\}$ of the states, the training objective
minimises the data-fit residual:

$$
\min_{\theta,\, x(\cdot),\, z(\cdot)} \sum_k \bigl\| x(\tau_k) - \hat{x}_k \bigr\|^2
$$

subject to the DAE dynamics.

---

## Collocation Discretization

The continuous DAE is discretized via **Lagrange–Radau collocation** on a mesh of $N_{fe}$
finite elements with $N_{cp}$ collocation points each. This replaces the ODE/DAE constraints
with a finite set of algebraic equations, turning the training problem into a finite-dimensional NLP.

```{note}
`nfe` and `ncp` control the accuracy–cost trade-off. Typical values: `nfe=40, ncp=3`.
```

Pyomo's [`dae.collocation`](https://pyomo.readthedocs.io/) transformation handles the
symbolic discretization; `ProblemDefinition.discretize` calls it by default.

---

## Defining Your Own Problem

Subclass `ProblemDefinition` and implement three abstract methods:

```python
from sindae.problem import ProblemDefinition
import pyomo.dae as dae
import pyomo.environ as pyo

class MyProblem(ProblemDefinition):
    def build_trajectory(self, block, traj_idx):
        """Declare Pyomo Var, DerivativeVar, and constraints (no NN yet)."""
        block.t    = dae.ContinuousSet(bounds=self.t_span)
        block.x    = pyo.Var(block.t, range(2), initialize=1.0)
        block.z    = pyo.Var(block.t, range(1))
        block.dxdt = dae.DerivativeVar(block.x, wrt=block.t)

        @block.Constraint(block.t)
        def ode(b, t):
            return b.dxdt[t, 0] == -b.x[t, 0] + b.z[t, 0]

        block.x[self.t_span[0], 0].fix(self.ics[traj_idx, 0])

    def get_input_vars(self, block, t):
        """Variables fed into the NN at time t."""
        return [block.x[t, j] for j in range(2)]

    def get_output_vars(self, block, t):
        """Variables produced by the NN at time t."""
        return [block.z[t, 0]]
```

Optionally override:

| Method | Default | Purpose |
|--------|---------|---------|
| `get_obs_vars` | same as `get_input_vars` | Observed variables in the data-fit objective |
| `get_aux_vars` | empty | Extra variables to record in `InstanceData` |
| `discretize` | Radau collocation | Override for custom schemes |
| `add_true_output_constraints` | `NotImplementedError` | True formula for `generate_data` |

See [](api/problem.md) for the full API.

---

## Smoother Pre-step

Before training the NN, SiNDAE solves a **smoother NLP** that:

1. Fits a smooth trajectory to the noisy observations.
2. Produces initial values for $x(t)$ and $z(t)$ to warm-start the training NLP.
3. Computes normalization statistics (mean/std) for the NN inputs and outputs.

The smoother penalises $\|\dot{z}_\text{smooth}\|^2$ weighted by `smooth_coef`.
A larger `smooth_coef` produces smoother but potentially less data-faithful $z$ estimates.

See [](api/smoother.md) and `sindae.algorithms.smoother.solve_smoother`.

---

## Built-in Example Problems

`sindae.example_problems` ships three benchmark systems:

| Class | System | DAE index |
|-------|--------|-----------|
| `FourTankProblem` | Four-tank hydraulic network (4 states, 5 algebraic, 2 NN outputs) | 2 |
| `LeslieGowerProblem` | Predator–prey ODE (2 states, 1 NN output) | — |
| `FedBatchBioreactorProblem` | Fed-batch bioreactor with Monod kinetics (4 states, 1 NN output) | — |
