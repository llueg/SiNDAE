# Hybrid DAE Overview

SiNDAE targets *hybrid* differential-algebraic equation (DAE) systems (also called
Universal Differential Equations or UDEs) in which one or more constitutive relations
are replaced by a trainable neural network. The notation on this page follows
{cite}`lueg2025simultaneous`.

---

## Problem formulation

SiNDAE considers the semi-explicit neural DAE on a horizon $[t_0, t_f]$:

$$
\begin{aligned}
\frac{d\mathbf{x}}{dt} &= \mathbf{f}\bigl(\mathbf{x}(t),\, \mathbf{y}(t),\, \mathbf{z}(t),\, \mathbf{p}\bigr), && \forall t \in [t_0, t_f] \\
0 &= \mathbf{h}\bigl(\mathbf{x}(t),\, \mathbf{y}(t),\, \mathbf{z}(t),\, \mathbf{p}\bigr), && \forall t \in [t_0, t_f] \\
0 &= \mathbf{z}(t) - \mathbf{f}_{NN}\bigl(\mathbf{v}(t),\, \boldsymbol{\theta}\bigr), && \forall t \in [t_0, t_f] \\
\mathbf{x}(t_0) &= \mathbf{x}_0(\mathbf{p}). &&
\end{aligned}
$$

The variables are the differential states $\mathbf{x}(t) \in \mathbb{R}^{n_x}$, the algebraic
variables $\mathbf{y}(t) \in \mathbb{R}^{n_y}$ and $\mathbf{z}(t) \in \mathbb{R}^{n_z}$, and the
independent static variables $\mathbf{p} \in \mathbb{R}^{n_p}$. The neural network

$$
\mathbf{f}_{NN} : \mathbb{R}^{n_v + n_\theta} \mapsto \mathbb{R}^{n_z},
\qquad \boldsymbol{\theta} \in \mathbb{R}^{n_\theta},
$$

supplies the unknown terms $\mathbf{z}(t)$ from a chosen subset of the remaining variables,
its inputs $\mathbf{v}(t) \in \mathbb{R}^{n_v}$ with
$\mathbf{v}(t) \subseteq \{\mathbf{x}(t),\, \mathbf{y}(t),\, \mathbf{p}\}$. Domain knowledge
defines this structural prior, that is, which variables enter $\mathbf{v}(t)$. The maps
$\mathbf{f} : \mathbb{R}^{n_x + n_y + n_z + n_p} \mapsto \mathbb{R}^{n_x}$ and
$\mathbf{h} : \mathbb{R}^{n_x + n_y + n_z + n_p} \mapsto \mathbb{R}^{n_y}$ (and $\mathbf{f}_{NN}$)
are assumed Lipschitz continuous on $[t_0, t_f]$ once $\mathbf{p}$ and $\boldsymbol{\theta}$ are
fixed. The initial differential state may depend on $\mathbf{p}$; if it is unknown it can be
absorbed into $\mathbf{p}$.

In the SiNDAE API this maps onto a `ProblemDefinition` as follows: $\mathbf{f}$ and
$\mathbf{h}$ are the differential and algebraic constraints written in `build_trajectory`,
the network inputs $\mathbf{v}(t)$ are returned by `get_input_vars`, the network outputs
$\mathbf{z}(t)$ by `get_output_vars`, and $\mathbf{f}_{NN}$ is a `SimpleMLP` (or any compatible
module).

### Differential index

Substituting the network relation into the other equations gives the algebraic constraint

$$
0 = \mathbf{h}\bigl(\mathbf{x}(t),\, \mathbf{y}(t),\, \mathbf{f}_{NN}(\mathbf{x}(t), \boldsymbol{\theta}),\, \mathbf{p}\bigr).
$$

For a given $\mathbf{p}$ and $\boldsymbol{\theta}$, if $\nabla_{\mathbf{y}} \mathbf{h}$ is nonsingular
for all $t \in [t_0, t_f]$, the DAE is index-1. Otherwise the index is the minimum number of
differentiations of the algebraic constraints required to obtain ODEs for the algebraic
variables $\mathbf{y}(t)$, exactly as for conventional DAEs {cite}`biegler2010nonlinear`. SiNDAE
places no restriction on the index; the four-tank example is index-2.

### Data and training objective

The training data come from a set of trajectories, or scenarios,
$\mathcal{S} = \{1, \dots, n_s\}$. The network outputs $\mathbf{z}(t)$ are usually unobserved;
instead we observe the variables that define the network input, which for the problems
considered here are the differential states $\mathbf{x}(t)$, sampled at times
$t_i \in \mathcal{T}_o^s$. Writing $\bar{\mathbf{x}}^{(s)}(t)$ for the ground-truth trajectory of
scenario $s$, the observations are

$$
\hat{\mathbf{x}}_i^{(s)} = \bar{\mathbf{x}}^{(s)}(t_i) + \boldsymbol{\epsilon}_i^{(s)},
\qquad \forall s \in \mathcal{S},\ \forall t_i \in \mathcal{T}_o^s,
$$

with zero-mean Gaussian observation noise $\boldsymbol{\epsilon}_i^{(s)}$. The data-fit loss for a
continuous state profile $\mathbf{x}^{(s)}(t)$ on scenario $s$ is

$$
\varphi^{(s)}\bigl(\mathbf{x}^{(s)}(t)\bigr) =
\sum_{t_i \in \mathcal{T}_o^s} \bigl\| \mathbf{x}^{(s)}(t_i) - \hat{\mathbf{x}}_i^{(s)} \bigr\|_2^2 .
$$

Training minimizes $\sum_{s \in \mathcal{S}} \varphi^{(s)}$ over the network parameters
$\boldsymbol{\theta}$ together with the state and algebraic trajectories, subject to the neural
DAE above. SiNDAE solves this either jointly as a single NLP (the
[simultaneous approach](simultaneous_solver.md)) or with an outer loop over
$\boldsymbol{\theta}$ around inner DAE solves (the
[decomposition approach](decomposition_solver.md)).

---

## Collocation discretization

The continuous neural DAE is discretized with **Lagrange-Radau collocation** on a mesh of
$N_{fe}$ finite elements with $N_{cp}$ collocation points each. This replaces the differential
and algebraic constraints with a finite set of algebraic equations, turning the training
problem into a finite-dimensional NLP. Following the paper, a discretized variable at
collocation index $k$ on scenario $s$ is written $\mathbf{x}_k^{(s)}$.

```{note}
`nfe` and `ncp` set the accuracy versus cost trade-off. A finer mesh resolves stiff or
fast dynamics more accurately at the cost of a larger NLP.
```

Pyomo's [`dae.collocation`](https://pyomo.readthedocs.io/) transformation performs the
symbolic discretization; `ProblemDefinition.discretize` calls it by default.

---

## Defining your own problem

Subclass `ProblemDefinition` and implement three abstract methods. `build_trajectory` writes
$\mathbf{f}$ and $\mathbf{h}$ and fixes the initial conditions; `get_input_vars` returns
$\mathbf{v}(t)$; `get_output_vars` returns $\mathbf{z}(t)$.

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
        """Network inputs v(t) at time t."""
        return [block.x[t, j] for j in range(2)]

    def get_output_vars(self, block, t):
        """Network outputs z(t) at time t."""
        return [block.z[t, 0]]
```

Optionally override:

| Method | Default | Purpose |
|--------|---------|---------|
| `get_obs_vars` | same as `get_input_vars` | Observed variables in the data-fit objective $\varphi^{(s)}$ |
| `get_aux_vars` | empty | Extra variables to record in `InstanceData` |
| `discretize` | Radau collocation | Override for custom schemes |
| `add_true_output_constraints` | `NotImplementedError` | True formula for $\mathbf{z}(t)$, used only by `generate_data` |

See [](api/problem.md) for the full API.

---

## Smoother pre-step

Before training the network, SiNDAE solves a **smoother NLP** that:

1. Fits a smooth state profile to the noisy observations $\hat{\mathbf{x}}_i^{(s)}$.
2. Produces initial values for $\mathbf{x}(t)$ and $\mathbf{z}(t)$ to warm-start the training NLP.
3. Computes the normalization statistics (mean and standard deviation) for the network inputs
   $\mathbf{v}(t)$ and outputs $\mathbf{z}(t)$.

The smoother penalizes $\|\dot{\mathbf{z}}_\text{smooth}\|^2$ weighted by `smooth_coef`. A larger
`smooth_coef` yields smoother but potentially less data-faithful $\mathbf{z}$ estimates.

See [](api/smoother.md) and `sindae.algorithms.smoother.solve_smoother`.

---

## Built-in example problems

`sindae.example_problems` ships three benchmark systems:

| Class | System | DAE index |
|-------|--------|-----------|
| `FourTankProblem` | Four-tank hydraulic network (4 differential, 5 algebraic, $n_z = 2$) | 2 |
| `LeslieGowerProblem` | Predator-prey ODE (2 differential, $n_z = 1$) | ODE |
| `FedBatchBioreactorProblem` | Fed-batch bioreactor with Monod kinetics (4 differential, $n_z = 1$) | ODE |
```
