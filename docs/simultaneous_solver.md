# Simultaneous Solver

In the simultaneous approach, the network parameters $\boldsymbol{\theta}$ are treated as
**additional decision variables** in the collocation NLP. The solver (POUNCE or cyipopt)
optimises the differential states $\mathbf{x}$, the algebraic variables $\mathbf{y}$ and
$\mathbf{z}$, the static parameters $\mathbf{p}$, and the weights $\boldsymbol{\theta}$ jointly in
a single solve.

---

## Mathematical Formulation

After Radau collocation, the simultaneous NLP (eq. 6 of {cite}`lueg2025simultaneous`) reads:

$$
\min_{\boldsymbol{\theta},\, \mathbf{p},\, \{\mathbf{x},\mathbf{y},\mathbf{z}\}} \quad
\sum_{s \in \mathcal{S}} \varphi^{(s)}\bigl(\mathbf{x}^{(s)}\bigr) + \alpha_r\, r(\boldsymbol{\theta}),
$$

subject to, at each collocation point $(i, k)$ of every scenario $s \in \mathcal{S}$,

$$
\begin{aligned}
\sum_j \mathbf{x}_{ij}^{(s)}\, \ell_j'(\tau_k) &= h_i\, \mathbf{f}\bigl(\mathbf{x}_{ik}^{(s)}, \mathbf{y}_{ik}^{(s)}, \mathbf{z}_{ik}^{(s)}, \mathbf{p}\bigr), && \text{(collocation)} \\
0 &= \mathbf{h}\bigl(\mathbf{x}_{ik}^{(s)}, \mathbf{y}_{ik}^{(s)}, \mathbf{z}_{ik}^{(s)}, \mathbf{p}\bigr), && \text{(algebraic)} \\
\mathbf{z}_{ik}^{(s)} &= \mathbf{f}_{NN}\bigl(\mathbf{v}_{ik}^{(s)}, \boldsymbol{\theta}\bigr). && \text{(network)}
\end{aligned}
$$

Here $\varphi^{(s)}$ is the data-fit loss and $\mathbf{f}_{NN}$, $\mathbf{f}$, $\mathbf{h}$ follow the
[Hybrid DAE Overview](hybrid_dae_overview.md). The term
$r(\boldsymbol{\theta}) = \tfrac{1}{2}\|\boldsymbol{\theta}\|_2^2$ is an L2 regularizer with weight
$\alpha_r$ (the `reg_coef` argument). The network constraint is embedded symbolically, either by
**expression-writing** (exact Hessian) or as a **grey-box model** (GBM, Jacobian-only, L-BFGS).

---

## Two Sub-Variants

### Expression-writing (default, `use_gbm=False`)

NN computations are written as explicit Pyomo expressions. This exposes the exact second-order
structure to IPOPT, enabling the full Hessian and typically faster convergence.

Solver: `SolverFactory('pounce')`.

### Grey-box model (`use_gbm=True`)

The NN forward pass is wrapped in a `NNSimulGreyBoxModel` (a `PyNumero`
`ExternalGreyBoxModel`). Only Jacobian-vector products are provided and the Hessian is
approximated via **L-BFGS**. Slower per iteration but scales better to large networks.

Solver: `SolverFactory('cyipopt')`.

---

## Usage

```python
from sindae.algorithms.simultaneous.train import SimultaneousConfig, solve_simultaneous

cfg = SimultaneousConfig(
    use_gbm=False,    # expression-writing (exact Hessian)
    reg_coef=1e-3,    # L2 regularization on NN weights
)
trained_m, mlp = solve_simultaneous(
    problem=problem,
    mlp=mlp,
    cfg=cfg,
    data=smoother_data,           # provides normalization stats
    smoother_model=smoother_m,    # warm-start (optional but recommended)
    pounce_options={
        'tol': 1e-6,
        'max_iter': 1000,
    },
)
```

`mlp` holds the trained network weights (extracted from the NLP solution).
`trained_m` is the solved Pyomo model; pass it to `extract_instance_data` for trajectories.

The simultaneous approach solves a single NLP, so there is no training-curve
history to return (unlike [](decomposition_solver.md), whose outer Adam loop
produces one). Solve progress is reported by the solver status and, with
`tee=True`, the live IPOPT iteration log.

---

## Warm-starting

Passing `smoother_model` reuses the already-discretised and solved smoother NLP as the
starting point. This avoids re-building the model and gives IPOPT a feasible (or near-feasible)
initial point, significantly reducing iteration count.

---

## POUNCE Options

Common options for the expression-writing path:

| Option | Typical value | Effect |
|--------|---------------|--------|
| `tol` | `1e-6` | Primal–dual feasibility tolerance |
| `max_iter` | `1000` | Maximum IPOPT iterations |
| `hessian_approximation` | `exact` (default) | Use `limited-memory` for large networks |

For the GBM path, also set `hessian_approximation: limited-memory` (required).

---

## When to Use

- Small to medium networks where the exact Hessian is tractable.
- Single-trajectory problems or few trajectories.
- When you want the simplest possible workflow (one function call, no outer loop).

For large networks, many trajectories, or MPI-parallel training, consider the
[](decomposition_solver.md).

---

## API Reference

See [](api/simultaneous.md) for `solve_simultaneous`, `build_simultaneous_model`,
`build_simultaneous_model_gbm`, and `extract_mlp`.
