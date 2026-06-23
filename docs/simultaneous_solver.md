# Simultaneous Solver

In the simultaneous approach, NN parameters $\theta$ are treated as **additional decision
variables** in the collocation NLP. IPOPT (via POUNCE or cyipopt) optimises states, NN outputs,
and network weights jointly in a single solve — no outer training loop is required.

---

## Mathematical Formulation

After Radau collocation, the simultaneous NLP reads:

$$
\min_{\theta,\, \mathbf{x},\, \mathbf{z}} \quad
\sum_{k} \bigl\| x(\tau_k) - \hat{x}_k \bigr\|^2
+ \lambda_\text{reg} \|\theta\|^2
$$

$$
\text{s.t.} \quad \text{collocation constraints on } (x, z), \quad
z(t_i, \cdot) = \phi_\theta\bigl(\xi(t_i, \cdot)\bigr), \quad \forall\, i.
$$

The NN is embedded symbolically — either via **expression-writing** (exact Hessian) or as a
**grey-box model** (GBM, Jacobian-only, L-BFGS).

---

## Two Sub-Variants

### Expression-writing (default, `use_gbm=False`)

NN computations are written as explicit Pyomo expressions. This exposes the exact second-order
structure to IPOPT, enabling the full Hessian and typically faster convergence.

Solver: `SolverFactory('pounce')` — a pure-Rust IPOPT port installed as `pounce-solver`.

### Grey-box model (`use_gbm=True`)

The NN forward pass is wrapped in a `NNSimulGreyBoxModel` (a `PyNumero`
`ExternalGreyBoxModel`). Only Jacobian-vector products are provided; the Hessian is
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

## IPOPT Options

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
