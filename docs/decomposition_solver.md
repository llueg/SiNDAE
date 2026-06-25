# Decomposition Solver

The decomposition approach is a **bi-level** scheme that decouples the network-parameter update
from the DAE solve. It alternates between:

1. **Inner step:** fix $\boldsymbol{\theta}$ and solve the per-scenario DAE NLP via cyipopt with a
   Grey-Box Model (GBM) of the network.
2. **Outer step:** compute $\nabla_{\boldsymbol{\theta}} \Phi$ by **implicit differentiation of the
   inner KKT conditions** and take an Adam step on $\boldsymbol{\theta}$.

This follows the bi-level formulation of {cite}`lueg2025simultaneous`, applied here to a
physics-informed setting.

---

## Algorithm

The outer problem minimizes

$$
\Phi(\boldsymbol{\theta}) = \alpha_r\, r(\boldsymbol{\theta}) + \sum_{s \in \mathcal{S}} \varphi^{(s)}\bigl(\tilde{\mathbf{x}}^{(s)}(\boldsymbol{\theta})\bigr),
$$

where $\tilde{\mathbf{x}}^{(s)}(\boldsymbol{\theta})$ is the solution of the inner DAE NLP for scenario
$s$ at the current weights, and $r(\boldsymbol{\theta}) = \tfrac{1}{2}\|\boldsymbol{\theta}\|_2^2$ with
weight $\alpha_r$ (`param_reg_coef`). Each outer iteration takes an Adam step,

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha\, \mathrm{Adam}\bigl(\nabla_{\boldsymbol{\theta}} \Phi\bigr).
$$

**Inner subproblem.** For fixed $\boldsymbol{\theta}$, the network constraint is relaxed with
non-negative $\ell_1$ slack variables $\boldsymbol{\Delta}^{\pm}$,

$$
\mathbf{z}_{ik}^{(s)} - \mathbf{f}_{NN}\bigl(\mathbf{v}_{ik}^{(s)}, \boldsymbol{\theta}\bigr) = \boldsymbol{\Delta}^{+}_{ik} - \boldsymbol{\Delta}^{-}_{ik},
$$

penalized in the inner objective with weight `slack_coef`. The relaxation keeps the inner NLP
feasible even when $\mathbf{f}_{NN}$ is a poor approximation early in training.

**Outer gradient.** Differentiating $\Phi$ gives

$$
\frac{d\Phi}{d\boldsymbol{\theta}} = \alpha_r \frac{dr}{d\boldsymbol{\theta}}
+ \sum_{s \in \mathcal{S}} \nabla_{\boldsymbol{\theta}} \tilde{\mathbf{x}}^{(s)}(\boldsymbol{\theta})^{\top} \frac{d\varphi^{(s)}}{d\tilde{\mathbf{x}}^{(s)}}.
$$

The sensitivity $\nabla_{\boldsymbol{\theta}} \tilde{\mathbf{x}}^{(s)}$ comes from implicit
differentiation of the inner KKT system: the linearized KKT matrix is factorized once per outer
iteration with **FERAL** (a pure-Rust sparse symmetric-indefinite LDL$^{\top}$ solver), and the
network terms enter through vector-Jacobian products evaluated by automatic differentiation.

---

## Usage

```python
from sindae.algorithms.pretrain import PretrainConfig, pretrain_mlp
from sindae.algorithms.decomp.train import DecompConfig, train_decomp

# (Optional) supervised pretrain on smoother output
mlp = pretrain_mlp(mlp, smoother_data, PretrainConfig(epochs=400, reg_coef=0.1))

cfg = DecompConfig(
    n_steps=300,
    lr=5e-3,
    init_slack_coef=10.0,       # initial L1 slack penalty
    slack_scale=2.0,             # multiply slack_coef every slack_update_interval steps
    slack_update_interval=50,
    max_slack_coef=1e3,
    param_reg_coef=1e-3,
    patience=20,                 # early stopping (set 0 to disable)
)

trained_m, mlp, history = train_decomp(
    problem=problem,
    mlp=mlp,
    cfg=cfg,
    data=smoother_data,           # normalization statistics
    smoother_model=smoother_m,    # warm-start the inner NLP
    cyipopt_options={'tol': 1e-6, 'max_iter': 300},
)

# trained_m is the solved NLP; recover trajectories with:
trained_data = extract_instance_data(problem, trained_m)
```

`history` is a dict with keys:
`obj_history`, `data_fit_history`, `grad_norm_history`, `diag_history`, `pouncetiming_history`.

---

## `DecompConfig` Reference

| Field | Default | Description |
|-------|---------|-------------|
| `n_steps` | 100 | Number of outer Adam iterations |
| `lr` | 0.01 | Adam learning rate (constant unless `lr_schedule` is set) |
| `grad_clip_norm` | `inf` | Gradient clip threshold (disable = `np.inf`) |
| `init_slack_coef` | 100 | Initial $\ell_1$ slack penalty weight |
| `slack_scale` | 2.0 | Multiplicative slack schedule factor |
| `slack_update_interval` | `inf` | Steps between slack schedule updates |
| `max_slack_coef` | 1000 | Cap on slack penalty |
| `mu_target` | 1e-10 | cyipopt barrier parameter target |
| `param_reg_coef` | 0.0 | L2 regularization on NN parameters |
| `patience` | 0 | Early stopping patience (0 = disabled) |
| `slack_tol` | 1e-6 | Feasibility threshold for early stopping |
| `lr_schedule` | `None` | Optional callable `(step: int) -> float` |

---

## MPI Parallelism

For multi-trajectory training, trajectories are distributed across MPI ranks. Each rank
maintains its own `TrajectoryBatchSubproblem` and gradients are All-Reduced before the Adam
step:

```python
from mpi4py import MPI
comm = MPI.COMM_WORLD

trained_m, mlp, history = train_decomp(problem, mlp, cfg, data, mpi_comm=comm)
```

Under MPI, `trained_m` is the **rank-local** model (holding only that rank's
trajectories); the returned `mlp` is identical across ranks (rank-0 authoritative).

Run with: `mpirun -n 4 python train.py`

---

## When to Use

- Large networks where the full Hessian is too expensive.
- Many independent training trajectories (MPI parallelism).
- When you want fine-grained control over the training schedule.

For small-to-medium networks with few trajectories, the [](simultaneous_solver.md) is
simpler and often faster.

---

## API Reference

See [](api/decomp.md) for `DecompConfig`, `train_decomp`, and `build_decomp_model`.
See [](api/pretrain.md) for `PretrainConfig` and `pretrain_mlp`.
