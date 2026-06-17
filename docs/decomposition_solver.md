# Decomposition Solver

The decomposition approach decouples the NN parameter update from the DAE solve. It alternates
between:

1. **Inner step** — fix $\theta$, solve the DAE NLP via cyipopt + Grey-Box Model (GBM).
2. **Outer step** — fix the DAE solution, compute $\nabla_\theta \mathcal{L}$ via
   **implicit differentiation of the KKT conditions**, and take an Adam step on $\theta$.

This mirrors the structure of implicit-function-theorem-based meta-learning and bi-level
optimisation, applied here to a physics-informed setting.

---

## Algorithm

For each outer iteration $k$:

$$
\theta^{k+1} = \theta^k - \alpha_k \cdot \text{Adam}\!\left(\nabla_\theta \mathcal{L}(\theta^k)\right)
$$

The gradient $\nabla_\theta \mathcal{L}$ is obtained by:

1. Solving the inner NLP:
   $\min_{x,z} \mathcal{L}_\text{data}(x) + \lambda_\text{slack} \|z - \phi_\theta(\xi)\|_1$
2. Extracting KKT multipliers.
3. Back-solving the linearised KKT system (via **FERAL** — a pure-Rust sparse LDL^T
   solver) to obtain $\partial (x^*, z^*) / \partial \theta$.
4. Applying the chain rule.

The $\ell_1$ slack relaxation (`slack_coef`) ensures the inner NLP is feasible even when
$\phi_\theta$ is a poor approximation early in training.

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

mlp, history = train_decomp(
    problem=problem,
    mlp=mlp,
    cfg=cfg,
    data=smoother_data,           # normalization statistics
    smoother_model=smoother_m,    # warm-start the inner NLP
    cyipopt_options={'tol': 1e-6, 'max_iter': 300},
)
```

`history` is a dict with keys:
`obj_history`, `data_fit_history`, `grad_norm_history`, `diag_history`, `ipopt_timing_history`.

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

mlp, history = train_decomp(problem, mlp, cfg, data, mpi_comm=comm)
```

Run with: `mpirun -n 4 python train.py`

---

## When to Use

- Large networks where the full Hessian is too expensive.
- Many independent training trajectories (MPI parallelism).
- When you want fine-grained control over the training schedule.

For small-to-medium networks with few trajectories, the {doc}`simultaneous_solver` is
simpler and often faster.

---

## API Reference

See {doc}`api/decomp` for `DecompConfig`, `train_decomp`, and `build_decomp_model`.
See {doc}`api/pretrain` for `PretrainConfig` and `pretrain_mlp`.
