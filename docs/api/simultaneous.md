# Simultaneous Solver

`sindae.algorithms.simultaneous`

The simultaneous approach embeds the NN weights and biases directly in a single large NLP
and optimizes states, NN outputs, and NN parameters **jointly** in one solver call — no
outer training loop. Two backends are available, selected by
[`SimultaneousConfig.use_gbm`](#sindae.algorithms.simultaneous.train.SimultaneousConfig):

| `use_gbm` | Backend | Solver | Hessian |
|-----------|---------|--------|---------|
| `False` (default) | expression-writing | POUNCE | exact |
| `True` | grey-box (GBM) | POUNCE | L-BFGS (limited-memory) |

The expression-writing backend rewrites the `SimpleMLP` as explicit Pyomo expressions and
gets an exact Hessian; the grey-box backend treats the network as a black box (function +
Jacobian) and works with any smooth Equinox module. See
[Defining a Network Architecture](network_architecture.md) for the trade-offs.

## Usage

```python
from sindae import extract_instance_data
from sindae.algorithms.simultaneous.train import SimultaneousConfig, solve_simultaneous

cfg = SimultaneousConfig(use_gbm=False, reg_coef=1e-3)   # expression-writing, exact Hessian

trained_m, mlp = solve_simultaneous(
    problem, mlp, cfg,
    data=smoother_data,              # normalization statistics
    smoother_model=smoother_m,       # reuse the discretized smoother as a warm start
    pounce_options={'tol': 1e-6, 'max_iter': 1000},
)
trained_data = extract_instance_data(problem, trained_m)
```

For problems whose exact Hessian is awkward (e.g. ratio terms like $P/X$), switch to the
grey-box variant with a limited-memory Hessian:

```python
cfg = SimultaneousConfig(use_gbm=True, reg_coef=1e-3)
trained_m, mlp = solve_simultaneous(
    problem, mlp, cfg, data=smoother_data, smoother_model=smoother_m,
    pounce_options={'tol': 1e-6, 'max_iter': 1000,
                    'hessian_approximation': 'limited-memory'},
)
```

## API reference

:::{include} _generated/simultaneous.md
:::
