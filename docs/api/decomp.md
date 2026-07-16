# Decomposition Solver

`sindae.algorithms.decomp`

The decomposition approach alternates between an inner NLP solve (with the network
embedded as a Grey-Box Model, solved by POUNCE by default) and an outer Adam update whose
gradient comes from KKT implicit differentiation of the inner solution. It supports MPI
parallelism across trajectories.

- [`DecompConfig`](#sindae.algorithms.decomp.train.DecompConfig) — the Adam / KKT / slack
  hyperparameters.
- [`train_decomp`](#sindae.algorithms.decomp.train.train_decomp) — runs the training loop
  and returns `(model, trained_mlp, history)`.
- [`build_decomp_model`](#sindae.algorithms.decomp.model_builder.build_decomp_model) —
  builds the inner NLP (rarely called directly).

## Usage

```python
from sindae import extract_instance_data
from sindae.algorithms.decomp.train import DecompConfig, train_decomp

cfg = DecompConfig(n_steps=300, lr=5e-3, init_slack_coef=1e1, param_reg_coef=1e-3)

trained_m, mlp, history = train_decomp(
    problem, mlp, cfg,
    data=smoother_data,              # normalization statistics
    smoother_model=smoother_m,       # reuse the discretized smoother
    solver_options={'tol': 1e-6, 'max_iter': 300},   # nlp_solver='pounce' by default
)
trained_data = extract_instance_data(problem, trained_m)
# history['data_fit_history'], history['grad_norm_history'], ... for diagnostics
```

To parallelize across trajectories, pass an MPI communicator (one trajectory batch per
rank) and launch with `mpirun`:

```python
from mpi4py import MPI
trained_m, mlp, history = train_decomp(
    problem, mlp, cfg, data=smoother_data, mpi_comm=MPI.COMM_WORLD,
)
```

## API reference

:::{include} _generated/decomp.md
:::
