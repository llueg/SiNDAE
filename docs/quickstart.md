# Quickstart Guide

This page walks through a complete training run on the Leslie-Gower predator–prey example,
from data generation to trained model. The same workflow applies to all built-in and
custom problems.

## Prerequisites

Follow the {doc}`installation` guide first. After installation, set JAX to 64-bit precision
at the top of your script:

```python
import jax
jax.config.update('jax_enable_x64', True)
```

---

## 1. Define or import a problem

SiNDAE represents a system via a `ProblemDefinition` subclass. Three built-in problems are
available in `sindae.example_problems`:

```python
from sindae.example_problems import LeslieGowerProblem

problem = LeslieGowerProblem(nfe=60, ncp=3)  # fine grid for data generation
```

See {doc}`hybrid_dae_overview` for an explanation of `nfe`/`ncp` and how to subclass
`ProblemDefinition` for your own system.

---

## 2. Construct the neural network

Use `SimpleMLP` from `sindae.nn_utils`:

```python
import jax
from sindae import SimpleMLP

mlp = SimpleMLP(
    in_size=problem.input_dim,   # fed from get_input_vars
    out_size=problem.z_dim,      # produced by get_output_vars
    widths=[16, 16],
    activations=[jax.nn.softplus, jax.nn.softplus],
    key=jax.random.PRNGKey(0),
)
```

---

## 3. Generate synthetic training data

```python
import numpy as np
from sindae import generate_data

data = generate_data(
    problem,
    noise_std=np.array([0.05, 0.05]),  # Gaussian noise on observations
    obs_every=4,                        # observe every 4th collocation point
    seed=0,
)
```

`generate_data` solves the true model with IPOPT, adds noise, and stores
`problem.obs_times` / `problem.obs_values` in-place.

---

## 4. Solve the smoother

The smoother fits a smooth trajectory to the noisy observations — it produces warm-start
values and normalization statistics for the subsequent training step.

```python
from sindae.algorithms.smoother import solve_smoother
from sindae.data_utils import extract_instance_data

# Switch to the coarser training grid
problem.nfe = 40
problem.ncp = 3

smoother_m    = solve_smoother(problem, mlp, smooth_coef=1.0)
smoother_data = extract_instance_data(problem, smoother_m)
```

---

## 5a. Train (simultaneous approach)

```python
from sindae.algorithms.simultaneous.train import solve_simultaneous

trained_m, mlp = solve_simultaneous(
    problem=problem,
    mlp=mlp,
    data=smoother_data,
    smoother_model=smoother_m,  # warm-start from smoother
    reg_coef=1e-3,
    pounceoptions={'tol': 1e-6, 'max_iter': 1000},
)
```

See {doc}`simultaneous_solver` for a detailed explanation.

---

## 5b. Train (decomposition approach)

```python
from sindae.algorithms.pretrain import PretrainConfig, pretrain_mlp
from sindae.algorithms.decomp.train import DecompConfig, train_decomp

# Optional: supervised pretrain on smoother arrays
mlp = pretrain_mlp(mlp, smoother_data, PretrainConfig(epochs=400))

cfg = DecompConfig(n_steps=300, lr=5e-3, init_slack_coef=10.0,
                   param_reg_coef=1e-3)
mlp, history = train_decomp(
    problem=problem, mlp=mlp, cfg=cfg,
    data=smoother_data, smoother_model=smoother_m,
)
```

See {doc}`decomposition_solver` for a detailed explanation.

---

## 6. Extract results and plot

```python
from sindae.data_utils import extract_instance_data
from sindae.plot_utils import plot_instance_data

trained_data = extract_instance_data(problem, trained_m)

fig, axes = plot_instance_data(
    datasets=[(trained_data, 'trained', {})],
    nn_input_names=['prey', 'predator'],
    nn_output_names=['z'],
    obs_times=problem.obs_times,
    obs_values=problem.obs_values,
    obs_names=['prey', 'predator'],
)
fig.savefig('result.pdf')
```

---

## Next steps

- {doc}`hybrid_dae_overview` — mathematical background and problem formulation
- {doc}`api/index` — full API reference
- {doc}`examples_gallery/index` — complete worked examples
