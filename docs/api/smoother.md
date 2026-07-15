# Smoother

`sindae.algorithms.smoother`

The smoother NLP fits a smooth trajectory to noisy observations and produces the
warm-start values and normalization statistics used by subsequent training. It penalises
the time derivative of the NN-output variable $z_\text{smooth}$, weighted by `smooth_coef`
(larger ⇒ smoother fit).

- [`build_smoother_model`](#sindae.algorithms.smoother.build_smoother_model) constructs
  the Pyomo model.
- [`solve_smoother`](#sindae.algorithms.smoother.solve_smoother) builds and solves it,
  returning the solved model.

Running the smoother is the first step of the standard workflow: the solved model both
warm-starts the main solve (it is already built and discretized) and supplies the
normalization constants for the network.

## Usage

```python
from sindae import extract_instance_data
from sindae.algorithms.smoother import solve_smoother

# problem.obs_times / problem.obs_values must be set (from data or generate_data)
smoother_m    = solve_smoother(problem, mlp, smooth_coef=1.0)
smoother_data = extract_instance_data(problem, smoother_m)

# smoother_data feeds pre-training and supplies normalization stats;
# smoother_m can be reused as the base model for the training solve.
```

## API reference

:::{include} _generated/smoother.md
:::
