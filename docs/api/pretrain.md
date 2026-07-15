# Pre-training

`sindae.algorithms.pretrain`

Supervised pre-training of the MLP on the smoother arrays before the main training loop.
This gives both solvers a sensible starting point and is applicable to the simultaneous
and decomposition approaches alike.

- [`PretrainConfig`](#sindae.algorithms.pretrain.PretrainConfig) — the few SGD
  hyperparameters (epochs, batch size, regularization).
- [`pretrain_mlp`](#sindae.algorithms.pretrain.pretrain_mlp) — fits the network to the
  `(nn_input, nn_output)` pairs in an `InstanceData`, applying input/output normalization
  internally.

## Usage

```python
from sindae import extract_instance_data
from sindae.algorithms.smoother import solve_smoother
from sindae.algorithms.pretrain import PretrainConfig, pretrain_mlp

# 1. Solve the smoother to get smooth (x, z) trajectories ...
smoother_m    = solve_smoother(problem, mlp, smooth_coef=1.0)
smoother_data = extract_instance_data(problem, smoother_m)

# 2. ... then pre-train the network on those pairs.
mlp = pretrain_mlp(
    mlp, smoother_data,
    PretrainConfig(epochs=200, batch_size=32, reg_coef=1e-3),
)
```

The returned `mlp` is ready to pass to
[`solve_simultaneous`](simultaneous.md) or [`train_decomp`](decomp.md).

## API reference

:::{include} _generated/pretrain.md
:::
