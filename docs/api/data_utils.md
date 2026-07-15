# Data Utilities

`sindae.data_utils`

Containers for multi-trajectory solution data and helpers for extracting and generating
training data from solved Pyomo models.

- [`TrajectoryData`](#sindae.data_utils.TrajectoryData) holds the per-trajectory arrays
  (`sampling_times`, `nn_input`, `nn_output`, `obs`, `aux_vars`).
- [`InstanceData`](#sindae.data_utils.InstanceData) is a list-like container of
  `TrajectoryData` that also computes the input/output normalization statistics used
  throughout training.
- [`extract_instance_data`](#sindae.data_utils.extract_instance_data) pulls an
  `InstanceData` out of any solved model.
- [`generate_data`](#sindae.data_utils.generate_data) solves the *true* model to
  synthesize (optionally noisy) observations for examples and benchmarks.

## Usage

```python
import numpy as np
from sindae import generate_data, extract_instance_data

# Synthesize noisy observations from a problem whose true output is known
# (requires the problem to implement add_true_output_constraints):
true_data = generate_data(problem, noise_std=np.array([0.05]), obs_every=2, seed=0)

# After any solve, pull the arrays out of the Pyomo model:
data = extract_instance_data(problem, solved_model)

traj0 = data[0]                    # a TrajectoryData
traj0.sampling_times               # (num_t,)
traj0.nn_input, traj0.nn_output    # (num_t, input_dim) / (num_t, output_dim)

data.input_mean, data.input_std    # normalization stats across all trajectories
data.output_mean, data.output_std
```

`generate_data` also populates `problem.obs_times` / `problem.obs_values` in place, so the
same `problem` is ready to hand to the smoother and training routines.

## API reference

:::{include} _generated/data_utils.md
:::
