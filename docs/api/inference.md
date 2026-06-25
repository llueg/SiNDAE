# Inference

`sindae.algorithms.inference`

Embed a trained MLP back into a new (or the same) DAE problem and solve for the trajectory
consistent with the learned dynamics. The network is enforced as a hard GBM output
constraint by default (`slack_coef=0`, a square system solved by POUNCE), or relaxed via
an $\ell_1$ slack (`slack_coef > 0`) when the trained network does not fit the inference
problem's dynamics exactly.

- [`make_inference_model`](#sindae.algorithms.inference.make_inference_model) builds the
  inference NLP.
- [`solve_inference`](#sindae.algorithms.inference.solve_inference) builds and solves it.

Because the mechanistic equations remain hard constraints, predictions stay physically
consistent even at initial conditions never seen during training.

## Usage

```python
import numpy as np
from sindae import extract_instance_data
from sindae.algorithms.inference import solve_inference

# New initial conditions — the discretization may also differ from training.
val_problem = MyProblem(
    ics=np.array([[5.0, 0.3]]),
    input_dim=2, z_dim=1, t_span=(0, 80), nfe=20, ncp=2,
)

inference_m = solve_inference(
    val_problem, mlp,
    data=trained_data,        # normalization stats from training
    slack_coef=1e-5,          # small l1 relaxation; use 0.0 for a hard constraint
)
prediction = extract_instance_data(val_problem, inference_m)
```

## API reference

:::{include} _generated/inference.md
:::
