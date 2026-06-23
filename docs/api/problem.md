# Problem Definition

`sindae.problem`

The [`ProblemDefinition`](#sindae.problem.ProblemDefinition) abstract base class is the
single entry point for declaring an ODE/DAE system to SiNDAE. Subclass it, implement the
three abstract methods, and pass an instance to any training or inference function.

The three required methods are:

- `build_trajectory(block, traj_idx)` — write the base DAE (Pyomo `Var`s,
  `DerivativeVar`s, and constraints) for one trajectory, leaving the unknown term as a
  free variable `z`. No neural network and no discretization here.
- `get_input_vars(block, t)` — the Pyomo vars fed **into** the network at time `t`.
- `get_output_vars(block, t)` — the Pyomo vars the network **produces** at time `t`.

Everything else has a sensible default: `discretize` applies Lagrange–Radau collocation,
`get_obs_vars` defaults to the NN inputs, and `get_aux_vars` defaults to none. Implement
the optional `add_true_output_constraints` only if you want to synthesize data with
[`generate_data`](data_utils.md).

## Usage

A minimal one-state ODE, `dx/dt = z`, where `z` is the unknown rate the network will
learn:

```python
import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
from sindae.problem import ProblemDefinition


class ExponentialDecay(ProblemDefinition):
    def build_trajectory(self, block, traj_idx):
        t0 = self.t_span[0]
        block.t    = dae.ContinuousSet(bounds=self.t_span)
        block.x    = pyo.Var(block.t, range(self.input_dim), initialize=1.0)
        block.z    = pyo.Var(block.t, range(self.z_dim), initialize=0.0)
        block.dxdt = dae.DerivativeVar(block.x, wrt=block.t)

        @block.Constraint(block.t, range(self.input_dim))
        def ode(b, t, i):
            return b.dxdt[t, i] == b.z[t, 0]      # z is supplied by the network

        block.x[t0, 0].fix(float(self.ics[traj_idx, 0]))

    def get_input_vars(self, block, t):
        return [block.x[t, 0]]                    # NN input  = state x

    def get_output_vars(self, block, t):
        return [block.z[t, 0]]                    # NN output = learned rate z


problem = ExponentialDecay(
    ics=np.array([[1.0]]),     # one trajectory, x(0) = 1
    input_dim=1, z_dim=1,
    t_span=(0.0, 5.0), nfe=20, ncp=3,
)
```

See the [examples gallery](../examples_gallery/index.md) for complete ODE and DAE
problems, and [Defining a Network Architecture](network_architecture.md) for the network
side.

## API reference

:::{include} _generated/problem.md
:::
