# SiNDAE — A SImultaneous approach for training Neural Differential-Algebraic systems of Equations

This repository contains code associated with the paper [A simultaneous approach for training neural differential-algebraic systems of equations](https://arxiv.org/abs/2504.04665).

## Installation

### Optional: Create a conda environment

```bash
conda create -n sindae python=3.11
conda activate sindae
```

### 1. Install IPOPT with HSL solvers

The MA27 linear solver (from HSL) is required.  Obtain a licence and follow the [IPOPT HSL installation guide](https://coin-or.github.io/Ipopt/INSTALL.html).

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install SiNDAE

```bash
git clone https://github.com/TODO/SiNDAE.git
cd SiNDAE
pip install -e .
```

## Examples

All examples are in the `examples/` directory.

| Script | System |
|--------|--------|
| `four_tank.py` | Four-tank hydraulic network (index-2 DAE) |
| `leslie_gower.py` | Leslie-Gower predator-prey (ODE) |
| `fedbatch.py` | Fed-batch bioreactor (ODE) | 
| `example_mpi.py` | Four-tank (multi-trajectory) |


## Defining your own problem

Subclass `ProblemDefinition` and implement four methods:

```python
from sindae.problem import ProblemDefinition
import pyomo.dae as dae
import pyomo.environ as pyo

class MyProblem(ProblemDefinition):
    def build_trajectory(self, block, traj_idx):
        """Add Pyomo Vars, DerivativeVars, and constraints (no NN yet)."""
        block.t    = dae.ContinuousSet(bounds=self.t_span)
        block.x    = pyo.Var(block.t, range(2), initialize=1.0)
        block.z    = pyo.Var(block.t, range(1))
        block.dxdt = dae.DerivativeVar(block.x, wrt=block.t)
        # ... add ODE/DAE constraints referencing block.z[t, 0] ...
        block.x[self.t_span[0], 0].fix(self.ics[traj_idx, 0])

    def get_input_vars(self, block, t):
        return [block.x[t, j] for j in range(2)]   # fed into NN

    def get_output_vars(self, block, t):
        return [block.z[t, 0]]                      # NN output

    def add_true_output_constraints(self, block):
        """True z for data generation only."""
        @block.Constraint(block.t)
        def true_z(b, t):
            return b.z[t, 0] == ...
```

See `sindae/example_problems.py` for complete implementations of the four-tank DAE, Leslie-Gower ODE, and fed-batch bioreactor.

## Citation

```bibtex
@article{lueg2025simultaneous,
  title={A simultaneous approach for training neural differential-algebraic systems of equations},
  author={Lueg, Laurens R and Alves, Victor and Schicksnus, Daniel and Kitchin, John R and Laird, Carl D and Biegler, Lorenz T},
  journal={arXiv preprint arXiv:2504.04665},
  year={2025}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
