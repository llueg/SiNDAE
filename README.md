# SiNDAE — A Simultaneous Approach for Training Neural Differential-Algebraic Equations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2504.04665-b31b1b.svg)](https://arxiv.org/abs/2504.04665)

**SiNDAE** is a Python package for hybrid modeling of dynamical systems. It learns
unknown nonlinear terms in ODE and DAE systems directly from data by embedding a
neural network inside the governing equations and training it as a single nonlinear
program (NLP). Because the mechanistic equations are kept as hard constraints, the
learned model stays physically consistent, including when predicting new operating
conditions never seen during training.

SiNDAE is the companion code to
[*A simultaneous approach for training neural differential-algebraic systems of
equations*](https://arxiv.org/abs/2504.04665) (Lueg et al., 2025).

## Features

- **Two training backends** behind a symmetric API: a *simultaneous* approach that
  solves for the network weights and the trajectory jointly in one NLP, and a
  *decomposition* approach that wraps an outer Adam loop around inner DAE solves with
  implicit-differentiation gradients (and supports MPI).
- **ODEs and high-index DAEs**, discretized with Pyomo collocation.
- **Bring your own data**: fit to measured time series, including the partially
  observed case where only some states are recorded.
- **Custom neural architectures** through a grey-box interface, in addition to the
  built-in `SimpleMLP`.
- **Inference under new conditions**: embed a trained model in a fresh problem and
  predict, with the mechanistic structure keeping the result physically feasible.
- **Binary-free install**: the pure-Rust [POUNCE](https://github.com/jkitchin/pounce)
  and [FERAL](https://github.com/jkitchin/feral) solvers replace HSL/MA27, so no
  licensed binaries are required.

## Installation

### conda (recommended)

```bash
git clone https://github.com/llueg/SiNDAE.git
cd SiNDAE
conda env create -f environment.yml
conda activate sindae
```

### pip

```bash
git clone https://github.com/llueg/SiNDAE.git
cd SiNDAE
pip install -e ".[full]"     # omit [full] to skip cyipopt and MPI
```

The `full` extra adds `cyipopt` and `mpi4py`, which are needed for the decomposition
approach, the grey-box simultaneous variant, and inference. Their wheels are
platform-dependent, so the conda route is preferred. See
[`docs/installation.md`](docs/installation.md) for GPU/Apple Silicon, MPI, and
troubleshooting notes.

## Quickstart

Generate noisy data from a built-in example, fit the hybrid model, and extract the
trained trajectory:

```python
import jax
from sindae import SimpleMLP, generate_data, extract_instance_data
from sindae.algorithms.smoother import solve_smoother
from sindae.algorithms.simultaneous.train import SimultaneousConfig, solve_simultaneous
from sindae.example_problems import LeslieGowerProblem

jax.config.update("jax_enable_x64", True)

problem = LeslieGowerProblem(nfe=40, ncp=3)
mlp = SimpleMLP(
    in_size=problem.input_dim, out_size=problem.z_dim,
    widths=[16, 16], activations=[jax.nn.softplus] * 2,
)

data       = generate_data(problem, noise_std=[0.05, 0.05])   # or load your own measurements
smoother_m = solve_smoother(problem, mlp)                      # smooth, warm-start the solve
cfg        = SimultaneousConfig(use_gbm=False, reg_coef=1e-3)
trained_m, mlp = solve_simultaneous(problem, mlp, cfg, data=data, smoother_model=smoother_m)

trained = extract_instance_data(problem, trained_m)            # states + learned term
```

The decomposition approach is a drop-in alternative with the same call shape:

```python
from sindae.algorithms.decomp.train import DecompConfig, train_decomp

cfg = DecompConfig(n_steps=300, lr=5e-3)
trained_m, mlp, history = train_decomp(problem, mlp, cfg, data=data, smoother_model=smoother_m)
```

See the [Quickstart guide](docs/quickstart.md) for the full walkthrough.

## How it works

A typical workflow has four stages: build a problem, solve a *smoother* to get smooth
warm-start trajectories and normalization statistics, pre-train the network on those,
then train the hybrid model with one of the two backends.

| Backend | Entry point | Idea |
|---------|-------------|------|
| Simultaneous | `solve_simultaneous` | Network weights, states, and algebraic variables are decision variables in a single NLP solved by POUNCE (exact Hessian) or cyipopt (L-BFGS). |
| Decomposition | `train_decomp` | An outer Adam loop updates the weights; each step solves the DAE and obtains gradients by implicit differentiation of the KKT conditions. Supports MPI across trajectories. |

Both require the network to be twice continuously differentiable, so SiNDAE uses
smooth activations (`tanh`, `softplus`, `swish`). See
[Defining a Network Architecture](docs/api/network_architecture.md).

## Documentation

The full documentation is a [Jupyter Book](https://jupyterbook.org/) (mystmd engine).

[INSERT PUBLISHED DOCS WEBSITE HERE]

## Examples

Rendered notebooks live in [`docs/examples_gallery/`](docs/examples_gallery/) and are
organized around package capabilities:

| Notebook | Demonstrates |
|----------|--------------|
| `four_tank_example.ipynb` | End-to-end simultaneous workflow on an index-2 DAE |
| `leslie_gower_example.ipynb` | Decomposition training with a custom Lyapunov path constraint |
| `fedbatch_example.ipynb` | Loading measured data from a CSV and inference under new conditions |
| `fedbatch_partial_obs_example.ipynb` | Fitting when only some states are measured, and reconstructing the rest |
| `fedbatch_validation_example.ipynb` | Held-out validation and choosing the network size |

The same systems are also available as runnable scripts in [`examples/`](examples/):

| Script | System |
|--------|--------|
| `four_tank.py` | Four-tank hydraulic network (index-2 DAE) |
| `leslie_gower.py` | Leslie-Gower predator-prey (ODE) |
| `fedbatch.py` | Fed-batch bioreactor (ODE) |
| `example_mpi.py` | Four-tank trained over MPI ranks |

Set `METHOD = 'simul'` or `METHOD = 'decomp'` at the top of each script to switch
backends.

## Defining your own problem

Subclass `ProblemDefinition` and implement the three required methods. The network
takes `get_input_vars` as input and produces `get_output_vars`; `build_trajectory`
writes the mechanistic ODE/DAE and fixes the initial conditions.

```python
import pyomo.environ as pyo
import pyomo.dae as dae
from sindae.problem import ProblemDefinition

class MyProblem(ProblemDefinition):
    def build_trajectory(self, block, traj_idx):
        block.t    = dae.ContinuousSet(bounds=self.t_span)
        block.x    = pyo.Var(block.t, range(2), initialize=1.0)
        block.z    = pyo.Var(block.t, range(1))            # the learned term
        block.dxdt = dae.DerivativeVar(block.x, wrt=block.t)
        # ... add ODE/DAE constraints that reference block.z[t, 0] ...
        block.x[self.t_span[0], 0].fix(self.ics[traj_idx, 0])

    def get_input_vars(self, block, t):
        return [block.x[t, j] for j in range(2)]           # fed into the network

    def get_output_vars(self, block, t):
        return [block.z[t, 0]]                             # produced by the network
```

Optional overrides let you customize the observation model (`get_obs_vars`), track
extra variables (`get_aux_vars`), or define the true term for synthetic data
generation (`add_true_output_constraints`, used only by `generate_data`). See
[`sindae/example_problems.py`](sindae/example_problems.py) for complete
implementations of the four-tank DAE, Leslie-Gower ODE, and fed-batch bioreactor.

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

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for
details.
