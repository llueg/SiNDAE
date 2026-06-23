# SiNDAE

**SiNDAE** (*Simultaneous Neural Differential-Algebraic Equation learning*) is a Python package
for identifying unknown nonlinear terms in ODE/DAE systems from noisy time-series observations.

Rather than training a neural network in isolation and plugging it into a simulator afterward,
SiNDAE embeds the network directly inside the physics-based model and trains it by solving a
single nonlinear program (NLP). This simultaneous approach preserves the DAE structure, allows
exact second-order derivatives, and is robust to index-2 constraints.

SiNDAE is the companion code to {cite}`lueg2025simultaneous`.

---

## Two training algorithms

| Algorithm | Description |
|-----------|-------------|
| **Simultaneous** | NN weights and DAE states are all decision variables in one large NLP. Solved with POUNCE (exact Hessian) or cyipopt (L-BFGS). |
| **Decomposition** | Outer Adam loop updates NN weights; inner NLP (cyipopt + GBM) solves the DAE at each step. Gradients via KKT implicit differentiation. Supports MPI. |

Both algorithms use [Pyomo DAE](https://pyomo.readthedocs.io/) for symbolic model building and
Lagrange–Radau collocation for time discretization.

---

## Quickstart

```python
import jax
from sindae import SimpleMLP, generate_data
from sindae.algorithms.smoother import solve_smoother
from sindae.algorithms.simultaneous.train import SimultaneousConfig, solve_simultaneous
from sindae.example_problems import LeslieGowerProblem

problem = LeslieGowerProblem(nfe=40, ncp=3)
mlp = SimpleMLP(in_size=2, out_size=1, widths=[16, 16],
                activations=[jax.nn.softplus]*2)

data = generate_data(problem, noise_std=[0.05, 0.05])
smoother_m = solve_smoother(problem, mlp)
cfg = SimultaneousConfig(use_gbm=False, reg_coef=1e-3)
trained_m, mlp = solve_simultaneous(problem, mlp, cfg, data=data,
                                    smoother_model=smoother_m)
```

See [](quickstart.md) for the full walkthrough.

---

## Contents

```{tableofcontents}
```
