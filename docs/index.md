# SiNDAE

**SiNDAE** (*Simultaneous Neural Differential-Algebraic Equation*) is a Python package
for learning unknown nonlinear terms in ODE/DAE systems from noisy time-series observations.

Rather than training a neural network in isolation and plugging it into a simulator afterward,
SiNDAE embeds the network directly inside the physics-based model and trains it by solving a
single nonlinear program (NLP). This simultaneous approach preserves the DAE structure, allows
exact second-order derivatives, and is robust to index-2 constraints.

SiNDAE is the companion code to {cite}`lueg2025simultaneous`.

---

## Two training algorithms

| Algorithm | Description |
|-----------|-------------|
| **Simultaneous** | NN weights and DAE states are all decision variables in one large NLP. Solved with POUNCE (exact Hessian, or L-BFGS for the grey-box variant). |
| **Decomposition** | Outer Adam loop updates NN weights; inner NLP (POUNCE + GBM) solves the DAE at each step. Gradients via KKT implicit differentiation. Supports MPI. |

Both algorithms use [Pyomo DAE](https://pyomo.readthedocs.io/) for symbolic model building and
Lagrange–Radau collocation for time discretization.

---

## Quickstart

```python
import numpy as np
import sindae as sd

problem = sd.LeslieGowerProblem(nfe=40, ncp=3)
sd.generate_data(problem, noise_std=[0.05, 0.05])

mlp = sd.SimpleMLP(in_size=2, out_size=1, widths=[16, 16],
                   activations=[jax.nn.softplus] * 2)

model = sd.HybridDAE(
    method="simultaneous",              # or "decomposition"
    net=mlp,
    train=sd.SimultaneousConfig(reg_coef=1e-3),
)
model.fit(problem)                      # smoother -> pretrain -> train

new_problem = sd.LeslieGowerProblem(ics=np.array([[1.2, 0.15]]), nfe=40, ncp=3)
pred = model.predict(new_problem, slack_coef=1e-5)
```

See [](quickstart.md) for the full walkthrough, including the stage-level
functions behind the wrapper.

---

## Contents

```{tableofcontents}
```
