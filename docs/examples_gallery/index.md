# Examples Gallery

The examples below demonstrate SiNDAE on benchmark hybrid DAE systems.
Each example covers data generation, smoother initialisation, and training with
both the simultaneous and decomposition approaches.

---

## Available Examples

### Leslie-Gower Predator-Prey

A two-state ODE. The NN replaces a modified Holling type II predator growth term.
Demonstrates the simplest SiNDAE workflow on a pure ODE.

**Script:** `examples/leslie_gower.py`

---

### Four-Tank Hydraulic Network

A four-state, index-2 DAE with five algebraic flow variables. The NN replaces two
nonlinear hydraulic functions (pump characteristic and tank discharge).

**Script:** `examples/four_tank.py`

---

### Fed-Batch Bioreactor

A four-state ODE with Monod growth kinetics. The NN replaces the specific growth rate
$\mu(S)$.

**Script:** `examples/fedbatch.py`

---

### Multi-Trajectory MPI (Four-Tank)

The four-tank system trained over three independent initial conditions in parallel using
MPI. Demonstrates the `mpi_comm` argument to `train_decomp`.

**Script:** `examples/example_mpi.py`

---

## Running an Example

```bash
# Single-process
python examples/leslie_gower.py

# MPI (decomposition, 4 ranks)
mpirun -n 4 python examples/example_mpi.py
```

Set `METHOD = 'simul'` or `METHOD = 'decomp'` at the top of each script to switch
between the two training algorithms.

---

```{note}
Jupyter notebook versions of these examples are coming soon.
```
