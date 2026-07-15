# Examples Gallery

Each notebook below walks through a complete workflow on a benchmark system, but
they are organised around **package capabilities** rather than the models
themselves. Pick the one that matches what you want to do.

---

## Notebooks

### [Four-Tank Hydraulic Network (Simultaneous)](four_tank_example.ipynb)

A four-state, **index-2 DAE** with five algebraic flow variables. The NN replaces
two nonlinear hydraulic relations. Demonstrates the end-to-end **simultaneous**
workflow: defining a DAE `ProblemDefinition`, data generation, smoother,
pre-training, training, and inference on new initial conditions.

### [Leslie-Gower Predator-Prey (Decomposition)](leslie_gower_example.ipynb)

A two-state ODE trained with the **decomposition** approach. Demonstrates adding a
**custom path constraint** (a Lyapunov descent inequality) to embed mechanistic
prior knowledge during training.

### [Fed-Batch Bioreactor: Importing Measured Data](fedbatch_example.ipynb)

A four-state ODE with Monod growth kinetics. Demonstrates the **bring-your-own-data**
workflow: loading measured time series from a CSV into `obs_times` / `obs_values`
(no synthetic data generation), and verifying that the trained model produces
**physically feasible predictions** under new operating conditions via inference.

### [Fed-Batch Bioreactor: Partial Observation](fedbatch_partial_obs_example.ipynb)

The same bioreactor, but only biomass and substrate are measured. Demonstrates the
**observation model** (`get_obs_vars` with `obs_dim` smaller than the number of
states), how to anchor unmeasured states with `unfix_io=False`, and reconstruction
of the unmeasured product and volume.

### [Fed-Batch Bioreactor: Validation and Model Selection](fedbatch_validation_example.ipynb)

Demonstrates **leave-one-batch-out validation**: holding out a batch, predicting it
with inference, and sweeping the network width to choose the model that generalises
best rather than the one that fits the training data hardest.

---

## Command-line scripts

The same systems are available as runnable scripts in the `examples/` directory,
including a multi-trajectory **MPI** example:

| Script | Demonstrates |
|--------|--------------|
| `examples/four_tank.py` | Four-tank DAE, `simul` / `decomp` toggle |
| `examples/leslie_gower.py` | Leslie-Gower ODE with Lyapunov constraint |
| `examples/fedbatch.py` | Fed-batch bioreactor |
| `examples/example_mpi.py` | Four-tank trained over MPI ranks |

```bash
# Single process
python examples/four_tank.py

# MPI (decomposition, 4 ranks)
mpirun -n 4 python examples/example_mpi.py
```

Set `METHOD = 'simul'` or `METHOD = 'decomp'` at the top of each script to switch
between the two training algorithms.
