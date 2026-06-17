# Installation

## Requirements

- **Python ≥ 3.9** (3.11 recommended)
- **conda** (recommended) or **pip** in a virtual environment
- macOS, Linux, or Windows (WSL2 recommended on Windows)

The full install includes:

| Component | Purpose |
|-----------|---------|
| `numpy`, `scipy`, `matplotlib` | Numerics and plotting |
| `jax`, `jaxlib` | Automatic differentiation (CPU by default) |
| `equinox`, `optax` | Neural network layers and optimizers |
| `pyomo` | Symbolic DAE model building and collocation |
| `pounce-solver` | IPOPT-compatible NLP solver (pure-Rust wheels, no HSL) |
| `feral-solver` | Sparse symmetric KKT solver for the decomp gradient (pure-Rust) |
| `cyipopt` | Python interface to IPOPT with MUMPS — needed for decomp and GBM variants |
| `mpi4py` | MPI parallelism for multi-trajectory decomp training |

---

## Option A — conda (recommended)

Conda installs `cyipopt` and `mpi4py` from conda-forge with pre-built IPOPT+MUMPS and
OpenMPI binaries, avoiding any compilation step.

### 1. Clone the repository

```bash
git clone https://github.com/TODO/SiNDAE.git
cd SiNDAE
```

### 2. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate sindae
```

This installs all core and optional dependencies. There may be a few OS-level
permission prompts for the ASL solver binaries.

### 3. Install SiNDAE in editable mode

The `environment.yml` already runs `pip install -e .` automatically. If you need to
reinstall manually:

```bash
pip install -e .
```

---

## Option B — pip in a virtual environment

Use this if you do not have conda. MPI support requires a system MPI
(e.g. `brew install open-mpi` on macOS); `cyipopt` wheels are platform-specific.

### 1. Clone and enter the repository

```bash
git clone https://github.com/TODO/SiNDAE.git
cd SiNDAE
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate     # Windows
```

### 3. Install dependencies

Core only (no MPI, no cyipopt):

```bash
pip install -e .
```

Full install including cyipopt and mpi4py:

```bash
pip install -e ".[full]"
```

macOS — install OpenMPI for MPI support:

```bash
brew install open-mpi
```

---

## GPU / Accelerator support (optional)

By default, JAX runs on CPU. To enable GPU or Apple Silicon acceleration, replace
the `jax[cpu]` install with the appropriate variant **before** or **after** the
environment setup:

```bash
# NVIDIA GPU (CUDA 12)
pip install -U "jax[cuda12]"

# Apple Silicon (Metal)
pip install -U "jax[metal]"
```

---

## Verify the installation

```python
import sindae
import jax
import pyomo.environ as pyo

# Confirm JAX 64-bit is available
jax.config.update('jax_enable_x64', True)
print(jax.devices())

# Confirm POUNCE is on PATH
solver = pyo.SolverFactory('pounce')
print("POUNCE available:", solver.available())

# Confirm cyipopt (optional — only needed for decomp / inference)
try:
    cy = pyo.SolverFactory('cyipopt')
    print("cyipopt available:", cy.available())
except Exception as e:
    print("cyipopt not available:", e)
```

---

## Troubleshooting

**`pounce not available`** — ensure the conda/venv environment is activated;
`pounce-solver` places the `pounce` binary in `$CONDA_PREFIX/bin` or `.venv/bin`.

**`cyipopt` import error on pip install** — use the conda route (Option A). The
pip wheel for `cyipopt` bundles a pre-built IPOPT+MUMPS only for specific
OS/architecture combinations.

**JAX `float32` warnings** — call `jax.config.update('jax_enable_x64', True)` at
the top of your script or set `JAX_ENABLE_X64=1` in your shell.

**MPI `mpirun: command not found`** — install OpenMPI:
`brew install open-mpi` (macOS) or `sudo apt install libopenmpi-dev` (Ubuntu).
