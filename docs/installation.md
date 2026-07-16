# Installation

SiNDAE is published on PyPI and installs with `pip`. conda is offered as an
alternative for platforms where the optional binary dependencies (`cyipopt`,
`mpi4py`) are easier to obtain as pre-built packages.

## Requirements

- **Python 3.11 or newer**
- **pip** (recommended) or **conda**
- Linux, macOS, or Windows (WSL2 recommended on Windows)

We recommend installing into a fresh virtual environment so SiNDAE and its
dependencies do not interfere with other projects:

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

## Install with pip (recommended)

### Core install

```bash
pip install sindae
```

This pulls the core stack (`numpy`, `scipy`, `jax`, `equinox`, `optax`, `pyomo`,
`matplotlib`) together with the pure-Rust solvers
[POUNCE](https://github.com/jkitchin/pounce) and
[FERAL](https://github.com/jkitchin/feral), which install from wheels with no
system libraries and no license. The core install runs the entire SiNDAE pipeline
(problem, smoother, **simultaneous** and **decomposition** training, grey-box models,
inference, and plotting) on POUNCE and FERAL.

### Full install

```bash
pip install "sindae[full]"
```

The `full` extra adds `mpi4py`, required for **MPI-parallel** decomposition training,
and `cyipopt`, an optional alternative NLP backend (POUNCE is the default for every
solve). Their wheels depend on your platform; if either fails to build from pip, use
the conda route below for those two packages.

### Model export (optional)

```bash
pip install "sindae[onnx]"    # HybridDAE.export to ONNX (+ a scaler sidecar) or JSON
pip install "sindae[omlt]"    # HybridDAE.to_omlt for an in-memory OMLT model
```

These extras are only needed to hand a trained network to a foreign optimization
tool. The core pipeline never imports them, and `HybridDAE.export(..., format="json")`
needs no extra at all. See the `HybridDAE` API page for usage.

## Install with conda (alternative)

conda-forge ships pre-built binaries for the dependencies that are awkward to build
with pip: `cyipopt` (bundled with IPOPT and MUMPS, no HSL license) and `mpi4py`
(linked against OpenMPI). Install those with conda, then install SiNDAE itself with
pip:

```bash
conda create -n sindae python=3.11
conda activate sindae
conda install -c conda-forge cyipopt mpi4py
pip install sindae
```

## Packages

| Component | Install | Purpose |
|-----------|---------|---------|
| `numpy`, `scipy`, `matplotlib` | core | Numerics and plotting |
| `jax`, `jaxlib` | core | Automatic differentiation (CPU by default) |
| `equinox`, `optax` | core | Neural network layers and optimizers |
| `pyomo` | core | Symbolic DAE model building and collocation |
| `pounce-solver` | core | IPOPT-compatible NLP solver (pure-Rust wheels, no HSL) |
| `feral-solver` | core | Sparse symmetric KKT solver for the decomposition gradient |
| `cyipopt` | `[full]` | Optional alternative NLP backend (IPOPT with MUMPS); POUNCE is the default |
| `mpi4py` | `[full]` | MPI parallelism for multi-trajectory decomposition training |
| `jax2onnx`, `onnxruntime` | `[onnx]` | Export a trained network to ONNX (`HybridDAE.export`) |
| `omlt` | `[omlt]` | Export a trained network as an OMLT model (`HybridDAE.to_omlt`) |

## GPU and accelerator support (optional)

By default JAX runs on CPU. To use a GPU or Apple Silicon, install the matching JAX
build after installing SiNDAE:

```bash
pip install -U "jax[cuda12]"     # NVIDIA GPU (CUDA 12)
pip install -U "jax[metal]"      # Apple Silicon (Metal)
```

## Development install (from source)

To contribute or to run the test suite, install an editable copy from a clone. The
`test` extra adds `pytest`:

```bash
git clone https://github.com/llueg/SiNDAE.git
cd SiNDAE
pip install -e ".[full,test]"
```

The repository also ships an `environment.yml` for a one-command conda development
environment (Python, `cyipopt`, `mpi4py`, and an editable install):

```bash
conda env create -f environment.yml
conda activate sindae
```

Run the fast test suite with:

```bash
pytest
```

## Verify the installation

```python
import sindae
import jax
import pyomo.environ as pyo

jax.config.update("jax_enable_x64", True)
print("JAX devices:", jax.devices())

# POUNCE backs the simultaneous workflow (core install).
print("POUNCE available:", pyo.SolverFactory("pounce").available())

# cyipopt is optional: a selectable alternative NLP backend, not required for any feature.
try:
    print("cyipopt available:", pyo.SolverFactory("cyipopt").available())
except Exception as exc:
    print("cyipopt not available:", exc)
```

## Troubleshooting

**`pounce` not available.** Make sure your virtual environment is active.
`pounce-solver` installs the `pounce` executable into the environment's `bin`
directory, which must be on `PATH`.

**`cyipopt` fails to build from pip.** Its wheels are platform-specific. Install it
from conda-forge instead (`conda install -c conda-forge cyipopt`), then
`pip install sindae` into the same environment.

**JAX emits `float32` warnings.** Set 64-bit mode at the top of your script with
`jax.config.update("jax_enable_x64", True)`, or set `JAX_ENABLE_X64=1` in your shell.

**`mpirun: command not found`.** Install an MPI implementation: OpenMPI via
`conda install -c conda-forge openmpi`, `brew install open-mpi` (macOS), or
`sudo apt install libopenmpi-dev` (Ubuntu).
