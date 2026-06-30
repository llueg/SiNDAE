# Solvers

`sindae.solvers`

Backend selection for the two solver roles in SiNDAE. The NLP solver optimises each
Pyomo model; the linear (KKT) solver runs the sparse back-solve inside the
decomposition gradient.

- [`make_nlp_solver`](#sindae.solvers.make_nlp_solver) selects the NLP backend:
  `pounce` (the pip-installable default), `ipopt`, or `cyipopt`.
- [`make_linear_solver`](#sindae.solvers.make_linear_solver) selects the linear/KKT
  backend: `feral` (the pip-installable default), `ma27`, or `scipy`.
- [`NLPSolver`](#sindae.solvers.NLPSolver) is the abstract backend interface returned
  by `make_nlp_solver`; [`NLPResult`](#sindae.solvers.NLPResult) is what its `solve`
  method returns.

The stage functions take these selectors directly, so a backend can be chosen without
touching the package internals. `solve_smoother`, `generate_data`, and the
expression-writing path of `solve_simultaneous` accept `backend=`; `solve_inference`
accepts `backend=` (its grey-box model needs cyipopt); `train_decomp` accepts
`linear_solver=` for the KKT back-solve. `cyipopt` and `ipopt` remain selectable, not
removed.

## Usage

```python
import sindae as sd
from sindae.algorithms.smoother import solve_smoother
from sindae.algorithms.decomp.train import train_decomp

# Pick the NLP backend for a stage (default is "pounce"):
smoother_m = solve_smoother(problem, mlp, smooth_coef=1.0, backend="ipopt")

# Pick the KKT linear solver for decomposition training (default is "feral"):
trained_m, mlp, history = train_decomp(
    problem, mlp, cfg, data=smoother_data, linear_solver="ma27",
)

# Build a configured solver directly (power-user path):
solver = sd.make_nlp_solver("cyipopt", options={"tol": 1e-8})
result = solver.solve(model)              # returns an NLPResult
```

## API reference

:::{include} _generated/solvers.md
:::
