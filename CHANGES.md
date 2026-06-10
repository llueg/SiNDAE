# Binary-free solver migration

Summary of the changes that remove all licensed/manually-installed binaries
(HSL MA27, standalone Ipopt) from the SiNDAE install. All solver components
now arrive via `pip`/`conda` wheels. Verified against the codebase as of
2026-06-10.

## 1. POUNCE replaces Ipopt for the non-GBM NLP solves

[POUNCE](https://github.com/jkitchin/pounce) is a pure-Rust port of Ipopt,
installed as prebuilt wheels (`pounce-solver`). Every `SolverFactory('ipopt')`
on the expression-writing (ASL) path was changed to `SolverFactory('pounce')`:

- `sindae/algorithms/simultaneous/train.py` — non-GBM simultaneous solve
- `sindae/algorithms/smoother.py` — smoother NLP
- `sindae/data_utils.py` — data-generation solve

How it registers: `pounce-solver` installs a `pounce` CLI that speaks the
AMPL NL/SOL protocol, so Pyomo's generic ASL solver interface picks it up by
name — no plugin import required. The `pounce` executable must be on `PATH`
(pip places it in the environment's `bin/`; verified working via
`SolverFactory('pounce').available()`).

Not changed: the GBM paths (decomp subproblems, GBM simultaneous, inference)
still use `SolverFactory('cyipopt')`. POUNCE is ASL-based and cannot consume
`ExternalGreyBoxBlock` models; cyipopt from conda-forge bundles IPOPT
compiled against MUMPS, so this also requires no HSL license.

## 2. FERAL replaces MA27 as the decomposition KKT linear solver

The decomp approach factorizes the primal-dual KKT matrix once per outer
iteration and back-solves it for the implicit-differentiation gradient
(`kkt_utils.py`). This used `InteriorPointMA27Interface` (licensed HSL
binary), interim-swapped to `ScipyInterface` (works, but general LU: no
symmetry exploitation, no inertia).

New: `sindae/interfaces/feral_interface.py` — `FeralInterface`, a wrapper
around [FERAL](https://github.com/jkitchin/feral) (`feral-solver` wheels;
pure-Rust sparse symmetric indefinite LDLᵀ with certified inertia, MIT).
It implements Pyomo's `IPLinearSolverInterface` protocol:

- `do_symbolic_factorization` — validates structure only; feral computes the
  symbolic factorization inside the first `factor()` call and caches it
  across refactorizations with the same sparsity pattern (matching the
  `_symbolic_done` factor-once/solve-many pattern in `subproblem.py`)
- `do_numeric_factorization` — converts the scipy/BlockMatrix KKT matrix via
  `feral.from_scipy(..., symmetric='full')`, maps feral statuses onto
  `LinearSolverStatus`
- `do_back_solve` — `solve_refined` (iterative refinement) by default;
  handles `BlockVector` round-trips
- `get_inertia` — `(n_pos, n_neg, n_zero)` free from the LDLᵀ pivots
  (scipy needed a dense eigendecomposition for this; MA27 parity)

`subproblem.py` and `kkt_utils.py` now instantiate/annotate `FeralInterface`.
KKT matrix *evaluation* is unchanged — that lives in PyNumero's
`InteriorPointInterface`, not the linear solver.

Verification: on a 500×500 random sparse symmetric indefinite system,
FeralInterface agrees with `scipy.splu` to ~1e-14, reports exactly the true
inertia, reuses the symbolic factorization across refactorizations
(`symbolic_call_count` stays 1), and round-trips BlockVectors.

## 3. Packaging: binary-free install

- `environment.yml` (new) — conda env: `cyipopt` from conda-forge (bundles
  IPOPT+MUMPS, no HSL), `mpi4py` from conda-forge (links OpenMPI), everything
  else via pip including `pounce-solver` and `feral-solver`, plus `-e .`
- `pyproject.toml` — core dependencies now include `pounce-solver>=0.4` and
  `feral-solver>=0.9`; cyipopt and mpi4py moved to the `full` extra since
  their pip wheels are platform-dependent (conda preferred)
- `requirements.txt` — mirrors the above for pip-only installs

## 4. Test cases

`tests/test_pounce_ipopt_swapin.py` — POUNCE vs IPOPT on the simultaneous
(non-GBM) method across all three example problems (four_tank, leslie_gower,
fedbatch), with both exact and limited-memory Hessian approximations.
Asserts final X/Z rel-RMSE agreement within 1e-3 and saves comparison plots.
GBM mode is excluded (requires cyipopt).

`tests/test_linear_solver_swapins.py` — two tests for the decomp KKT linear
solver swap (ma27 / scipy / feral, skipping unavailable ones):

- `test_back_solve_consistency` — direct check: each interface runs the exact
  `do_symbolic → do_numeric → do_back_solve` protocol from `kkt_utils` on a
  synthetic symmetric indefinite KKT-like system; residuals must be < 1e-10
  and all solutions must agree to 1e-8. Catches a wrong solve immediately.
- `test_linear_solver_swapins` — end-to-end: full decomp training per problem
  per solver (swapped via `mock.patch` of `subproblem.FeralInterface`),
  asserting final-result rel-RMSE vs the reference solver (ma27 if available,
  else scipy) below 1e-2. Prints a problems × solvers RMSE comparison table,
  saves it to `tests/plots/linear_solver_rmse.csv`, and writes per-problem
  trajectory comparison plots.

The two are complementary: the direct test is the sharp correctness check
(cheap, tight tolerance); the end-to-end test covers integration
(BlockMatrix/BlockVector plumbing, symbolic caching) but measures
solver-vs-solver consistency through 300+ optimizer steps, so its RMSE
reflects optimization-path divergence as much as solver accuracy and its
tolerance is intentionally loose.

## 5. Fixes made while double-checking

- `feral-solver` was missing from all three dependency files even though
  `subproblem.py` imports it unconditionally — added (`>=0.9`).
- `pyproject.toml` pinned `pounce-solver>=0.3` while `environment.yml` and
  `requirements.txt` had `>=0.4` — aligned to `>=0.4` (activation-function
  support landed in 0.4).
- The swap-in test still patched `subproblem.ScipyInterface` after the code
  switched to instantiating `FeralInterface()` — the patch had become a
  no-op, so every "solver variant" would silently have run feral and the test
  would trivially pass. Patch target updated to `subproblem.FeralInterface`.
- Stale docstring in `subproblem.py` ("Symbolic factorization of MA27 done
  exactly once") updated.

## Caveats

- FERAL is research-grade, pre-1.0 (self-described). The swap-in tests are
  the guard; keep an eye on `solve_failed` warnings from `kkt_utils`.
- POUNCE follows upstream Ipopt's algorithm and options but is also pre-1.0;
  the swap-in test pins agreement at 1e-3 rel-RMSE.
- `SolverFactory('pounce')` requires the `pounce` executable on `PATH` —
  fresh shells/CI need the environment's `bin` directory active.
