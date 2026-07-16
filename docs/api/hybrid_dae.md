# HybridDAE

`sindae.hybrid_dae`

The high-level fit/predict wrapper, and the primary way to use SiNDAE. One
`HybridDAE` object runs the whole pipeline: `fit(problem)` solves the smoother,
pretrains the network, and trains with the chosen method;
`predict(new_problem)` embeds the trained network in a new problem and solves
the inference NLP.

- [`HybridDAE`](#sindae.hybrid_dae.HybridDAE) selects the training approach
  (`method='simultaneous'` or `'decomposition'`) and the solver stack
  (`nlp_solver=`, `linear_solver=`). The network comes in as a prebuilt
  [`SimpleMLP`](nn_utils.md) (`net=`); defining it stays outside the wrapper.
- Each stage is configured with its config dataclass, the same objects the
  stage functions use: [`SmootherConfig`](smoother.md),
  [`PretrainConfig`](pretrain.md), [`SimultaneousConfig`](simultaneous.md) or
  [`DecompConfig`](decomp.md) (matching `method`), and
  [`SolverConfig`](solvers.md) for the fit-time NLP solver options. Everything
  is validated at construction, so a typo fails before any solve. The
  inference solve in `predict` does not inherit the constructor's
  `solver_options`; pass `predict(..., solver_options=...)` to tune it, so a
  bare `predict` matches a bare `solve_inference` call.
- Cross-cutting choices (`nlp_solver=`, `linear_solver=`, `solver_options=`,
  `unfix_io=`) live only on the wrapper, never inside a stage config, so no
  stage can silently override them.

After `fit`, the training solve's termination condition is on
`model.termination` (None for the decomposition method, whose per-step inner
solves are tracked in `model.history`), and any non-optimal solve raises a
`UserWarning`. Every intermediate stays reachable: the solved smoother model
(`smoother_model`), the normalization data (`smoother_data`), the solved
training model (`training_model`), its extracted trajectories
(`trained_data`), and the trained network (`net`). The stage functions
([`solve_smoother`](smoother.md), [`pretrain_mlp`](pretrain.md),
[`solve_simultaneous`](simultaneous.md), [`train_decomp`](decomp.md),
[`solve_inference`](inference.md)) remain the low-level escape hatch when you
need stage-level control.

## Usage

```python
import jax
import numpy as np
import sindae as sd

jax.config.update('jax_enable_x64', True)

problem = sd.LeslieGowerProblem(nfe=40, ncp=3)
sd.generate_data(problem, noise_std=np.array([0.05, 0.05]), obs_every=4)

mlp = sd.SimpleMLP(
    in_size=problem.input_dim, out_size=problem.z_dim,
    widths=[16, 16], activations=[jax.nn.softplus] * 2,
    key=jax.random.PRNGKey(0),
)

model = sd.HybridDAE(
    method="simultaneous",             # or "decomposition"
    nlp_solver="pounce",               # "ipopt" / "cyipopt" selectable
    linear_solver="feral",             # decomposition KKT solver; "ma27" / "scipy"
    net=mlp,
    train=sd.SimultaneousConfig(reg_coef=1e-3),  # DecompConfig for "decomposition"
    solver_options=sd.SolverConfig(tol=1e-6, max_iter=1000),
)
model.fit(problem)                     # smoother -> pretrain -> train
print(model.termination)               # "optimal"

# Predict under new initial conditions
new_problem = sd.LeslieGowerProblem(ics=np.array([[1.2, 0.15]]), nfe=40, ncp=3)
pred = model.predict(new_problem, slack_coef=1e-5)

mu_hat = model.net                     # the trained SimpleMLP

# Persist the trained network and its scaler, then reload
model.save("mu_hat.eqx")
reloaded = sd.HybridDAE.load("mu_hat.eqx")
pred = reloaded.predict(new_problem, slack_coef=1e-5)   # scaler restored

# Export for a foreign optimization tool
model.export("mu_hat.json")            # plain-text bundle, no extra dependencies
model.export("mu_hat.onnx")            # ONNX graph (normalized) + scaler sidecar
model.export("mu_hat.onnx", scaled=True)  # scaler baked in: raw-in, raw-out graph
net_def = model.to_omlt()              # in-memory OMLT NetworkDefinition
```

`save` writes the network weights, its architecture, and the four
normalization vectors the inference stage needs, so a reloaded model can
`predict` right away or warm-start a fresh `fit` from the loaded weights. The
stage configs and training trajectories are not persisted, so `load` is for
resuming or serving a trained network, not for reproducing the original solve
bit for bit.

`save`/`load` round-trip back into SiNDAE; `export` and `to_omlt` are a one-way
handoff to another modeling tool. Both carry the scaler, so the network is
evaluated in the space it was trained in. `export(path)` writes a file (`.json`
for a dependency-free bundle of weights, activations, scaler, input bounds, and
the ordered input/output variable names; `.onnx` for the graph plus a scaler
sidecar) and needs the matching extra (`pip install 'sindae[onnx]'`).
`to_omlt()` returns an `omlt.neuralnet.NetworkDefinition` whose inputs and
outputs are the raw physical variables (the normalization rides along as an
OMLT `OffsetScaling`), ready to drop into your own optimization model with an
OMLT formulation; it needs `pip install 'sindae[omlt]'`. The ONNX graph itself
stays in normalized space by default because OMLT applies the scaler separately;
pass `export(path, scaled=True)` to bake the scaler into the graph as affine
layers instead, giving a self-contained model that maps raw physical inputs to
raw physical outputs in any ONNX runtime (the sidecar's `scaling` field records
which contract applies, so a consumer never double-applies the scaler).

The smoother stage is configured the same way when its defaults are not right,
for example `smoother=sd.SmootherConfig(smooth_coef=10.0)` for noisier data.
Pretraining always runs; `pretrain=None` means `PretrainConfig()` (200 epochs),
and `pretrain=sd.PretrainConfig(epochs=0)` disables it. For partially observed
problems (unmeasured states with no data anchor), pass `unfix_io=False`.

## API reference

:::{include} _generated/hybrid_dae.md
:::
