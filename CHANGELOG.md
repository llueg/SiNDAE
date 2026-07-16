# Changelog

All notable user-visible changes to SiNDAE. The format follows
[Keep a Changelog](https://keepachangelog.com/); the project is pre-release.

## [Unreleased]

### Added
- `HybridDAE`: high-level scikit-learn-style fit/predict wrapper over the full
  pipeline (smoother, pretraining, simultaneous or decomposition training,
  inference), with solver selectors (`nlp_solver=`, `linear_solver=`,
  `solver_options=`) and per-stage configuration (`net=`, `smoother=`,
  `pretrain=`, `train=`, `unfix_io=`). The network comes in as a prebuilt
  `SimpleMLP` (`net=`); stage configuration is passed as config dataclasses
  (`SmootherConfig`, `PretrainConfig`, `SimultaneousConfig` / `DecompConfig`,
  `SolverConfig`) and validated at construction. Pretraining runs by default
  (`PretrainConfig(epochs=0)` disables it). The constructor's `solver_options`
  configures the fit-time solves (smoother and training); `predict` takes its
  own `solver_options` and does not inherit the fit-time ones, so a bare
  `predict` matches a bare `solve_inference` call. After `fit`,
  `model.termination` holds the training solve's termination condition, and any
  non-optimal solve raises a `UserWarning`. Importable as `sindae.HybridDAE`.
- New config dataclasses: `SmootherConfig` (smoother-stage hyperparameters) and
  `SolverConfig` (typed NLP solver options; accepted interchangeably with a
  plain dict by `make_nlp_solver` and the stage functions).
- `train_decomp` now accepts `unfix_io=` and forwards it to the decomposition
  model build (previously only `build_decomp_model` exposed it).
- `HybridDAE.save(path)` / `HybridDAE.load(path)`: persist a fitted model as a
  one-line JSON manifest (architecture, activation names, and the four
  normalization vectors) followed by the Equinox leaf arrays. `load` is a
  classmethod returning a fitted wrapper that can `predict` immediately (the
  scaler is restored) or warm-start a fresh `fit` from the loaded weights. The
  stage configs and training trajectories are not persisted.
- New `sindae.NormStats` dataclass: the four normalization vectors
  (`input_mean`/`input_std`/`output_mean`/`output_std`) that the inference stage
  consumes. A drop-in for `InstanceData` wherever only normalization statistics
  are needed; restored onto `smoother_data` by `HybridDAE.load`.
- `HybridDAE.export(path, format=None, scaled=False)`: one-way export of a
  trained network to a file for a foreign optimization tool. `format='json'`
  writes a dependency-free bundle (weights, activations, scaler, data-derived
  input bounds, and the ordered input/output variable-name contract);
  `format='onnx'` writes the network graph plus a `<path>.json` scaler sidecar.
  For ONNX, `scaled=True` bakes the scaler into the graph as affine layers so the
  exported model maps raw physical inputs to raw physical outputs (self-contained
  inference, no sidecar arithmetic); the sidecar's `scaling` field records
  `"baked"` or `"external"` so a consumer never double-applies the scaler. The
  format is inferred from the path suffix when omitted. The `onnx` target needs
  the new `onnx` extra (`pip install 'sindae[onnx]'`).
- `HybridDAE.to_omlt()`: build an in-memory `omlt.neuralnet.NetworkDefinition`
  whose inputs/outputs are the raw physical variables (the normalization is
  attached as an OMLT `OffsetScaling`, and the data-derived input bounds as its
  `scaled_input_bounds`). Needs the new `omlt` extra
  (`pip install 'sindae[omlt]'`).
- After `fit`, `HybridDAE.io_names` records the ordered NN input/output variable
  names (e.g. `{"inputs": ["x[0]", "x[1]"], "outputs": ["z"]}`); it is persisted
  by `save`/`load` and carried into every export.
- New optional extras `onnx` (`jax2onnx`, `onnxruntime`) and `omlt` (`omlt`) for
  model export. The core install does not depend on them and never imports them.

### Changed
- The stage functions' NLP backend selector is now `nlp_solver=` (was
  `backend=`) on `solve_smoother`, `generate_data`, `solve_simultaneous`,
  `solve_inference`, `train_decomp`, and `TrajectoryBatchSubproblem`.
- The solver-options argument is now `solver_options=` everywhere (was
  `pounce_options=` on `solve_smoother`, `generate_data`, and
  `solve_simultaneous`).
- `PretrainConfig.epochs` default changed from 400 to 200 (the value every
  example uses; it is also the wrapper's default pretraining length).
