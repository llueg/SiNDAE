# API Reference

All public symbols are importable directly from `sindae`:

```python
import sindae

sindae.SimpleMLP
sindae.ProblemDefinition
sindae.InstanceData
sindae.TrajectoryData
sindae.generate_data
sindae.solve_simultaneous    # via sindae.algorithms.simultaneous.train
sindae.train_decomp          # via sindae.algorithms.decomp.train
# ... etc.
```

---

## Module overview

| Module | Key symbols |
|--------|-------------|
| [problem](problem.md) | `ProblemDefinition` |
| [data_utils](data_utils.md) | `TrajectoryData`, `InstanceData`, `extract_instance_data`, `generate_data` |
| [nn_utils](nn_utils.md) | `SimpleMLP`, `flatten_fn`, `make_unflatten_fn` |
| [pretrain](pretrain.md) | `PretrainConfig`, `pretrain_mlp` |
| [smoother](smoother.md) | `build_smoother_model`, `solve_smoother` |
| [simultaneous](simultaneous.md) | `SimultaneousConfig`, `build_simultaneous_model`, `solve_simultaneous`, `extract_mlp` |
| [decomp](decomp.md) | `DecompConfig`, `train_decomp`, `build_decomp_model` |
| [inference](inference.md) | `make_inference_model`, `solve_inference` |
| [solvers](solvers.md) | `make_nlp_solver`, `make_linear_solver`, `NLPSolver`, `NLPResult` |

The full end-to-end workflow — problem → smoother → pre-training → training → inference —
is shown in the [examples gallery](../examples_gallery/index.md).
