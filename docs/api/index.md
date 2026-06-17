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
| {doc}`problem` | `ProblemDefinition` |
| {doc}`data_utils` | `TrajectoryData`, `InstanceData`, `extract_instance_data`, `generate_data` |
| {doc}`nn_utils` | `SimpleMLP`, `flatten_fn`, `make_unflatten_fn` |
| {doc}`pretrain` | `PretrainConfig`, `pretrain_mlp` |
| {doc}`smoother` | `build_smoother_model`, `solve_smoother` |
| {doc}`simultaneous` | `build_simultaneous_model`, `solve_simultaneous`, `extract_mlp` |
| {doc}`decomp` | `DecompConfig`, `train_decomp`, `build_decomp_model` |
| {doc}`inference` | `make_inference_model`, `solve_inference` |

```{toctree}
:hidden:

problem
data_utils
nn_utils
pretrain
smoother
simultaneous
decomp
inference
```
