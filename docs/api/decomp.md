# Decomposition Solver

`sindae.algorithms.decomp`

The decomposition approach alternates between an inner cyipopt NLP solve (with
the NN embedded as a Grey-Box Model) and an outer Adam update driven by KKT
implicit differentiation. Supports MPI parallelism across trajectories.

## Configuration

```{autoclass} sindae.algorithms.decomp.train.DecompConfig
:members:
:undoc-members:
```

## Training

```{autofunction} sindae.algorithms.decomp.train.train_decomp
```

## Model Builder

```{autofunction} sindae.algorithms.decomp.model_builder.build_decomp_model
```

---

## Usage Example

```python
# [Add your usage example here]
```
