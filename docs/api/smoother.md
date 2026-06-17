# Smoother

`sindae.algorithms.smoother`

The smoother NLP fits a smooth trajectory to noisy observations and produces
warm-start values and normalization statistics for subsequent training. It
penalises the time derivative of the NN output variable $z_\text{smooth}$
weighted by `smooth_coef`.

```{autofunction} sindae.algorithms.smoother.build_smoother_model
```

```{autofunction} sindae.algorithms.smoother.solve_smoother
```

---

## Usage Example

```python
# [Add your usage example here]
```
