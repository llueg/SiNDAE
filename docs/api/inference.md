# Inference

`sindae.algorithms.inference`

Embed a trained MLP back into a new (or the same) DAE problem and solve for
the trajectory consistent with the learned dynamics. The NN is enforced as a
hard GBM output constraint (default) or relaxed via an $\ell_1$ slack.

```{autofunction} sindae.algorithms.inference.make_inference_model
```

```{autofunction} sindae.algorithms.inference.solve_inference
```

---

## Usage Example

```python
# [Add your usage example here]
```
