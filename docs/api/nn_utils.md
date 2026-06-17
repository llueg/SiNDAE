# Neural Network Utilities

`sindae.nn_utils`

MLP architecture and parameter utilities built on
[Equinox](https://docs.kidger.site/equinox/).

## Network

```{autoclass} sindae.nn_utils.SimpleMLP
:members:
:undoc-members:
:show-inheritance:
:special-members: __init__, __call__
```

## Parameter Utilities

These helpers convert between the Equinox parameter pytree and a flat 1-D NumPy
array — the representation used by the decomposition KKT utilities and the
simultaneous NLP expression-writing backend.

```{autofunction} sindae.nn_utils.flatten_fn
```

```{autofunction} sindae.nn_utils.make_unflatten_fn
```

---

## Usage Example

```python
# [Add your usage example here]
```
