# Simultaneous Solver

`sindae.algorithms.simultaneous`

The simultaneous approach embeds NN parameters as decision variables in a single
large NLP and solves it with POUNCE (expression-writing, exact Hessian) or
cyipopt (GBM / L-BFGS).

## Configuration

```{autoclass} sindae.algorithms.simultaneous.train.SimultaneousConfig
:members:
:undoc-members:
```

## Model Builders

```{autofunction} sindae.algorithms.simultaneous.model_builder.build_simultaneous_model
```

```{autofunction} sindae.algorithms.simultaneous.model_builder.build_simultaneous_model_gbm
```

```{autofunction} sindae.algorithms.simultaneous.model_builder.extract_mlp
```

## Training

```{autofunction} sindae.algorithms.simultaneous.train.solve_simultaneous
```

---

## Usage Example

```python
# [Add your usage example here]
```
