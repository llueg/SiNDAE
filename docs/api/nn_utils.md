# Neural Network Utilities

`sindae.nn_utils`

MLP architecture and parameter utilities built on
[Equinox](https://docs.kidger.site/equinox/).

- [`SimpleMLP`](#sindae.nn_utils.SimpleMLP) is the built-in dense feed-forward network
  used as the unknown term $z = f_{NN}(x)$. See
  [Defining a Network Architecture](network_architecture.md) for the requirements it
  satisfies and how to supply a custom module.
- [`flatten_fn`](#sindae.nn_utils.flatten_fn) and
  [`make_unflatten_fn`](#sindae.nn_utils.make_unflatten_fn) convert between the Equinox
  parameter pytree and a flat 1-D array — the representation used by the decomposition KKT
  utilities and the simultaneous NLP backend.

## Usage

```python
import jax
import jax.numpy as jnp
from sindae import SimpleMLP, flatten_fn, make_unflatten_fn

mlp = SimpleMLP(
    in_size=2, out_size=1,
    widths=[16, 16],                              # two hidden layers
    activations=[jax.nn.softplus, jax.nn.softplus],
    key=jax.random.PRNGKey(0),
)

y = mlp(jnp.array([0.5, 1.0]))                    # forward pass -> shape (1,)

flat    = flatten_fn(mlp)                         # trainable params as a 1-D array
rebuild = make_unflatten_fn(mlp)                  # inverse: flat array -> SimpleMLP
mlp2    = rebuild(flat)                           # round-trips back to the same network
```

## API reference

:::{include} _generated/nn_utils.md
:::
