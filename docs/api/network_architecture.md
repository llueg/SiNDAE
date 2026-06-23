# Defining a Network Architecture

In SiNDAE the neural network supplies the unknown term $z = f_{NN}(x)$ inside the
DAE. Because that network is embedded in a nonlinear program (NLP) and solved with
an interior-point method, the architecture is subject to requirements that ordinary
deep-learning models are not. This page explains those requirements, how the
built-in [`SimpleMLP`](nn_utils.md) satisfies them, and how to supply a fully custom
architecture.

## Requirements

The simultaneous formulation solves a single NLP over the states, the algebraic
variables, and the network parameters jointly. Interior-point solvers use the first
derivatives of every constraint and, on the exact-Hessian path, the second
derivatives as well. The network must therefore be **twice continuously
differentiable**. In practice this means two things:

- **Dense feed-forward layers.** The supported building block is a stack of affine
  layers (`weight @ x + bias`) with elementwise activations.
- **Smooth activations.** Use sigmoid-like, infinitely differentiable functions.
  SiNDAE ships `tanh`, `softplus`, and `swish`. Avoid `ReLU` and other
  piecewise-linear activations: their second derivative is zero almost everywhere
  with a kink at the origin, which impacts the Hessian information the solver
  relies on.

From {cite}`lueg2025simultaneous`:

> It is important to note that the formulation of the NLP and its solution with an
> interior point method makes certain restrictions on the type of neural network we
> can consider. Namely, we require the function $f_{NN}$ to have smooth second-order
> derivatives. In the description below, only dense feed-forward neural networks with 
> smooth activation functions, e.g. 'sigmoid', 'softplus', or similar.

## The built-in SimpleMLP

`SimpleMLP` is a dense feed-forward network built on
[Equinox](https://docs.kidger.site/equinox/). Configure its depth and width through
`widths`, and its nonlinearity through `activations`:

```python
import jax
from sindae import SimpleMLP

mlp = SimpleMLP(
    in_size=4,                                   # number of states fed to the NN
    out_size=1,                                  # number of learned terms z
    widths=[20, 20],                             # two hidden layers of width 20
    activations=[jax.nn.softplus, jax.nn.softplus],
    key=jax.random.PRNGKey(0),
)
```

`widths` is the list of hidden-layer sizes, and `activations` gives one smooth
activation per hidden layer (so `len(activations) == len(widths)`). The output layer
is always linear, which lets the network represent unbounded targets. A few
variations:

```python
# Deeper and narrower, with tanh
SimpleMLP(in_size=4, out_size=1, widths=[12, 12, 12],
          activations=[jax.nn.tanh] * 3, key=key)

# A single wide hidden layer, with swish
SimpleMLP(in_size=4, out_size=1, widths=[64],
          activations=[jax.nn.swish], key=key)
```

The three activations the package understands are `jax.nn.tanh`, `jax.nn.softplus`,
and `jax.nn.swish`. The reason the set is limited is explained next.

## How the network enters the NLP

SiNDAE embeds the network into the NLP in one of two ways, and they place different
demands on the architecture. The grey-box path is the key enabler for custom
networks, as noted in {cite}`lueg2025simultaneous`:

> It is not required to be able generate algebraic expressions encoding the neural
> network. An interface to evaluate the function, its Jacobian, and optionally its
> Hessian is sufficient.

| | Expression-writing | Grey-box (GBM) |
|--|--|--|
| Used by | `solve_simultaneous(use_gbm=False)` | `train_decomp`, `solve_simultaneous(use_gbm=True)` |
| NN representation | rewritten as explicit Pyomo algebraic expressions | evaluated by JAX (function, Jacobian, optional Hessian) |
| Hessian | exact | L-BFGS (simultaneous) or exact via autodiff (decomposition) |
| Architecture support | `SimpleMLP` only, supported activations | any smooth Equinox module |

The expression-writing backend walks the dense `SimpleMLP` layer by layer and writes
each affine map and activation as a Pyomo expression, so it is specific to that
structure and to the activations it knows how to translate into Pyomo. The grey-box
backend treats the network as a black box, only calling the network and its
autodiff derivatives, allowing for arbitrary smooth architectures.

## Input and output normalization

Normalization layers wrap the network's input and output and improve the robustness
of the solve. The model builders add them automatically. From {cite}`lueg2025simultaneous`:

> Normalization layers for the input and output of the neural network proved to aid
> the robustness of our approach. The associated constants can be computed from the
> initialization procedure.

Concretely, the smoother solve that precedes training produces an `InstanceData`
whose `input_mean` / `input_std` and `output_mean` / `output_std` become the fixed
normalization constants (the mean and standard deviation of the smoothed
trajectories). This is one reason the workflow always runs the smoother before the
main solve. See the [Simultaneous Solver](../simultaneous_solver.md) and
[Decomposition Solver](../decomposition_solver.md) pages.

## Adding a smooth activation

The built-in activations are limited to those that can be represented in both JAX and
Pyomo. To register another one for the expression-writing path, extend three small
maps in `sindae.nn_utils` and `sindae.algorithms.simultaneous.nn_block`:

- `_act_str2jax` and `_act_jax2str`: name to and from the JAX callable,
- `_act_str2pyo`: name to the equivalent Pyomo expression.

For the grey-box path no registration is needed. Any JAX-differentiable activation
can be used directly inside a custom module, described next.

## Custom architectures (grey-box and decomposition)

Because the grey-box backend only evaluates the network and its derivatives, you can
replace `SimpleMLP` with any Equinox module that

1. exposes integer `in_size` and `out_size` attributes, and
2. has a smooth, twice-differentiable `__call__(self, x)` mapping a length-`in_size`
   vector to a length-`out_size` vector.

Such a module works with `train_decomp` and `solve_simultaneous(use_gbm=True)`, as
well as with `solve_smoother` and `pretrain_mlp`, because the parameter flattening
(`flatten_fn` / `make_unflatten_fn`) and the autodiff in the grey-box models are
generic over Equinox modules. It does **not** work with the expression-writing path
`solve_simultaneous(use_gbm=False)`, which is specialized to `SimpleMLP`.

The example below adds a smooth residual connection:

```python
import jax
import equinox as eqx

class ResidualMLP(eqx.Module):
    layers: list
    in_size: int
    out_size: int

    def __init__(self, in_size, out_size, width, *, key):
        self.in_size, self.out_size = in_size, out_size
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(in_size, width, key=k1),
            eqx.nn.Linear(width, width, key=k2),
            eqx.nn.Linear(width, out_size, key=k3),
        ]

    def __call__(self, x):
        h = jax.nn.softplus(self.layers[0](x))
        h = h + jax.nn.softplus(self.layers[1](h))   # smooth residual connection
        return self.layers[2](h)                     # linear output

net = ResidualMLP(in_size=4, out_size=1, width=20, key=jax.random.PRNGKey(0))

# Trains exactly like SimpleMLP on the grey-box / decomposition path:
trained_m, mlp, history = train_decomp(
    problem, net, cfg, data, smoother_model=smoother_m,
)
```

Keep every activation smooth. On the decomposition path `jax.hessian` is taken
through the network, so a non-differentiable activation produces zero or undefined
second derivatives and the solve will stall.

## See also

- [Neural Network Utilities](nn_utils.md): the autodoc for `SimpleMLP`, `flatten_fn`,
  and `make_unflatten_fn`.
- [Simultaneous Solver](../simultaneous_solver.md) and
  [Decomposition Solver](../decomposition_solver.md): the two training backends.
