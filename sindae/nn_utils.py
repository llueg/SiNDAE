import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import operator
from jax import flatten_util
# from tqdm.autonotebook import tqdm

from typing import List, Callable


def flatten_fn(mlp):
    """Flatten an equinox SimpleMLP's trainable parameters to a 1-D array."""
    params, _ = eqx.partition(mlp, eqx.is_array)
    flattened, _ = flatten_util.ravel_pytree(params)
    return flattened


def make_unflatten_fn(mlp):
    """Return a callable that rebuilds a SimpleMLP from a flat parameter array."""
    params, static = eqx.partition(mlp, eqx.is_array)
    _, _unflatten_fn = flatten_util.ravel_pytree(params)

    def unflatten_fn(flat_params):
        _params = _unflatten_fn(flat_params)
        return eqx.combine(static, _params)

    return unflatten_fn


def _act_str2jax(activation: str):
    if activation == "tanh":
        return jnp.tanh
    elif activation == "softplus":
        return jax.nn.softplus
    elif activation == "swish":
        return jax.nn.swish
    else:
        raise NotImplementedError(f"Activation {activation} not implemented.")


def _act_jax2str(activation: Callable):
    if activation == jnp.tanh:
        return "tanh"
    elif activation == jax.nn.softplus:
        return "softplus"
    elif activation == jax.nn.swish:
        return "swish"
    else:
        raise NotImplementedError(f"Activation {activation} not implemented.")


class SimpleMLP(eqx.Module):
    layers: List[eqx.nn.Linear]
    activations: List[Callable]
    in_size: int
    out_size: int
    widths: List[int]
    num_hidden_layers: int

    def __init__(
        self,
        in_size: int,
        out_size: int,
        widths: List[int],
        activations: List[Callable],
        *,
        key: jax.Array = jax.random.PRNGKey(0),
    ):
        self.in_size = in_size
        self.out_size = out_size
        self.widths = widths
        assert len(widths) == len(activations), (
            "Number of widths and activations must be the same."
        )
        self.num_hidden_layers = len(widths)
        self.activations = activations

        keys = jax.random.split(key, self.num_hidden_layers + 2)
        self.layers = [eqx.nn.Linear(in_size, widths[0], key=keys[0])]
        self.layers += [
            eqx.nn.Linear(widths[i], widths[i + 1], key=keys[i])
            for i in range(self.num_hidden_layers - 1)
        ]
        self.layers += [eqx.nn.Linear(widths[-1], out_size, key=keys[-1])]

    def __call__(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activations[i](x)
        x = self.layers[-1](x)
        return x


def train_eqx_mlp(
    mlp: SimpleMLP,
    input_data: jax.Array,
    output_data: jax.Array,
    num_epochs: int = 200,
    batch_size: int = 32,
    lr_schedule: optax.Schedule = optax.constant_schedule(1e-3),
    reg_coef: float = 0.01,
    reg_type: str = "l2",
    key: jax.Array = jax.random.PRNGKey(0),
):
    def is_linear(x):
        return isinstance(x, eqx.nn.Linear)
    def get_params(m):
        return [
            (x.weight, x.bias)
            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
            if is_linear(x)
        ]
    if reg_type == "l2":
        def reg_loss_fcn(model):
            return jax.tree.reduce(
                    operator.add,
                    jax.tree.map(lambda x: jnp.mean(jnp.square(x)), get_params(model)),
                )
    elif reg_type == "l1":
        def reg_loss_fcn(model):
            return jax.tree.reduce(
                    operator.add,
                    jax.tree.map(lambda x: jnp.mean(jnp.abs(x)), get_params(model)),
                )
    else:
        raise NotImplementedError(f"Regularization type {reg_type} not implemented.")

    def loss_fn(model, x, y, reg_coef=0.01):
        y_pred = jax.vmap(model)(x)
        loss = jnp.mean(jnp.square(y_pred - y))
        reg_loss = reg_loss_fcn(model)

        return loss + reg_coef * reg_loss

    def dataloader(key, x, y, batch_size):
        num_samples = x.shape[0]
        perm = jax.random.permutation(key, num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = perm[i : i + batch_size]
            yield x[batch_indices], y[batch_indices]

    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(eqx.filter(mlp, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, state, x, y, reg_coef):
        loss, grad = eqx.filter_value_and_grad(loss_fn)(model, x, y, reg_coef)
        val_loss = loss_fn(model, input_data, output_data, reg_coef)
        updates, new_state = optimizer.update(grad, state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_state, loss, val_loss

    loss_history = []
    val_loss_history = []

    key, shuffle_key = jax.random.split(key)
    for epoch in range(num_epochs):
        shuffle_key, subkey = jax.random.split(shuffle_key)
        for batch_x, batch_y in dataloader(
            subkey,
            input_data,
            output_data,
            batch_size=batch_size,
        ):
            mlp, opt_state, loss, val_loss = make_step(
                mlp, opt_state, batch_x, batch_y, reg_coef
            )
            loss_history.append(loss)
            val_loss_history.append(val_loss)

    return mlp, loss_history, val_loss_history


