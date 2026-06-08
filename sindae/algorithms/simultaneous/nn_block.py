# type: ignore
import jax
import jax.numpy as jnp
import equinox as eqx
import pyomo.environ as pe
from typing import List, Callable, Optional
import numpy as np

from sindae.nn_utils import SimpleMLP, _act_jax2str, _act_str2jax


def _act_str2pyo(activation: str) -> Callable:
    if activation == "tanh":
        return pe.tanh
        # return lambda x: 1 - 2 / (1 + pe.exp(2 * x))
    elif activation == "softplus":
        return lambda x: pe.log(1 + pe.exp(x))
    elif activation == "swish":
        return lambda x: 0.5 * x * pe.tanh(0.5 * x) + 0.5 * x
        # return lambda x: x / (1 + pe.exp(-x))
    elif activation is None:
        return lambda x: x
    else:
        raise NotImplementedError(f"Activation {activation} not implemented.")


def default_init(shape, rng: np.random.Generator, lim: float) -> np.ndarray:
    # arr = rng.uniform(low=-lim, high=lim, size=shape)
    arr = rng.normal(loc=0, scale=1, size=shape)
    return arr


class LayerBlock:
    in_features: int
    out_features: int
    activation: str
    use_bias: bool
    _layer_block: pe.Block
    _weight_fix_threshold: float

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = None,
        use_bias: bool = True,
        weight_fix_threshold: float = 0,
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self._weight_fix_threshold = weight_fix_threshold
        self.use_bias = use_bias
        self._layer_block = pe.Block()
        lim = 1 / np.sqrt(in_features)
        init_w = default_init(shape=(out_features, in_features), rng=rng, lim=lim)
        self._layer_block.weight = pe.Var(
            range(out_features),
            range(in_features),
            within=pe.Reals,
            initialize=lambda b, i, j: init_w[i, j],
        )
        if use_bias:
            init_b = default_init(shape=(out_features,), rng=rng, lim=lim)
            self._layer_block.bias = pe.Var(
                range(out_features), within=pe.Reals, initialize=lambda b, i: init_b[i]
            )

        self._layer_block.construct()

    @property
    def weight(self) -> pe.Var:
        return self._layer_block.weight

    @weight.setter
    def weight(self, _w):
        assert _w.shape == (self.out_features, self.in_features), (
            "Weight shape must match the layer dimensions"
        )
        for i in range(self.out_features):
            for j in range(self.in_features):
                self.weight[i, j].value = _w[i, j]
                if np.abs(self.weight[i, j].value) < self._weight_fix_threshold:
                    self.weight[i, j].value = 0.0
                    self.weight[i, j].fix()

    @property
    def _w(self) -> np.ndarray:
        return np.array(
            [pe.value(self.weight[i, :]) for i in range(self.out_features)]
        ).reshape(self.out_features, self.in_features)

    @property
    def bias(self) -> pe.Var:
        if self.use_bias:
            return self._layer_block.bias
        return ValueError("No bias in this layer.")

    @bias.setter
    def bias(self, _b):
        assert _b.shape == (self.out_features,), (
            "Bias shape must match the layer dimensions"
        )
        for i in range(self.out_features):
            self.bias[i].value = _b[i]

    @property
    def _b(self) -> np.ndarray:
        if self.use_bias:
            return np.array(pe.value(self.bias[:])).reshape(self.out_features)
        return ValueError("No bias in this layer.")

    @property
    def block(self) -> pe.Block:
        return self._layer_block


class NNBlock:
    nn_block: pe.Block
    layers: List[LayerBlock]
    input_dim: int
    output_dim: int
    widths: List[int]
    activations: List[str]
    formulation: str

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        widths: List[int],
        activations: List[str],
        weight_fix_threshold: float = 0,
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.widths = widths
        self.activations = activations

        self.layers = [
            LayerBlock(
                in_features=input_dim,
                out_features=widths[0],
                activation=activations[0],
                weight_fix_threshold=weight_fix_threshold,
                rng=rng,
            )
        ]
        self.layers += [
            LayerBlock(
                in_features=widths[i],
                out_features=widths[i + 1],
                activation=activations[i + 1],
                weight_fix_threshold=weight_fix_threshold,
                rng=rng,
            )
            for i in range(len(widths) - 1)
        ]
        self.layers += [
            LayerBlock(
                in_features=widths[-1],
                out_features=output_dim,
                weight_fix_threshold=weight_fix_threshold,
                rng=rng,
            )
        ]

    def fix(self):
        for layer in self.layers:
            layer.weight.fix()
            if layer.use_bias:
                layer.bias.fix()


def make_nn_output_expression(
    nn_block: NNBlock,
    input_vars: List[pe.Var],
) -> pe.Expression:
    assert len(input_vars) == nn_block.input_dim, (
        "Number of input variables must match the input dimension of the neural network"
    )

    z_prev = {i: v for i, v in enumerate(input_vars)}
    for lix, layer in enumerate(nn_block.layers):
        z_expr = {}
        for j in range(layer.out_features):
            z_expr[j] = _act_str2pyo(layer.activation)(
                sum(layer.weight[j, i] * z_prev[i] for i in range(layer.in_features))
                + layer.bias[j]
            )
        z_prev = z_expr

    return z_expr




# ── NNBlock ↔ SimpleMLP conversion helpers ────────────────────────────────────

def nn_block_to_eqx_mlp(nn_block: NNBlock) -> SimpleMLP:
    """Convert a Pyomo NNBlock (after a simultaneous solve) to a JAX SimpleMLP."""
    mlp = SimpleMLP(
        in_size=nn_block.input_dim,
        out_size=nn_block.output_dim,
        widths=nn_block.widths,
        activations=[_act_str2jax(actf) for actf in nn_block.activations],
    )

    def is_linear(x):
        return isinstance(x, eqx.nn.Linear)
    def get_weights(m):
        return [x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
    def get_bias(m):
        return [x.bias  for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]

    new_weights = [jnp.array(layer._w) for layer in nn_block.layers]
    new_bias    = [jnp.array(layer._b) for layer in nn_block.layers]
    new_mlp = eqx.tree_at(get_weights, mlp, new_weights)
    new_mlp = eqx.tree_at(get_bias,    new_mlp, new_bias)
    return new_mlp


def eqx_mlp_to_nn_block(mlp: SimpleMLP, weight_fix_threshold: float = 0) -> NNBlock:
    """Convert a JAX SimpleMLP to a Pyomo NNBlock (for simultaneous model building)."""
    nn_block = NNBlock(
        input_dim=mlp.in_size,
        output_dim=mlp.out_size,
        widths=mlp.widths,
        weight_fix_threshold=weight_fix_threshold,
        activations=[_act_jax2str(actf) for actf in mlp.activations],
    )
    weights = [np.array(layer.weight) for layer in mlp.layers]
    bias    = [np.array(layer.bias)   for layer in mlp.layers]
    for i, layer in enumerate(nn_block.layers):
        layer.weight = weights[i]
        layer.bias   = bias[i]
    return nn_block

