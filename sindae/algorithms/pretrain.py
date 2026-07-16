"""
pretrain.py

Pretrain an MLP on smoother (or any other) arrays before the main training loop.
Usable with both the decomposition and simultaneous approaches.

API
---
  PretrainConfig — hyperparameters for supervised MLP pretraining

  pretrain_mlp(mlp, data, cfg) -> SimpleMLP
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from sindae.data_utils import InstanceData
from sindae.nn_utils import SimpleMLP, train_eqx_mlp


@dataclass
class PretrainConfig:
    """Hyperparameters for supervised MLP pretraining on smoother arrays.

    ``epochs=0`` runs no pretraining passes (the MLP is returned unchanged),
    which is how ``HybridDAE`` users opt out of the pretraining stage.
    """
    epochs:     int   = 200
    batch_size: int   = 32
    reg_coef:   float = 1e-1


def pretrain_mlp(
    mlp: SimpleMLP,
    data: InstanceData,
    cfg: PretrainConfig,
) -> SimpleMLP:
    """
    Pretrain the MLP on (nn_input, nn_output) pairs from ``data`` using SGD.

    Normalisation is applied internally using statistics from ``data``
    (``data.input_mean/std``, ``data.output_mean/std``).

    Parameters
    ----------
    mlp  : SimpleMLP
    data : InstanceData  (typically from the solved smoother model)
    cfg  : PretrainConfig

    Returns
    -------
    mlp : SimpleMLP  (updated weights)
    """
    all_inputs  = np.vstack(data.nn_input)
    all_outputs = np.vstack(data.nn_output)
    norm_in  = (all_inputs  - data.input_mean)  / data.input_std
    norm_out = (all_outputs - data.output_mean) / data.output_std

    mlp, _, _ = train_eqx_mlp(
        mlp=mlp,
        input_data=jnp.array(norm_in),
        output_data=jnp.array(norm_out),
        num_epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        reg_coef=cfg.reg_coef,
    )
    return mlp
