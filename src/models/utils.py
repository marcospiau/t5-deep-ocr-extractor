import numpy as np
import torch.nn as nn
from typing import Dict
from math import ceil


def get_num_parameters(model: nn.Module) -> Dict[str, int]:
    """Calculates number of parameters in a Pytorch Module.

    Args:
        model (nn.Module): Pytorch module.

    Returns:
        Dict[str, int]: with both total number of parameters and trainable.
    """

    sizes = np.array([p.numel() for p in model.parameters()])
    req_grad = np.array([p.requires_grad for p in model.parameters()])

    dict_ret = dict()
    dict_ret['all'] = sizes.sum()
    dict_ret['trainable_only'] = sizes[req_grad].sum()

    return dict_ret


def get_model_size_from_prefix(prefix):
    known_sizes = ['small', 'base', 'large']
    for size in known_sizes:
        if size in prefix:
            return size
    raise ValueError("Model size coudn't be inferred from model prefix. "
                     f"Known model sizes are {known_sizes}")


def get_accumulate_grad_batches(virtual_batch_size: int,
                                real_batch_size: int) -> int:
    """Calculation of how many batches with size `real_batch_size` should be
        accumulated in order to attain `virtual_batch_size` batch size.

    Args:
        virtual_batch_size (int): batch size after accumulation.
        real_batch_size (int): real batch size, before accumulation.

    Returns:
        int: how many batches should be accumulated before before each gradient
        optimization step.
    """
    assert (virtual_batch_size > real_batch_size)
    return ceil(virtual_batch_size / real_batch_size)
