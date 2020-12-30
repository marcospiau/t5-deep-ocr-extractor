import numpy as np
import torch.nn as nn
from typing import Dict


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
