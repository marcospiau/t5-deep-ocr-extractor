import numpy as np
import torch.nn as nn
from typing import Dict


def get_num_parameters(model: nn.Module) -> Dict[str, int]:
    """Calcula número de parâmetros de um módulo Pytorch.

    Args:
        model (nn.Module): módulo Pytorch

    Returns:
        Dict[str, int]: Número de parâmetros
    """

    sizes = np.array([p.numel() for p in model.parameters()])
    req_grad = np.array([p.requires_grad for p in model.parameters()])

    dict_ret = dict()
    dict_ret['all'] = sizes.sum()
    dict_ret['trainable_only'] = sizes[req_grad].sum()

    return dict_ret
