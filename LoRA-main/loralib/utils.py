#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """

    @param model:
    @param bias:
    @return:
    """
    # The function first iterates over all the named parameters in the model using model.named_parameters().
    # If the parameter name contains the string lora_, its requires_grad attribute is set to True to make it trainable.
    # Conversely, all parameters that do not have lora_ in their name are set to False to make them non-trainable.
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    # The bias argument controls whether bias parameters should be included or excluded from the LoRA layers.
    # If the argument is 'none', only the LoRA layers are marked as trainable. If it is 'all', both the LoRA layers and the bias parameters are marked as trainable.
    # If it is 'lora_only', only the LoRA layers and their corresponding bias parameters are marked as trainable.

    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError



def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    """
     lora_state_dict is a function that returns a dictionary containing only the state of LoRA weights in the model.
     This function is useful when we want to manipulate the state or configuration of the attention weights, but we do not want to modify the rest of the model's state.
     The function first retrieves the entire state dictionary of the given PyTorch model using model.state_dict().
     Then, based on the bias parameter, it returns a subset of the state dictionary. If bias is set to 'none', only the LoRA weights
    @param model:
    @param bias:
    @return:
    """
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
