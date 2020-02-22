import logging

import torch
import torch.nn as nn

from retina_face_net.errors import WeightsLoadingError

logging.getLogger().setLevel(logging.INFO)


def load_model(model: nn.Module,
               weights_path: str,
               use_gpu: bool = False
               ) -> nn.Module:
    logging.info('Loading pre-trained model from {}'.format(weights_path))
    if use_gpu:
        device = torch.cuda.current_device()
        variables_dict = torch.load(
            weights_path,
            map_location=lambda storage, loc: storage.cuda(device)
        )
    else:
        variables_dict = torch.load(
            weights_path,
            map_location=lambda storage, loc: storage
        )
    if "state_dict" in variables_dict.keys():
        variables_dict = remove_prefix(variables_dict['state_dict'], 'module.')
    else:
        variables_dict = remove_prefix(variables_dict, 'module.')
    check_keys(model, variables_dict)
    model.load_state_dict(variables_dict, strict=False)
    return model


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    logging.info('Missing keys:{}'.format(len(missing_keys)))
    logging.info('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    logging.info('Used keys:{}'.format(len(used_pretrained_keys)))
    if len(used_pretrained_keys) <= 0:
        raise WeightsLoadingError("No weights can be loaded from pointed file.")
    return True


def remove_prefix(state_dict, prefix):
    remover = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {remover(key): value for key, value in state_dict.items()}

