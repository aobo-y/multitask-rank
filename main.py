"""
SouthPark Chatbot
"""

import os
import argparse
import torch
import json

import config
from src.utils import CheckpointManager, Vocabulary
from src.models import MultiTaskMLP

DIR_PATH = os.path.dirname(__file__)
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


def build_model(model_config, checkpoint):
    if model_config.MODEL_TYPE == 'MultiTaskMLP':
        model = MultiTaskMLP(
            config.N_USERS,
            config.N_ITEMS,
            model_config.EBD_SIZE,

            config.N_TASKS,
            model_config.SHARED_LAYER_SIZES,
            model_config.TASK_LAYER_SIZES
        )
    else:
        raise Exception('invalid model name')

    if checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif model_config.MODEL_TYPE == 'NNR' \
        or model_config.MODEL_TYPE == 'ReviewRanker':
        model.load_pretrained_word_ebd(pre_we_weight)

    # Use appropriate device
    model = model.to(device)

    return model


def init(mdl_name=None, ckpt_name=None):
    if not mdl_name:
        mdl_name=config.DEFAULT_MODEL_NAME

    SAVE_PATH = os.path.join(DIR_PATH, config.SAVE_DIR, mdl_name)
    print('Saving path:', SAVE_PATH)

    ckpt_mng = CheckpointManager(SAVE_PATH)

    checkpoint = None
    if ckpt_name:
        print('Load checkpoint:', ckpt_name)
        checkpoint = ckpt_mng.load(ckpt_name, device)

    model_config = config.load(mdl_name)
    model = build_model(model_config, checkpoint)

    return model, {
        'checkpoint': checkpoint,
        'ckpt_mng': ckpt_mng,
        'model_config': model_config
    }

