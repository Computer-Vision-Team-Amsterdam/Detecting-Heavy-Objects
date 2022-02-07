"""
This module contains functionality to train a default detectron2 model.
"""

import argparse
import itertools
import os
import yaml
from pathlib import Path

from detectron2.engine import DefaultTrainer

from dataset import DATASET_NAME, register_dataset
from inference import setup_cfg

CONTAINER_DETECTION_MODEL = None


def init_train(flags) -> None:
    """
    Loads a pre-trained model and fine-tunes it on the data dataset
    """
    global CONTAINER_DETECTION_MODEL  # pylint: disable=global-statement
    config_file = Path(flags.config)

    cfg = setup_cfg(config_file)
    cfg.MODEL.DEVICE = flags.device
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    CONTAINER_DETECTION_MODEL = DefaultTrainer(cfg)
    CONTAINER_DETECTION_MODEL.resume_or_load(resume=False)
    CONTAINER_DETECTION_MODEL.train()


def collect_nested_lists(dictionary, composed_key, nested_lists):
    """
    This method parses a nested dictionary recursively and collects the (composed) keys where the value is a list.
    :param dictionary:
    :param composed_key:
    :param nested_lists:

    :return: keys and values of the @dictionary such that values are of type list.
    """

    for k, v in dictionary.items():
        if isinstance(v, dict):
            if composed_key == "":
                collect_nested_lists(v, k, nested_lists)
            else:
                collect_nested_lists(v, composed_key+"."+k, nested_lists)
        if isinstance(v, list):
            if composed_key == "":
                nested_lists[k] = v
            else:
                nested_lists[composed_key+"."+k] = v


    return nested_lists


def replace_lists(hyperparameters_yaml, info):
    for composed_key, value in info.items():
        keys = composed_key.split(".")
        levels = len(keys)
        for i in range(levels):




def handle_hyperparameters(config):

    # open yaml file as dict
    with open(config) as f:
        yaml_hyperparams = yaml.safe_load(f)

    hyperparams_grid_space = collect_nested_lists(yaml_hyperparams, "", {})

    for param_vals in itertools.product(*hyperparams_grid_space.values()):
        single_config = dict(zip(hyperparams_grid_space.keys(), param_vals))
        with open('configs/temp.yaml', 'w') as outfile:
            yaml.dump(single_config, outfile, default_flow_style=False)
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/")
    parser.add_argument("--config", type=str, default="configs/temp.yaml")
    parser.add_argument("--device", type=str, default="cpu")

    flags = parser.parse_args()


    #handle_hyperparameters("configs/hyperparameter-search.yaml")

    register_dataset(DATASET_NAME)
    init_train(flags)
