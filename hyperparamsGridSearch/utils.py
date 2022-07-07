"""
This module contains functionality to handle configuration files where values are lists.
From the lists we compute the Cartezian Product and use the combinations in a grid search.
"""

import itertools
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


def collect_nested_lists(
    dictionary: Dict[str, Any],
    composed_key: str,
    nested_lists: Dict[str, List[str]],
) -> Dict[str, List[str]]:
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
                collect_nested_lists(v, composed_key + "." + k, nested_lists)
        if isinstance(v, list):
            if composed_key == "":
                nested_lists[k] = v
            else:
                nested_lists[composed_key + "." + k] = v

    return nested_lists


def generate_config_file(file: Any, configuration: Dict[Any, Any], name: int) -> None:
    for composed_name, value in configuration.items():
        names = composed_name.split(".")
        if len(names) == 1:
            file[names[0]] = value
        if len(names) == 2:
            file[names[0]][names[1]] = value
        if len(names) == 3:
            file[names[0]][names[1]][names[2]] = value

    with open(f"configs/temp_{name}.yaml", "w") as outfile:
        yaml.dump(file, outfile, sort_keys=False)
    outfile.close()


def handle_hyperparameters(config: Union[str, Path]) -> int:
    # open yaml file as dict
    with open(config) as f:
        file = yaml.safe_load(f)
    f.close()
    # get rows for which we do hyperparameter search
    grid_space = collect_nested_lists(file, "", {})

    count = 0
    for combination in itertools.product(*grid_space.values()):
        configuration = dict(zip(grid_space.keys(), combination))
        generate_config_file(file, configuration, count)
        count = count + 1

    return count
