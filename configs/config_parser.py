"""
This module is responsible for parsing console arguments.
"""
import argparse


def arg_parser() -> argparse.Namespace:
    """
    This method parses command line arguments.

    :returns: namespace with set flags.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_format", type=str, default="coco", help="coco, via")
    parser.add_argument(
        "--dataset", default="blurred-container-data", help="name of dataset on Azure"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="container",
        help="name of obj in instance segmentation",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data",
        help="name of the folder where data is located",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--inference", action="store_true", help="True if we infer on Azure"
    )
    parser.add_argument(
        "--name", required=True, type=str, help="Name of Azure trained model to load"
    )
    parser.add_argument("--subset", type=str, default="train", help="train, val, test")
    parser.add_argument(
        "--train", action="store_true", help="True if we train on Azure"
    )
    parser.add_argument("--version", type=int, help="Version of trained model.")

    flags, _ = parser.parse_known_args()

    print(50 * "=")
    print(f"train is {flags.train}")
    print(f"inference is {flags.inference}.")
    if flags.train and flags.inference:
        parser.error("Select either --train or --inference mode.")
    if flags.inference is True and not flags.version:
        parser.error(
            f"Specify which version of {flags.name} model to load by using --version."
        )
    if flags.train is True and flags.subset != "train":
        parser.error(
            "Training model should happen on the training subset. Set --subset to train."
        )

    return flags
