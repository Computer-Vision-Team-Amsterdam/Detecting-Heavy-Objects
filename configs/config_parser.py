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

    parser.add_argument(
        "--config", type=str, default="configs/container_detection.yaml"
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="coco",
        help="Annotations format. Options: coco, via",
    )
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
        default="data_sample",
        help="name of the folder where data is located",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--image",
        type=str,
        default="blurred.jpg",
        help="Path to image on which to run a prediction",
    )
    parser.add_argument(
        "--name", default="best", type=str, help="Name of Azure trained model to load"
    )
    parser.add_argument("--subset", type=str, default="test", help="train, val, test")
    parser.add_argument("--version", default="latest", help="Version of trained model.")
    parser.add_argument(
        "--mode",
        type=str,
        default="ann",
        help="options: ann, pred. Whether to visualize annotated containers or"
        "predicted containers",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=50,
        help="Number of sample images to plot for one model",
    )

    parser.add_argument("--output_path", type=str, help="Full path to output folder")
    parser.add_argument("--weights", type=str, help="Full path to weight file")

    flags, _ = parser.parse_known_args()

    return flags
