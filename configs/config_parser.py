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
        "--name",
        type=str,
        default="dummy_detectron",
        help="Name of Azure trained model to load",
    )
    parser.add_argument(
        "--version", type=int, default=2, help="Version of trained model."
    )

    flags, _ = parser.parse_known_args()
    return flags
