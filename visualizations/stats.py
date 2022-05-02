import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt

import utils


class DataStatistics:
    """
    Compute and visualize different statistics from the data
    """

    def __init__(self, json_file, output_dir=None):
        """
        json file can be either COCO annotation or COCO results file
        """
        with open(json_file) as f:
            self.data = json.load(f)
        f.close()

        self.output_dir = output_dir
        self.widths, self.heights = utils.collect_dimensions(self.data)
        self.areas = [
            width * height for width, height in zip(self.widths, self.heights)
        ]

    def plot_dimensions_distribution(self, plot_name: str):
        """
        Scatter plot with height and widths of containers

         plot_name : includes the file type, i.e. jpg
        """
        if self.output_dir is None:
            raise ValueError("output_dir cannot be None")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.scatter(self.widths, self.heights, alpha=0.5)
        plt.xlabel("width")
        plt.ylabel("height")
        plt.savefig(Path(self.output_dir, plot_name))

    def plot_areas_distribution(self, plot_name: str):
        """
        Histogram with containers areas
        plot_name : includes the file type, i.e. jpg
        """

        if self.output_dir is None:
            raise ValueError("output_dir cannot be None")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.hist(self.areas, bins=30, range=(0, 50000))
        plt.xlabel("Container bbox area")
        plt.ylabel("Count")
        plt.savefig(Path(self.output_dir, plot_name))
