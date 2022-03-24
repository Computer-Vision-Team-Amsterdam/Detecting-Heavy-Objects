import json
from pathlib import Path
import matplotlib.pyplot as plt


def plot_avg_bbox(data, output_dir):

    widths, heights, areas = [], [], []
    for ann in data["annotations"]:
        width = ann["bbox"][2]
        height = ann["bbox"][3]
        area = width * height

        widths.append(int(width))
        heights.append(int(height))
        areas.append(int(area))

    plt.figure()
    plt.scatter(widths, heights, alpha=0.5)
    plt.xlabel('width')
    plt.ylabel('height')
    plt.savefig(Path(output_dir, "Height_width.jpg"))

    plt.figure()
    plt.hist(areas, bins=30, range=(0, 50000))
    plt.xlabel("Container bbox area")
    plt.ylabel('Count')
    plt.savefig(Path(output_dir, "Areas.jpg"))


with open("/Users/dianaepureanu/Documents/data/train/containers-annotated-COCO-train.json") as f:
    data = json.load(f)
plot_avg_bbox(data, output_dir="/Users/dianaepureanu/Documents/Projects/Detecting-Heavy-Objects/visualizations")