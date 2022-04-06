"""
This model applies different post processing steps on the results file of the detectron2 model
"""
from visualizations.stats import DataStatistics


def discard_objects(predictions: str, smaller_than: float):
    """
    Filter out all predictions where area of the object is smaller than @param smaller_than
    """

    stats = DataStatistics(json_file=predictions)
    indices_to_keep = [idx for idx, area in enumerate(stats.areas) if area < smaller_than]
    predictions_to_keep = [stats.data[idx] for idx in indices_to_keep]

    return predictions_to_keep