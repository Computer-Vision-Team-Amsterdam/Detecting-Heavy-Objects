"""
This model performs splitting into train, validation and test based on the coordinates of the images.
"""

import json
from pathlib import Path
from typing import List

from visualizations.stats import DataStatistics


class CoordinateSplit:
    """
   This class clusters coordinates and splits the data into train, validation and test.

   We start from the dataset which was split based on features.
   We start there so we do not have to do all the other preprocessing on the images (2000x1000 vs 4000x2000) or on the
   annotation files (missing keys, area of 0)
   """

    def __init__(self):
        pass

    def _create_new_folders(self):
       pass

    def _move_faulty_panoramas(self):
       pass

    def _get_predictions_objects(self):
       pass

    def _get_clustered_prediction_objects(self):
       pass

    def _move_images_based_on_cluster(self):
       pass

    def _regenerate_annotation_files(self):
       pass

    def _move_annotation_files(self):
       pass

    def main(self):
      self._create_new_folders()
      self._move_faulty_panoramas()
      self._get_predictions_objects()
      self._get_clustered_prediction_objects()
      self._move_images_based_on_cluster()
      self._regenerate_annotation_files()
      self._move_annotation_files()