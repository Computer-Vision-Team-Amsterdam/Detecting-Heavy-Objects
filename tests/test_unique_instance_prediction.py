from unittest import TestCase

from visualizations.model import ModelPrediction
from visualizations.unique_instance_prediction import geo_clustering


class Test(TestCase):
    def test_geo_clustering(self) -> None:
        container_metadata_with_geohash = []
        pred_one = ModelPrediction(
            filename="",
            coords=(52.3508179052329, 5.0029046131412),
            geohash="u179cf5j85u0",
        )
        pred_two = ModelPrediction(
            filename="",
            coords=(52.3767806850864, 4.89314072016433),
            geohash="u173zqghf7w4",
        )
        pred_three = ModelPrediction(
            filename="",
            coords=(52.3472723292609, 4.91466061641611),
            geohash="u173zcd8zyud",
        )

        container_metadata_with_geohash.append(pred_one)
        container_metadata_with_geohash.append(pred_two)
        container_metadata_with_geohash.append(pred_three)
        prefix_length = 5

        expected_containter_info = []
        expected_one = ModelPrediction(
            filename="",
            coords=(52.3508179052329, 5.0029046131412),
            geohash="u179cf5j85u0",
            cluster=0,
        )
        expected_two = ModelPrediction(
            filename="",
            coords=(52.3767806850864, 4.89314072016433),
            geohash="u173zqghf7w4",
            cluster=1,
        )
        expected_three = ModelPrediction(
            filename="",
            coords=(52.3472723292609, 4.91466061641611),
            geohash="u173zcd8zyud",
            cluster=1,
        )

        expected_containter_info.append(expected_one)
        expected_containter_info.append(expected_two)
        expected_containter_info.append(expected_three)
        expected_nr_clusters = 2

        actual_container_info, actual_nr_clusters = geo_clustering(
            container_metadata_with_geohash, prefix_length=prefix_length
        )

        self.assertListEqual(expected_containter_info, actual_container_info)
        self.assertEqual(expected_nr_clusters, actual_nr_clusters)
