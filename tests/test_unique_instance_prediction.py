from unittest import TestCase

from visualizations.unique_instance_prediction import geo_clustering


class Test(TestCase):
    def test_geo_clustering(self):
        container_metadata_with_geohash = [
            {
                "coords": [52.3508179052329, 5.0029046131412],
                "score": 0.85,
                "geohash": "u179cf5j85u0",
            },
            {
                "coords": [52.3767806850864, 4.89314072016433],
                "score": 0.85,
                "geohash": "u173zqghf7w4",
            },
            {
                "coords": [52.3472723292609, 4.91466061641611],
                "score": 0.85,
                "geohash": "u173zcd8zyud",
            },
        ]
        prefix_length = 5

        expected_containter_info = [
            {
                "coords": [52.3508179052329, 5.0029046131412],
                "score": 0.85,
                "geohash": "u179cf5j85u0",
                "cluster": 0,
            },
            {
                "coords": [52.3767806850864, 4.89314072016433],
                "score": 0.85,
                "geohash": "u173zqghf7w4",
                "cluster": 1,
            },
            {
                "coords": [52.3472723292609, 4.91466061641611],
                "score": 0.85,
                "geohash": "u173zcd8zyud",
                "cluster": 1,
            },
        ]
        expected_nr_clusters = 2

        actual_container_info, actual_nr_clusters = geo_clustering(
            container_metadata_with_geohash, prefix_length=prefix_length
        )

        self.assertListEqual(expected_containter_info, actual_container_info)
        self.assertEqual(expected_nr_clusters, actual_nr_clusters)
