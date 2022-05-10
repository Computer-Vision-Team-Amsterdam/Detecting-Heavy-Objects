from unittest import TestCase

import folium
import pytest
from folium.plugins import MarkerCluster

from visualizations.model import ModelPrediction

# from panorama.client import PanoramaClient  # pylint: disable=import-error

# from visualizations.unique_instance_prediction import geo_clustering


class Test(TestCase):
    @pytest.mark.skip(
        reason="temporarily skipping this because it requires geo_clustering"
    )
    def test_geo_clustering(self) -> None:
        container_metadata_with_geohash = []
        pred_one = ModelPrediction(
            pano_id="",
            coords=(52.3508179052329, 5.0029046131412),
            geohash="u179cf5j85u0",
        )
        pred_two = ModelPrediction(
            pano_id="",
            coords=(52.3767806850864, 4.89314072016433),
            geohash="u173zqghf7w4",
        )
        pred_three = ModelPrediction(
            pano_id="",
            coords=(52.3472723292609, 4.91466061641611),
            geohash="u173zcd8zyud",
        )

        container_metadata_with_geohash.append(pred_one)
        container_metadata_with_geohash.append(pred_two)
        container_metadata_with_geohash.append(pred_three)
        prefix_length = 5

        expected_containter_info = []
        expected_one = ModelPrediction(
            pano_id="",
            coords=(52.3508179052329, 5.0029046131412),
            geohash="u179cf5j85u0",
            cluster=0,
        )
        expected_two = ModelPrediction(
            pano_id="",
            coords=(52.3767806850864, 4.89314072016433),
            geohash="u173zqghf7w4",
            cluster=1,
        )
        expected_three = ModelPrediction(
            pano_id="",
            coords=(52.3472723292609, 4.91466061641611),
            geohash="u173zcd8zyud",
            cluster=1,
        )

        expected_containter_info.append(expected_one)
        expected_containter_info.append(expected_two)
        expected_containter_info.append(expected_three)
        expected_nr_clusters = 2

        # actual_container_info, actual_nr_clusters = geo_clustering(
        #    container_metadata_with_geohash, prefix_length=prefix_length
        # )
        actual_container_info, actual_nr_clusters = [], 1  # type: ignore
        self.assertListEqual(expected_containter_info, actual_container_info)
        self.assertEqual(expected_nr_clusters, actual_nr_clusters)

    @pytest.mark.skip(reason="this is just returning a map with sanity checks")
    def test_map_with_image(self) -> None:
        """This test check whether the icon in the map can contain an image"""
        # Amsterdam coordinates
        latitude = 52.377956
        longitude = 4.897070

        predictions = [
            [52.337909, 4.892184],
            [52.340400, 4.892549],
        ]

        # create empty map zoomed on Amsterdam
        Map = folium.Map(location=[latitude, longitude], zoom_start=12)

        marker_cluster = MarkerCluster().add_to(Map)

        panorama_id = "TMX7316010203-001698_pano_0001_004410"
        # image = PanoramaClient.get_panorama(panorama_id)
        image = None
        # link = image.links.equirectangular_small.href
        link = "None"
        score = 0.85
        html = (
            f"""
        <!DOCTYPE html>
        <html>
        <h2> Score is {score} </h2>
        <center><img src=\""""
            + link
            + """\" width=400 height=200 ></center>
        </html>
        """
        )
        for i in range(len(predictions)):

            popup = folium.Popup(folium.Html(html, script=True), max_width=500)
            icon = folium.Icon(color="red", icon="ok")
            folium.Marker(
                location=[predictions[i][0], predictions[i][1]],
                popup=popup,
                icon=icon,
                radius=15,
            ).add_to(marker_cluster)

        Map.save("test_jpg_icon.html")
