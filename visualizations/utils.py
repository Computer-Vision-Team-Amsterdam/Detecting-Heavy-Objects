"""
Utility module containing shared code required to create visualizations,
that may be used for more generic postprocessing tasks
"""
import json
import os
import re
import xml
import xml.etree.ElementTree as Xet
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path
from typing import List, Tuple, Union

import geojson
import requests
from osgeo import osr  # pylint: disable-all
from tqdm import tqdm


def get_permit_locations(
    file: Union[Path, str], date_to_check: datetime
) -> Tuple[List[List[float]], List[str], List[str]]:
    """
    Returns all the containers permits from an decos objects permit file.
    """

    def is_permit_valid_on_day(permit: xml.etree.ElementTree.Element) -> bool:
        """
        Check whether container is valid on that day
        """
        try:
            start_date = permit.find("DATE6")
            end_date = permit.find("DATE7")

            start_date = datetime.strptime(
                start_date.text, "%Y-%m-%dT%H:%M:%S"  # type:ignore
            )
            end_date = datetime.strptime(
                end_date.text, "%Y-%m-%dT%H:%M:%S"  # type:ignore
            )

            # Check if permit is valid
            if end_date >= date_to_check >= start_date:  # type:ignore
                return True
        except Exception as e:
            print(f"There was an exception in the is_permit_valid_on_day function: {e}")

        return False

    def split_dutch_street_address(raw_address: str) -> List[str]:
        """
        This function separates an address string (street name, house number, house number extension and zipcode)
        into parts.
        Regular expression quantifiers:
        X?  X, once or not at all
        X*  X, zero or more times
        X+  X, one or more times
        """

        regex = "(\D+)\s+(\d+)\s?(.*)\s+(\d{4}\s*?[A-z]{2})"
        return re.findall(regex, raw_address)

    def is_container_permit(permit: xml.etree.ElementTree.Element) -> bool:
        """
        Check whether permit is for a container
        """
        container_words = [
            "puinbak",
            "container",
            "keet",
            "cabin",
        ]
        description = permit.find("TEXT8")
        try:
            if any(
                get_close_matches(word, container_words)
                for word in description.text.split(" ")  # type:ignore
            ):
                return True
        except Exception as e:
            print(f"There was an exception in the is_container_permit function: {e}")

        return False

    xmlparse = Xet.parse(file)
    root = xmlparse.getroot()
    permit_locations = []
    permit_keys = []
    permit_locations_failed = []
    print("Parsing the permits information")
    running_in_k8s = "KUBERNETES_SERVICE_HOST" in os.environ
    bag_url = f"https://api.data.amsterdam.nl/atlas/search/adres/?q="
    for item in tqdm(root, disable=running_in_k8s):
        # The permits seem to have a quite free format. Let's catch some exceptions
        if is_container_permit(item) and is_permit_valid_on_day(item):
            try:
                if len(item.getchildren()[-1]) > 0:  # Check if c_object exists
                    address = item.getchildren()[-1].getchildren()[0]
                    address_format = (
                        address.find("TEXT8").text + " " + address.find("INITIALS").text
                    )
                else:
                    address_raw = item.find("TEXT6").text
                    address = split_dutch_street_address(address_raw)
                    address_format = address[0][0] + " " + address[0][1]
            except Exception as ex:
                print(
                    f"XML scrape for item {item.find('ITEM_KEY').text} failed with error: {ex}."
                )
                permit_locations_failed.append(item.find("ITEM_KEY").text)
                # Continue to next iteration
                continue

            try:
                with requests.get(bag_url + address_format) as response:
                    bag_data_location = json.loads(response.content)["results"][0][
                        "centroid"
                    ]
                lonlat = [bag_data_location[1], bag_data_location[0]]
            except Exception as ex:
                print(
                    f"BAG scrape failed with error: {ex}. Address is {address_format}"
                )
                permit_locations_failed.append(item.find("ITEM_KEY").text)
                # Continue to next iteration
                continue

            permit_locations.append(lonlat)
            permit_keys.append(item.find("ITEM_KEY").text)
    return permit_locations, permit_keys, permit_locations_failed


def get_bridge_information(file: Union[Path, str]) -> List[List[List[float]]]:
    """
    Return a list of coordinates where to find vulnerable bridges and canal walls
    """

    def rd_to_wgs(coordinates: List[float]) -> List[float]:
        """
        Convert rijksdriehoekcoordinates into WGS84 cooridnates. Input parameters: x (float), y (float).
        """
        epsg28992 = osr.SpatialReference()
        epsg28992.ImportFromEPSG(28992)

        epsg4326 = osr.SpatialReference()
        epsg4326.ImportFromEPSG(4326)

        rd2latlon = osr.CoordinateTransformation(epsg28992, epsg4326)
        lonlatz = rd2latlon.TransformPoint(coordinates[0], coordinates[1])
        return [float(value) for value in lonlatz[:2]]

    bridges_coords = []
    with open(file) as f:
        gj = geojson.load(f)
    features = gj["features"]
    print("Parsing the bridges information")
    running_in_k8s = "KUBERNETES_SERVICE_HOST" in os.environ
    for feature in tqdm(features, disable=running_in_k8s):
        bridge_coords = []
        if feature["geometry"]["coordinates"]:
            for idx, coords in enumerate(feature["geometry"]["coordinates"][0]):
                bridge_coords.append(rd_to_wgs(coords))
            # only add to the list when there are coordinates
            bridges_coords.append(bridge_coords)
    return bridges_coords
