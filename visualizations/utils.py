"""
Utility module containing shared code required to create visualizations,
that may be used for more generic postprocessing tasks
"""
import os
import re
import xml
import xml.etree.ElementTree as Xet
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path
from typing import Any, List, Union
import requests
import json

import geojson
from osgeo import osr  # pylint: disable-all
from tqdm import tqdm


def get_permit_locations(
    file: Union[Path, str], date_to_check: datetime
) -> List[List[float]]:
    """
    Returns all the containers permits from an decos objects permit file.
    """

    def is_form_valid(permit: xml.etree.ElementTree.Element) -> bool:
        """
        Check if a form has all the requirements
        """
        address = permit.find("TEXT6")
        description = permit.find("TEXT8")
        start_date = permit.find("DATE6")
        end_date = permit.find("DATE7")
        if (
            address.text  # type:ignore
            and description.text  # type:ignore
            and start_date.text  # type:ignore
            and end_date.text  # type:ignore
        ):
            return True
        return False

    def is_permit_valid_on_day(permit: xml.etree.ElementTree.Element) -> bool:
        """
        Check whether container is valid on that day
        """
        start_date = permit.find("DATE6")
        end_date = permit.find("DATE7")

        start_date = datetime.strptime(  # type:ignore
            start_date.text, "%Y-%m-%dT%H:%M:%S"  # type:ignore
        )
        end_date = datetime.strptime(end_date.text, "%Y-%m-%dT%H:%M:%S")  # type:ignore

        # Check if permit is valid
        if end_date >= date_to_check >= start_date:  # type:ignore
            return True
        return False

    def remove_postal_code(address: str) -> str:
        """
        geopy/Nominatim only recognize (Dutch) postal codes with a space in them between the digits and letters.
        Postal codes may or may not be present in the Decos permit, and may or may not be formatted using a space. They
        are, however, not required to retrieve the coordinates of an address: the combination of street name, house
        number, and place name suffices.

        This function removes the postal code, if present, from an address, and returns the cleaned up string.
        """
        postal_code_ex = r"\d{4}\s*?[A-z]{2}"  # 4 digits, any amount of whitespace, and 2 letters (case-insensitive)
        return re.sub(postal_code_ex, "", address).strip()

    def split_dutch_street_address(address: str) -> List:
        """
        TODO use a function like this
        This function separates an address string (street name, house number, house number extension and zipcode)
        into parts.

        Regular expression quantifiers:
        X?  X, once or not at all
        X*  X, zero or more times
        X+  X, one or more times
        """

        regex = "(\D+)\s+(\d+)\s?(.*)\s+(\d{4}\s*?[A-z]{2})"
        return re.findall(regex, address)

    def is_container_permit(permit: Any) -> bool:
        """
        Check whether permit is for a container
        """
        # TODO remove *container* -> "puinbak", "container", "keet",
        container_words = [
            "puinbak",
            "puincontainer",
            "container",
            "afvalcontainer",
            "zeecontainer",
            "keet",
            "schaftkeet",
            "vuilcontainer",
        ]
        description = permit.find("TEXT8")
        if any(
            get_close_matches(word, container_words)
            for word in description.text.split(" ")
        ):
            return True

        return False

    xmlparse = Xet.parse(file)
    root = xmlparse.getroot()
    permit_locations = []
    print("Parsing the permits information")
    running_in_k8s = "KUBERNETES_SERVICE_HOST" in os.environ
    bag_url = f"https://api.data.amsterdam.nl/atlas/search/adres/?q="
    for item in tqdm(root, disable=running_in_k8s):
        # The permits seem to have a quite free format. Let's catch some exceptions
        if (
            is_form_valid(item)
            and is_container_permit(item)
            and is_permit_valid_on_day(item)
        ):
            try:
                # todo: use split_dutch_street_address(item.find("TEXT6").text)
                address = remove_postal_code(item.find("TEXT6").text).replace(" ", "%20")

                with requests.get(bag_url + address) as response:
                    bag_data_location = json.loads(response.content)['results'][0]['centroid']

                lonlat = [bag_data_location[1], bag_data_location[0]]
                permit_locations.append(lonlat)
            except Exception as ex:
                raise ex
    return permit_locations


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
