from dataclasses import dataclass
from typing import Tuple


@dataclass
class PointOfInterest:
    pano_id: str
    coords: Tuple[float, float] = (-1, -1)  # lat, long
    geohash: str = ""
    cluster: int = 0
