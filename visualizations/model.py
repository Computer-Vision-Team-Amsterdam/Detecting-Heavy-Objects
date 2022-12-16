from dataclasses import dataclass
from typing import Tuple


@dataclass
class PointOfInterest:
    pano_id: str
    coords: Tuple[float, float] = (-1, -1)
    geohash: str = ""
    cluster: int = 0
    closest_permit: Tuple[float, float] = None
