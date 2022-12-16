from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PointOfInterest:
    pano_id: str
    coords: Tuple[float, float] = (-1, -1)
    geohash: str = ""
    cluster: int = 0
    closest_permit: Tuple[float, float] = (-1.0, -1.0)
    score: float = 0.0
