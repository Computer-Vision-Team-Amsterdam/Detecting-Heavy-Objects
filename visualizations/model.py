from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelPrediction:
    filename: str
    coords: Tuple[float, float] = (-1, -1)
    geohash: str = ""
    cluster: int = 0
    score: float = 0.85
