from typing import List
from shapely.geometry import Polygon
from shapely.errors import GEOSException


class AnnotationFilter:
    """Annotation과 Query polygon이 얼마나 겹친지 확인하는 클레스
    - True인 경우, 겹침이 없음을 의미.
    """

    def __init__(self, label_polygons: List, threshold: float = 0.05) -> None:
        self.label_polygons = label_polygons
        self.threshold = threshold

    def __call__(self, polygon: Polygon) -> bool:
        for label_polygon in self.label_polygons:
            try:
                intersection: Polygon = label_polygon.intersection(polygon)
            except GEOSException:
                label_polygon = label_polygon.buffer(0)
                intersection: Polygon = label_polygon.intersection(polygon)

            if intersection.is_empty:
                continue

            overlap = intersection.area / polygon.area
            if overlap > self.threshold:
                return False

        return True



