from rozvidrought_datasets.locator import point_to_pixel


class GridLocator:
    def locate_point(self, lon: float, lat: float) -> int:
        pixel_id = point_to_pixel(lon, lat)

        if pixel_id is None or pixel_id < 0:
            raise ValueError("Coordinate outside Rozvi grid")

        return int(pixel_id)