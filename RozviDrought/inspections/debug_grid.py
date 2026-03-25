from rozvidrought_datasets.grid import bounds, coord_to_row_col, coord_to_pixel_id
from rozvidrought_datasets.locator import point_to_pixel, point_to_row_col

TESTS = [
    (31.05, -17.83),
    (29.50, -19.00),
    (30.00, -20.00),
]

print("GRID BOUNDS:")
print(bounds)

for lon, lat in TESTS:
    print("\n" + "=" * 60)
    print(f"TEST lon={lon}, lat={lat}")

    for name, fn in [
        ("grid.coord_to_row_col", lambda: coord_to_row_col(lon, lat)),
        ("grid.coord_to_pixel_id", lambda: coord_to_pixel_id(lon, lat)),
        ("locator.point_to_row_col", lambda: point_to_row_col(lon, lat)),
        ("locator.point_to_pixel", lambda: point_to_pixel(lon, lat)),
    ]:
        try:
            print(name, "->", fn())
        except Exception as e:
            print(name, "-> ERROR:", e)