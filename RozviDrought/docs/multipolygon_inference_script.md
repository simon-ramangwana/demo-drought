``` python 
"""
MultiPolygon inference example.
Useful for districts or multiple regions.
"""

from shapely.geometry import Polygon, MultiPolygon

from rozvidrought_datasets.extractor import DatasetExtractor
from rozvidrought_inputs.builder import FeatureBuilder
import rozvidrought_subsystems as subs
from rozvidrought.cli import predict_spi3


def run_multipolygon_inference():

    poly1 = Polygon([
        (30.0, -18.0),
        (30.1, -18.0),
        (30.1, -18.1),
        (30.0, -18.1),
        (30.0, -18.0)
    ])

    poly2 = Polygon([
        (30.2, -18.2),
        (30.3, -18.2),
        (30.3, -18.3),
        (30.2, -18.3),
        (30.2, -18.2)
    ])

    geometry = MultiPolygon([poly1, poly2])

    extractor = DatasetExtractor({})

    df = extractor.run_drought_folder_pipeline(
        "rasters",
        geometry
    )

    builder = FeatureBuilder()

    features = builder.build(df)

    atm = subs.predict_atmospheric_proba(features)
    soil = subs.predict_soil_proba(features)
    veg = subs.predict_veg_proba(features)
    hyd = subs.predict_hydro_proba(features)

    result = predict_spi3(
        atmospheric=atm,
        soil=soil,
        vegetation=veg,
        hydrology=hyd,
        model="hybrid"
    )

    print(result)


if __name__ == "__main__":

    run_multipolygon_inference()
```