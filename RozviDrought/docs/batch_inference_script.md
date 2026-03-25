``` python
"""
Batch inference example.
Runs inference for multiple geometries.
"""

from shapely.geometry import Point

from rozvidrought_datasets.extractor import DatasetExtractor
from rozvidrought_inputs.builder import FeatureBuilder
import rozvidrought_subsystems as subs
from rozvidrought.cli import predict_spi3


def run_batch_inference():

    geometries = [
        Point(30.344238, -18.788573),
        Point(30.250000, -18.650000),
        Point(30.150000, -18.550000)
    ]

    extractor = DatasetExtractor({})

    builder = FeatureBuilder()

    for geometry in geometries:

        print("Running inference for:", geometry)

        df = extractor.run_drought_folder_pipeline(
            "rasters",
            geometry
        )

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
        print("-" * 50)


if __name__ == "__main__":

    run_batch_inference()

```