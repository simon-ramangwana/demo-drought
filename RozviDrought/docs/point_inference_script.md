```python 
"""
Point inference example using rozvidrought packages.
Run from inside the RozviDrought environment.
"""

from shapely.geometry import Point

from rozvidrought_datasets.extractor import DatasetExtractor
from rozvidrought_inputs.builder import FeatureBuilder
import rozvidrought_subsystems as subs
from rozvidrought.cli import predict_spi3


def run_point_inference():

    geometry = Point(30.344238, -18.788573)

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

    run_point_inference()


```