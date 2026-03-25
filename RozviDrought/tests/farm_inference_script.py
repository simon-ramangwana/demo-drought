import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from shapely.geometry import Polygon

from app.services.polygon_inference_service import PolygonInferenceService


from pathlib import Path

# tests → RozviDrought → workspace root
WORKSPACE_DIR = Path(__file__).resolve().parents[2]

MASTER_PATH = (
    WORKSPACE_DIR
    / "data"
    / "master_inputs"
    / "master_inputs_long_198001_205012.parquet"
)

print("Using dataset:", MASTER_PATH)


def main():
    master_df = pd.read_parquet(MASTER_PATH)

    farm_geometry = Polygon([
        (30.00, -18.00),
        (30.10, -18.00),
        (30.10, -18.10),
        (30.00, -18.10),
        (30.00, -18.00),
    ])

    service = PolygonInferenceService(master_df=master_df)

    result = service.infer_polygon(
        geometry=farm_geometry,
        scenario="ssp245",
        yyyymm=202701,
        model="hybrid",
    )

    print("SUMMARY")
    print(result.summary)
    print("\nFIRST 5 CELL RESULTS")
    for row in result.cell_results[:5]:
        print(row)


if __name__ == "__main__":
    main()