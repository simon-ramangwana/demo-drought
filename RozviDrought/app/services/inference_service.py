from typing import Dict, Any

import pandas as pd

from app.services.grid_locator import GridLocator
from app.services.parquet_reader import MasterInputReader
from app.services.feature_service import FeatureService
from app.services.subsystem_service import SubsystemService
from app.services.fusion_service import FusionService


class InferenceService:
    def __init__(self):
        self.locator = GridLocator()
        self.reader = MasterInputReader(
            parquet_path=r"C:\Projects\Infer RozviDrought\data\master_inputs\master_inputs_long_198001_205012.parquet",
            pixels_per_block=26775,
        )
        self.feature_service = FeatureService()
        self.subsystem_service = SubsystemService()
        self.fusion_service = FusionService()

    def _prepare_hydrology_sequence(
        self,
        prepared: Dict[str, pd.DataFrame],
        run_yyyymm: int,
    ) -> pd.DataFrame:
        hyd = prepared["hydrology"].copy()
        hyd = hyd.sort_values("yyyymm").reset_index(drop=True)
        hyd_seq = hyd[hyd["yyyymm"] <= str(run_yyyymm)].copy()

        if hyd_seq.empty:
            raise ValueError(f"No hydrology history up to {run_yyyymm}")

        return hyd_seq

    def infer_point(
        self,
        lon: float,
        lat: float,
        scenario: str,
        yyyymm: int,
    ) -> Dict[str, Any]:
        pixel_id = self.locator.locate_point(lon, lat)

        current_row = self.reader.get_pixel_row(
            scenario=scenario,
            yyyymm=str(yyyymm),
            pixel_id=pixel_id,
        )

        timeseries = self.reader.get_pixel_timeseries(
            scenario=scenario,
            pixel_id=pixel_id,
        ).sort_values("yyyymm").reset_index(drop=True)

        prepared = self.feature_service.prepare_subsystem_inputs(
            timeseries=timeseries,
            run_yyyymm=yyyymm,
        )

        target_prepared = {}

        for name, df in prepared.items():
            if name == "hydrology":
                target_prepared[name] = self._prepare_hydrology_sequence(
                    prepared,
                    yyyymm,
                )
            else:
                target_prepared[name] = (
                    df[df["yyyymm"] == str(yyyymm)]
                    .copy()
                    .reset_index(drop=True)
                )
        hyd = target_prepared["hydrology"]
        print("HYD SHAPE:", hyd.shape)
        print("HYD COLS:", hyd.columns.tolist())
        print(hyd.tail(5).to_string())
        subsystem_outputs = self.subsystem_service.run_subsystems(
            target_prepared
        )

        if subsystem_outputs["hydrology"].empty:
            raise ValueError(
                f"Hydrology returned no rows for scenario={scenario}, "
                f"yyyymm={yyyymm}, pixel_id={pixel_id}"
            )

        fusion = self.fusion_service.run(
            subsystem_outputs=subsystem_outputs,
            model="hybrid",
        )

        return {
            "pixel_id": pixel_id,
            "current_row": current_row,
            "timeseries": timeseries,
            "prepared": target_prepared,
            "subsystem_outputs": subsystem_outputs,
            "fusion": fusion,
        }