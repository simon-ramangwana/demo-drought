import pandas as pd
from rozvidrought.cli import predict_spi3


class FusionService:
    def _build_fusion_row(self, subsystem_outputs: dict[str, pd.DataFrame]) -> dict:
        atm = subsystem_outputs["atmospheric"].iloc[0]
        soil = subsystem_outputs["soil"].iloc[0]
        veg = subsystem_outputs["vegetation"].iloc[0]
        hyd = subsystem_outputs["hydrology"].iloc[0]

        return {
            "atm_p0": float(atm["p0"]),
            "atm_p1": float(atm["p1"]),
            "atm_p2": float(atm["p2"]),
            "atm_p3": float(atm["p3"]),
            "soil_p0": float(soil["p0"]),
            "soil_p1": float(soil["p1"]),
            "soil_p2": float(soil["p2"]),
            "soil_p3": float(soil["p3"]),
            "veg_p0": float(veg["p0"]),
            "veg_p1": float(veg["p1"]),
            "veg_p2": float(veg["p2"]),
            "veg_p3": float(veg["p3"]),
            "hyd_p0": float(hyd["p0"]),
            "hyd_p1": float(hyd["p1"]),
            "hyd_p2": float(hyd["p2"]),
            "hyd_p3": float(hyd["p3"]),
        }

    def run(self, subsystem_outputs: dict[str, pd.DataFrame], model: str = "hybrid"):
        fusion_row = self._build_fusion_row(subsystem_outputs)

        result = predict_spi3(
            rows=[fusion_row],
            model=model,
        )

        return {
            "fusion_input": pd.DataFrame([fusion_row]),
            "fusion_result": result,
        }