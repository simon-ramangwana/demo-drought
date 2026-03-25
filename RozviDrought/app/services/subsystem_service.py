import pandas as pd

import rozvidrought_subsystems as subs


class SubsystemService:
    def run_subsystems(self, prepared: dict[str, pd.DataFrame]) -> dict:
        atm_df = prepared["atmospheric"]
        soil_df = prepared["soil"]
        veg_df = prepared["vegetation"]
        hydro_df = prepared["hydrology"]

        atm_out = subs.predict_atmospheric_proba(atm_df)
        soil_out = subs.predict_soil_proba(soil_df)
        veg_out = subs.predict_vegetation_proba(veg_df)
        hydro_out = subs.predict_hydrology_proba_df(hydro_df)

        hydro_out = hydro_out.tail(1).reset_index(drop=True)

        return {
            "atmospheric": atm_out,
            "soil": soil_out,
            "vegetation": veg_out,
            "hydrology": hydro_out,
        }