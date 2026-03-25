# app/services/feature_service.py
import pandas as pd
import rozvidrought_inputs as inputs


class FeatureService:
    def prepare_subsystem_inputs(self, timeseries: pd.DataFrame, run_yyyymm: int):
        prepared = inputs.prepare_from_raw(
            atmospheric_df=timeseries.copy(),
            soil_df=timeseries.copy(),
            vegetation_df=timeseries.copy(),
            hydrology_df=timeseries.copy(),
            run_yyyymm=run_yyyymm,
        )
        return prepared