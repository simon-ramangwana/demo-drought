import pandas as pd


class MasterInputReader:
    def __init__(self, parquet_path: str, pixels_per_block: int):
        self.parquet_path = parquet_path
        self.pixels_per_block = pixels_per_block
        self._df = None

    def _load(self):
        if self._df is None:
            self._df = pd.read_parquet(self.parquet_path)

    def get_pixel_row(self, scenario: str, yyyymm: str, pixel_id: int) -> pd.Series:
        self._load()

        block = self._df[
            (self._df["scenario"] == scenario) &
            (self._df["yyyymm"] == yyyymm)
        ].reset_index(drop=True)

        if block.empty:
            raise ValueError(f"No data found for scenario={scenario}, yyyymm={yyyymm}")

        if not 0 <= pixel_id < self.pixels_per_block:
            raise ValueError(f"pixel_id must be between 0 and {self.pixels_per_block - 1}")

        return block.iloc[pixel_id]

    def _get_single_scenario_timeseries(self, scenario: str, pixel_id: int) -> pd.DataFrame:
        df = self._df[self._df["scenario"] == scenario].copy()
        if df.empty:
            raise ValueError(f"No data found for scenario={scenario}")

        rows = []
        for yyyymm, block in df.groupby("yyyymm", sort=True):
            block = block.reset_index(drop=True)
            if len(block) != self.pixels_per_block:
                raise ValueError(f"Unexpected block size for {scenario}-{yyyymm}: {len(block)}")
            rows.append(block.iloc[pixel_id])

        return pd.DataFrame(rows).reset_index(drop=True)

    def get_pixel_timeseries(self, scenario: str, pixel_id: int) -> pd.DataFrame:
        self._load()

        if not 0 <= pixel_id < self.pixels_per_block:
            raise ValueError(f"pixel_id must be between 0 and {self.pixels_per_block - 1}")

        # Historical request: just return historical
        if scenario == "historical":
            ts = self._get_single_scenario_timeseries("historical", pixel_id)
            return ts.sort_values("yyyymm").reset_index(drop=True)

        # Future request: stitch historical + future scenario
        historical_ts = self._get_single_scenario_timeseries("historical", pixel_id)
        future_ts = self._get_single_scenario_timeseries(scenario, pixel_id)

        historical_ts = historical_ts[historical_ts["yyyymm"] <= "202512"].copy()
        future_ts = future_ts[future_ts["yyyymm"] >= "202601"].copy()

        stitched = pd.concat([historical_ts, future_ts], axis=0, ignore_index=True)
        stitched = stitched.sort_values("yyyymm").drop_duplicates(subset=["yyyymm"], keep="last")
        stitched = stitched.reset_index(drop=True)

        return stitched