from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    lat: float = Field(..., description="Latitude in decimal degrees")
    lon: float = Field(..., description="Longitude in decimal degrees")
    yyyymm: str = Field(..., description="Target year-month in YYYYMM format")
    scenario: str = Field(..., description="Scenario name, e.g. historical, ssp245, ssp370, ssp585")