from pathlib import Path
import math

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.services.inference_service import InferenceService


BASE_DIR = Path(__file__).resolve().parents[2]

app = FastAPI(
    title="RozviDrought Inference API",
    version="1.0.0",
)

service = InferenceService()

templates = Jinja2Templates(
    directory=str(BASE_DIR / "app" / "frontend" / "templates")
)


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [json_safe(v) for v in value]

    if isinstance(value, pd.DataFrame):
        return json_safe(value.to_dict(orient="records"))

    if isinstance(value, pd.Series):
        return json_safe(value.to_dict())

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if hasattr(value, "__dict__") and not isinstance(value, type):
        try:
            return json_safe(vars(value))
        except Exception:
            return str(value)

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None

    return value


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "RozviDrought inference",
    }


@app.get("/infer/point")
def infer_point(
    lon: float = Query(..., description="Longitude"),
    lat: float = Query(..., description="Latitude"),
    scenario: str = Query(..., description="historical, ssp245, ssp370, ssp585"),
    yyyymm: int = Query(..., description="YYYYMM"),
):
    result = service.infer_point(
        lon=lon,
        lat=lat,
        scenario=scenario,
        yyyymm=yyyymm,
    )

    return JSONResponse(content=json_safe(result))