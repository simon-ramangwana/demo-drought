from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon

from app.services.feature_service import FeatureService
from app.services.fusion_service import FusionService
from app.services.subsystem_service import SubsystemService


@dataclass
class PolygonInferenceResult:
    cell_results: list[dict[str, Any]]
    summary: dict[str, Any]


class PolygonInferenceService:
    def __init__(self, master_df: pd.DataFrame):
        self.master_df = master_df.copy()
        self.master_df["yyyymm"] = self.master_df["yyyymm"].astype(str)

        self.feature_service = FeatureService()
        self.subsystem_service = SubsystemService()
        self.fusion_service = FusionService()

    def _geometry_mask(self, geometry: Polygon | MultiPolygon) -> pd.Series:
        points = self.master_df[["pixel_id", "lon", "lat"]].drop_duplicates().copy()
        points["_inside"] = points.apply(
            lambda r: geometry.contains(Point(r["lon"], r["lat"]))
            or geometry.touches(Point(r["lon"], r["lat"])),
            axis=1,
        )
        inside_pixel_ids = points.loc[points["_inside"], "pixel_id"].unique()
        return self.master_df["pixel_id"].isin(inside_pixel_ids)

    def _prepare_timeseries(
        self,
        pixel_id: int,
        scenario: str,
        run_yyyymm: int,
    ) -> pd.DataFrame:
        run_yyyymm = str(run_yyyymm)
        base = self.master_df[self.master_df["pixel_id"] == pixel_id].copy()

        if scenario == "historical":
            ts = base[base["scenario"] == "historical"].copy()
            ts = ts[ts["yyyymm"] <= run_yyyymm]
        else:
            hist = base[
                (base["scenario"] == "historical")
                & (base["yyyymm"] <= "202512")
            ].copy()

            fut = base[
                (base["scenario"] == scenario)
                & (base["yyyymm"] >= "202601")
                & (base["yyyymm"] <= run_yyyymm)
            ].copy()

            ts = pd.concat([hist, fut], ignore_index=True)

        ts = (
            ts.sort_values(["yyyymm"])
            .drop_duplicates(subset=["pixel_id", "yyyymm"], keep="last")
            .reset_index(drop=True)
        )
        return ts

    def _infer_one_pixel(
        self,
        pixel_id: int,
        scenario: str,
        run_yyyymm: int,
        model: str,
    ) -> dict[str, Any] | None:
        ts = self._prepare_timeseries(
            pixel_id=pixel_id,
            scenario=scenario,
            run_yyyymm=run_yyyymm,
        )

        if ts.empty:
            return None

        prepared = self.feature_service.prepare_subsystem_inputs(
            ts,
            run_yyyymm=run_yyyymm,
        )
        subsystem_outputs = self.subsystem_service.run_subsystems(prepared)
        fusion_result = self.fusion_service.run(subsystem_outputs, model=model)

        if hasattr(fusion_result, "to_dict"):
            fusion_result = fusion_result.to_dict()

        latest = ts.iloc[-1]

        return {
            "pixel_id": int(pixel_id),
            "lon": float(latest["lon"]),
            "lat": float(latest["lat"]),
            "result": fusion_result,
        }

    def _extract_class_and_conf(self, result: Any) -> tuple[str | None, float | None]:
        if result is None:
            return None, None

        if isinstance(result, dict) and "fusion_result" in result:
            result = result["fusion_result"]

        if hasattr(result, "drought_class_spi3_pred") and hasattr(result, "confidence"):
            pred = result.drought_class_spi3_pred
            conf = result.confidence

            pred_val = int(pred[0]) if len(pred) else None
            conf_val = float(conf[0]) if len(conf) else None

            class_map = {
                0: "normal",
                1: "moderate",
                2: "severe",
                3: "extreme",
            }
            return class_map.get(pred_val, str(pred_val)), conf_val

        return None, None

    def _summarize(self, cell_results: list[dict[str, Any]]) -> dict[str, Any]:
        if not cell_results:
            return {
                "cells_inferred": 0,
                "mean_confidence": None,
                "dominant_class": None,
            }

        extracted = [self._extract_class_and_conf(r.get("result")) for r in cell_results]
        classes = [cls for cls, _ in extracted if cls is not None]
        confidences = [conf for _, conf in extracted if conf is not None]

        dominant_class = max(set(classes), key=classes.count) if classes else None
        mean_confidence = sum(confidences) / len(confidences) if confidences else None

        return {
            "cells_inferred": len(cell_results),
            "mean_confidence": mean_confidence,
            "dominant_class": dominant_class,
        }

    def infer_polygon(
        self,
        geometry: Polygon | MultiPolygon,
        scenario: str,
        yyyymm: int,
        model: str = "hybrid",
    ) -> PolygonInferenceResult:
        mask = self._geometry_mask(geometry)
        pixel_ids = sorted(self.master_df.loc[mask, "pixel_id"].unique().tolist())

        cell_results: list[dict[str, Any]] = []
        for pixel_id in pixel_ids:
            out = self._infer_one_pixel(
                pixel_id=pixel_id,
                scenario=scenario,
                run_yyyymm=yyyymm,
                model=model,
            )
            if out is not None:
                cell_results.append(out)

        summary = self._summarize(cell_results)

        return PolygonInferenceResult(
            cell_results=cell_results,
            summary=summary,
        )