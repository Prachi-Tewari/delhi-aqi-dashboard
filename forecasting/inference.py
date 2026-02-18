"""
AQI Forecasting — Inference Module
====================================
Production-ready inference for the Streamlit dashboard.
Loads saved model artifacts and generates 6-hour recursive forecasts
from live OpenAQ data.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from forecasting.features import (
    build_features, get_feature_columns,
    POLLUTANTS, METEO_FEATURES, LAG_HOURS, ROLLING_WINDOWS,
)

MODEL_DIR = Path(__file__).resolve().parent / "models"


@dataclass
class ForecastPoint:
    hour_offset: int
    aqi: float
    category: str
    lower: float  # uncertainty band
    upper: float


@dataclass
class AQIForecast:
    """Container for 6-hour forecast results."""
    forecasted_aqi: List[Dict]
    overall_trend: str        # "worsening" | "improving" | "stable"
    confidence_level: str     # "high" | "medium" | "low"
    model_name: str
    summary: str
    current_aqi: float = 0.0


def _classify_aqi(val: float) -> str:
    """India NAQI category classification."""
    if val <= 50:
        return "Good"
    elif val <= 100:
        return "Satisfactory"
    elif val <= 200:
        return "Moderate"
    elif val <= 300:
        return "Poor"
    elif val <= 400:
        return "Very Poor"
    else:
        return "Severe"


def _load_artifacts():
    """Load model and metadata. Cached at module level."""
    model = joblib.load(MODEL_DIR / "best_model.pkl")
    with open(MODEL_DIR / "feature_cols.json") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "training_results.json") as f:
        meta = json.load(f)
    return model, feat_cols, meta


# Module-level cache
_cached_model = None
_cached_feat_cols = None
_cached_meta = None


def _get_model():
    global _cached_model, _cached_feat_cols, _cached_meta
    if _cached_model is None:
        _cached_model, _cached_feat_cols, _cached_meta = _load_artifacts()
    return _cached_model, _cached_feat_cols, _cached_meta


def prepare_live_data(df_hourly: pd.DataFrame,
                      current_aqi: float = None,
                      pollutant_vals: dict = None) -> pd.DataFrame:
    """
    Transform live OpenAQ hourly data into the feature format
    expected by the trained model.

    The OpenAQ ``get_hourly_data`` function returns **long-format** data:
        parameter | value | unit | date_utc | location
    This function pivots it to wide format, computes per-hour AQI,
    and builds all model features.

    Parameters
    ----------
    df_hourly : pd.DataFrame
        Hourly data from OpenAQ (get_hourly_data output).
        Long format with columns: parameter, value, unit, date_utc, location.
    current_aqi : float
        The current computed AQI value.
    pollutant_vals : dict
        Current pollutant concentrations {param: value, ...}.

    Returns
    -------
    pd.DataFrame with model features, indexed by timestamp.
    """
    df = df_hourly.copy()

    # ── Detect format: long (OpenAQ) vs wide ────────────────────────
    is_long_format = ("parameter" in df.columns and "value" in df.columns)

    if is_long_format:
        df = _pivot_long_to_wide(df, current_aqi, pollutant_vals)
    else:
        df = _normalise_wide(df, current_aqi)

    if df is None or len(df) < 3:
        return pd.DataFrame()

    # Interpolate small gaps
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].interpolate(method="time", limit=6,
                                          limit_direction="forward")

    # Build all features
    df = build_features(df, target="aqi", include_meteo=True)

    return df


# ── OpenAQ parameter name normalisation ─────────────────────────────
_PARAM_MAP = {
    "pm25": "pm25", "pm2.5": "pm25",
    "pm10": "pm10",
    "no2": "no2", "nitrogen dioxide": "no2",
    "so2": "so2", "sulfur dioxide": "so2", "sulphur dioxide": "so2",
    "co": "co", "carbon monoxide": "co",
    "o3": "o3", "ozone": "o3",
    "nh3": "nh3", "ammonia": "nh3",
    "no": "no", "nitric oxide": "no",
    "nox": "nox",
    "temperature": "temperature", "at": "temperature",
    "humidity": "humidity", "rh": "humidity",
    "wind speed": "wind_speed", "ws": "wind_speed", "wind_speed": "wind_speed",
    "wind direction": "wind_dir", "wd": "wind_dir", "wind_dir": "wind_dir",
}


def _pivot_long_to_wide(df_long: pd.DataFrame,
                        current_aqi: float = None,
                        pollutant_vals: dict = None) -> pd.DataFrame:
    """Pivot OpenAQ long-format hourly data to wide-format with AQI."""
    df = df_long.copy()

    # Parse timestamps
    if "date_utc" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date_utc"], errors="coerce", utc=True)
    elif "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    else:
        return None

    df = df.dropna(subset=["timestamp"])
    if len(df) == 0:
        return None

    # Normalise parameter names
    df["param_clean"] = df["parameter"].str.lower().str.strip().map(_PARAM_MAP)
    df = df.dropna(subset=["param_clean"])

    # Floor to hour
    df["hour"] = df["timestamp"].dt.floor("h")

    # Average across stations per hour per parameter
    pivot = df.pivot_table(
        index="hour", columns="param_clean", values="value", aggfunc="mean"
    )
    pivot = pivot.sort_index()
    pivot.index.name = None
    pivot.index = pivot.index.tz_localize(None)  # remove tz for consistency

    # Compute per-hour AQI from available pollutants
    pivot["aqi"] = _compute_hourly_aqi(pivot)

    # If current_aqi provided, override the last hour
    if current_aqi is not None and len(pivot) > 0:
        pivot.loc[pivot.index[-1], "aqi"] = current_aqi

    # Back-fill AQI for hours where it couldn't be computed
    # using pollutant_vals for the latest hour if needed
    if pivot["aqi"].isna().all() and current_aqi is not None:
        pivot["aqi"] = current_aqi  # constant fill as last resort

    # Forward-fill AQI NaNs from computed values
    pivot["aqi"] = pivot["aqi"].ffill().bfill()

    return pivot


def _compute_hourly_aqi(df_wide: pd.DataFrame) -> pd.Series:
    """Compute India NAQI AQI for each hour from pollutant columns."""
    # Import the AQI computation function
    try:
        from visualization.plots import compute_sub_index, _convert_to_ugm3
    except ImportError:
        # Fallback: simple PM2.5-based AQI estimation
        if "pm25" in df_wide.columns:
            return df_wide["pm25"].apply(lambda x: x * 1.0 if not np.isnan(x) else np.nan)
        return pd.Series(np.nan, index=df_wide.index)

    aqi_series = []
    for idx, row in df_wide.iterrows():
        sub_indices = []
        # Map column names to (value, assumed_unit) pairs
        param_units = {
            "pm25": "µg/m³", "pm10": "µg/m³",
            "no2": "µg/m³", "so2": "µg/m³",
            "nh3": "µg/m³", "co": "mg/m³", "o3": "µg/m³",
        }
        for param, unit in param_units.items():
            if param in row.index and not np.isnan(row[param]):
                conc = _convert_to_ugm3(param, row[param], unit)
                si = compute_sub_index(param, conc)
                if si >= 0:
                    sub_indices.append(si)
        if sub_indices:
            aqi_series.append(max(sub_indices))
        else:
            aqi_series.append(np.nan)

    return pd.Series(aqi_series, index=df_wide.index)


def _normalise_wide(df: pd.DataFrame, current_aqi: float = None) -> pd.DataFrame:
    """Handle already-wide-format data (e.g., from CSV or pre-processed)."""
    # Ensure datetime index
    for col in ("date", "date_utc", "timestamp", "Timestamp"):
        if col in df.columns:
            df["timestamp"] = pd.to_datetime(df[col], errors="coerce")
            break

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Standardise column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        mapped = _PARAM_MAP.get(cl)
        if mapped:
            col_map[c] = mapped
    df.rename(columns=col_map, inplace=True)

    if "aqi" not in df.columns:
        df["aqi"] = np.nan

    if current_aqi is not None and len(df) > 0:
        df.loc[df.index[-1], "aqi"] = current_aqi

    return df


def forecast_next_6_hours(df_hourly: pd.DataFrame,
                          current_aqi: float = None,
                          pollutant_vals: dict = None,
                          horizon: int = 6) -> Optional[AQIForecast]:
    """
    Main inference entry point.
    Generates a recursive 6-hour AQI forecast from live data.

    Parameters
    ----------
    df_hourly : pd.DataFrame
        Recent hourly data (ideally 24+ hours, min 8 hours).
    current_aqi : float
        Current computed AQI value.
    pollutant_vals : dict
        Current pollutant concentrations {pm25: ..., pm10: ..., ...}
    horizon : int
        Number of hours to forecast (default 6).

    Returns
    -------
    AQIForecast or None if insufficient data.
    """
    try:
        model, feat_cols, meta = _get_model()
    except (FileNotFoundError, Exception):
        return None

    if df_hourly is None or df_hourly.empty:
        return None

    # Prepare features (handles both long and wide format)
    df = prepare_live_data(df_hourly, current_aqi, pollutant_vals)

    if df is None or df.empty or len(df) < 3:
        return None

    # Check we have at least some AQI values
    aqi_valid = df["aqi"].dropna().shape[0] if "aqi" in df.columns else 0
    if aqi_valid < 2:
        return None

    # Ensure all feature columns exist (fill missing with 0)
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0

    # Forward-fill remaining NaNs in features
    df[feat_cols] = df[feat_cols].ffill().fillna(0)

    # Recursive forecast
    predictions = _recursive_forecast_live(model, df, feat_cols, "aqi", horizon)

    if not predictions:
        return None

    # Compute uncertainty bands (empirical: ±15% for +1h, growing with horizon)
    forecast_points = []
    for i, pred in enumerate(predictions):
        h = i + 1
        uncertainty = 0.10 + 0.05 * h  # 15% at +1h, 40% at +6h
        lower = max(0, pred * (1 - uncertainty))
        upper = pred * (1 + uncertainty)
        forecast_points.append({
            "hour_offset": h,
            "aqi": round(pred, 1),
            "category": _classify_aqi(pred),
            "lower": round(lower, 1),
            "upper": round(upper, 1),
        })

    # Trend analysis
    base_aqi = current_aqi or df["aqi"].dropna().iloc[-1]
    final_pred = predictions[-1]
    pct_change = (final_pred - base_aqi) / max(base_aqi, 1) * 100

    if pct_change > 10:
        trend = "worsening"
    elif pct_change < -10:
        trend = "improving"
    else:
        trend = "stable"

    # Confidence assessment
    data_hours = df["aqi"].dropna().shape[0]
    if data_hours >= 24:
        confidence = "high"
    elif data_hours >= 12:
        confidence = "medium"
    else:
        confidence = "low"

    # Summary text
    pred_range = f"{min(predictions):.0f}-{max(predictions):.0f}"
    summary = (
        f"AQI is expected to {'increase' if trend == 'worsening' else 'decrease' if trend == 'improving' else 'remain stable'} "
        f"over the next {horizon} hours. "
        f"Predicted range: {pred_range} "
        f"({_classify_aqi(min(predictions))} to {_classify_aqi(max(predictions))}). "
        f"Confidence: {confidence}."
    )

    return AQIForecast(
        forecasted_aqi=forecast_points,
        overall_trend=trend,
        confidence_level=confidence,
        model_name=meta.get("best_model", "LightGBM"),
        summary=summary,
        current_aqi=base_aqi,
    )


def _recursive_forecast_live(model, df, feat_cols, target, horizon):
    """Recursive multi-step forecast for live inference."""
    predictions = []
    work = df.copy()
    last_idx = work.index[-1]

    for step in range(1, horizon + 1):
        X_input = work.loc[[last_idx], feat_cols]
        # Fill any NaN
        X_input = X_input.ffill(axis=1).fillna(0)

        pred = float(model.predict(X_input)[0])
        pred = max(pred, 0)
        predictions.append(pred)

        # Create next row
        next_ts = last_idx + pd.Timedelta(hours=1)
        new_row = work.loc[[last_idx]].copy()
        new_row.index = [next_ts]
        new_row[target] = pred

        # Update lags
        for lag in LAG_HOURS:
            col = f"{target}_lag{lag}"
            if col in new_row.columns:
                if lag == 1:
                    new_row[col] = work.loc[last_idx, target]
                else:
                    look_ts = next_ts - pd.Timedelta(hours=lag)
                    if look_ts in work.index:
                        new_row[col] = work.loc[look_ts, target]

        # Update rolling
        for w in ROLLING_WINDOWS:
            rmean_col = f"{target}_rmean{w}"
            rstd_col = f"{target}_rstd{w}"
            if rmean_col in new_row.columns:
                recent = work[target].iloc[-w:]
                new_row[rmean_col] = recent.mean()
                new_row[rstd_col] = recent.std() if len(recent) > 1 else 0

        # Update deltas
        if f"{target}_delta1" in new_row.columns:
            new_row[f"{target}_delta1"] = pred - work.loc[last_idx, target]

        # Update temporal
        new_row["hour"] = next_ts.hour
        new_row["hour_sin"] = np.sin(2 * np.pi * next_ts.hour / 24)
        new_row["hour_cos"] = np.cos(2 * np.pi * next_ts.hour / 24)
        new_row["dow"] = next_ts.dayofweek
        new_row["dow_sin"] = np.sin(2 * np.pi * next_ts.dayofweek / 7)
        new_row["dow_cos"] = np.cos(2 * np.pi * next_ts.dayofweek / 7)
        new_row["month"] = next_ts.month
        new_row["month_sin"] = np.sin(2 * np.pi * next_ts.month / 12)
        new_row["month_cos"] = np.cos(2 * np.pi * next_ts.month / 12)
        new_row["is_weekend"] = int(next_ts.dayofweek >= 5)

        work = pd.concat([work, new_row])
        last_idx = next_ts

    return predictions
