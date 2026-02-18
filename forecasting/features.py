"""
Feature engineering for AQI forecasting.
=========================================
Reusable module shared by the training pipeline and inference.
All operations are strictly causal (no future leakage).
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

# ── Column Mapping ──────────────────────────────────────────────────
RAW_COL_MAP = {
    "PM2.5 (µg/m³)":  "pm25",
    "PM10 (µg/m³)":   "pm10",
    "NO (µg/m³)":     "no",
    "NO2 (µg/m³)":    "no2",
    "NOx (ppb)":      "nox",
    "NH3 (µg/m³)":    "nh3",
    "SO2 (µg/m³)":    "so2",
    "CO (mg/m³)":     "co",
    "Ozone (µg/m³)":  "o3",
    "AT (°C)":        "temperature",
    "RH (%)":         "humidity",
    "WS (m/s)":       "wind_speed",
    "WD (deg)":       "wind_dir",
    "RF (mm)":        "rainfall",
    "SR (W/mt2)":     "solar_rad",
    "BP (mmHg)":      "pressure",
    "AQI":            "aqi",
}

# Core pollutants used as features
POLLUTANTS = ["pm25", "pm10", "no2", "so2", "nh3", "co", "o3"]

# Meteorological features
METEO_FEATURES = ["temperature", "humidity", "wind_speed", "wind_dir",
                  "rainfall", "solar_rad", "pressure"]

# Lag horizons (hours)
LAG_HOURS = [1, 2, 3, 6, 12, 24, 168]

# Rolling window sizes (hours)
ROLLING_WINDOWS = [3, 6, 24]


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load raw CSV, rename columns, set datetime index, sort."""
    df = pd.read_csv(csv_path)
    # Normalise column names: strip whitespace, lowercase the Timestamp column
    df.columns = df.columns.str.strip()
    if "Timestamp" in df.columns:
        df.rename(columns={"Timestamp": "timestamp"}, inplace=True)
    df.rename(columns=RAW_COL_MAP, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Keep only renamed columns that exist
    keep = [c for c in list(RAW_COL_MAP.values()) if c in df.columns and c != "timestamp"]
    df = df[[c for c in keep if c in df.columns]]

    # Drop duplicate timestamps, keep last
    df = df[~df.index.duplicated(keep="last")]

    # Ensure hourly frequency via resampling (median for robustness)
    df = df.resample("1h").median()

    return df


def handle_missing(df: pd.DataFrame, max_gap: int = 6) -> pd.DataFrame:
    """
    Interpolate small gaps (≤ max_gap hours) linearly.
    Larger gaps stay NaN and rows are later dropped during feature creation.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="time", limit=max_gap, limit_direction="forward"
    )
    return df


def cap_outliers(df: pd.DataFrame,
                 col: str = "aqi",
                 lower_q: float = 0.001,
                 upper_q: float = 0.999) -> pd.DataFrame:
    """Winsorize extreme values to prevent outlier-driven distortion."""
    lo = df[col].quantile(lower_q)
    hi = df[col].quantile(upper_q)
    df[col] = df[col].clip(lo, hi)
    return df


def add_lag_features(df: pd.DataFrame, target: str = "aqi") -> pd.DataFrame:
    """Create lag features for AQI. Strictly causal."""
    for lag in LAG_HOURS:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    return df


def add_pollutant_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag-1 and lag-3 for each key pollutant."""
    for p in POLLUTANTS:
        if p in df.columns:
            df[f"{p}_lag1"] = df[p].shift(1)
            df[f"{p}_lag3"] = df[p].shift(3)
    return df


def add_rolling_features(df: pd.DataFrame, target: str = "aqi") -> pd.DataFrame:
    """Rolling mean and std — window ends at current row (no leakage)."""
    for w in ROLLING_WINDOWS:
        df[f"{target}_rmean{w}"] = (
            df[target].shift(1).rolling(window=w, min_periods=1).mean()
        )
        df[f"{target}_rstd{w}"] = (
            df[target].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
        )
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar and cyclical time features."""
    idx = df.index
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_meteo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Wind vector decomposition and interaction features."""
    if "wind_speed" in df.columns and "wind_dir" in df.columns:
        wd_rad = np.deg2rad(df["wind_dir"])
        df["wind_u"] = df["wind_speed"] * np.sin(wd_rad)
        df["wind_v"] = df["wind_speed"] * np.cos(wd_rad)

    # Temperature-humidity interaction (proxy for atmospheric stability)
    if "temperature" in df.columns and "humidity" in df.columns:
        df["temp_hum_interaction"] = df["temperature"] * df["humidity"] / 100.0

    # Diurnal range proxy: solar radiation * temperature
    if "solar_rad" in df.columns and "temperature" in df.columns:
        df["solar_temp"] = df["solar_rad"] * df["temperature"] / 1000.0
    return df


def add_rate_of_change(df: pd.DataFrame, target: str = "aqi") -> pd.DataFrame:
    """AQI rate of change (delta) features — recent momentum."""
    df[f"{target}_delta1"] = df[target].diff(1)
    df[f"{target}_delta3"] = df[target].diff(3)
    df[f"{target}_delta6"] = df[target].diff(6)
    # Acceleration (second-order delta)
    df[f"{target}_accel"] = df[f"{target}_delta1"].diff(1)
    return df


def build_features(df: pd.DataFrame, target: str = "aqi",
                   include_meteo: bool = True) -> pd.DataFrame:
    """Full feature engineering pipeline. Order matters for causality."""
    df = add_lag_features(df, target)
    df = add_pollutant_lags(df)
    df = add_rolling_features(df, target)
    df = add_rate_of_change(df, target)
    df = add_temporal_features(df)
    if include_meteo:
        df = add_meteo_features(df)
    return df


def get_feature_columns(df: pd.DataFrame, target: str = "aqi") -> List[str]:
    """Return the list of feature columns (everything except target and raw pollutants)."""
    exclude = {target, "Timestamp", "AQI_Category", "Prominent_Pollutant",
               "aqi_category", "prominent_pollutant"}
    # Include: lags, rolling, temporal, meteo, pollutants (including their lags)
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in (np.float64, np.int64, float, int)]
    return sorted(feat_cols)


def time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict time-based split: train 2020-2022, val 2023, test 2024."""
    train = df.loc[:"2022-12-31"]
    val = df.loc["2023-01-01":"2023-12-31"]
    test = df.loc["2024-01-01":]
    return train, val, test
