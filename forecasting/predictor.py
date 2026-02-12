"""Short-term AQI forecasting (1–6 hours ahead).

Approach
--------
1. **Diurnal profile** — Delhi AQI follows a strong daily pattern: peaks
   at morning rush (~8–10 AM IST) and evening (~8–11 PM IST), dips in the
   afternoon.  We encode typical diurnal multipliers and blend them with
   the recent observed pattern.
2. **Damped trend** — short-term momentum (last 3–6 hours) is captured
   via exponential smoothing with heavy damping so forecasts don't
   run away.
3. **Physical clamps** — forecasts are bound by realistic pollutant
   limits and max hourly change.

Each forecast includes:
  • point estimate
  • confidence interval (upper / lower)
  • trend indicator (rising / falling / stable)
  • confidence level (high / medium / low)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ForecastPoint:
    """A single forecast data point."""
    hour_offset: int        # hours from now (1, 2, 3)
    timestamp: pd.Timestamp
    value: float            # predicted concentration
    lower: float            # lower confidence bound
    upper: float            # upper confidence bound
    confidence: float       # 0-1 confidence in prediction


@dataclass
class PollutantForecast:
    """Forecast for one pollutant."""
    parameter: str
    unit: str
    trend: str              # "rising", "falling", "stable"
    trend_pct: float        # % change per hour
    trend_emoji: str        # ^ v =
    points: List[ForecastPoint] = field(default_factory=list)
    current_value: float = 0.0
    method: str = ""        # "holt" or "linear"


@dataclass
class AQIForecast:
    """Complete forecast including AQI prediction."""
    current_aqi: int
    forecasted_aqi: List[dict] = field(default_factory=list)
    pollutant_forecasts: dict = field(default_factory=dict)
    overall_trend: str = "stable"
    overall_trend_emoji: str = "="
    confidence_level: str = "medium"
    summary: str = ""


# Physical upper bounds per pollutant (µg/m³ — extreme but possible for Delhi)
_MAX_BOUNDS = {
    "pm25": 999, "pm10": 999, "no2": 600, "so2": 800,
    "co": 50000, "o3": 600,
}

# Maximum hourly change ratio (forecast can't jump more than ±12% per hour)
_MAX_HOURLY_CHANGE = 0.12

# ── Typical Delhi diurnal pattern (IST hour → multiplier relative to daily mean)
# Derived from CPCB multi-year hourly averages for Delhi.
# Pattern: peaks at 8-10 AM and 8-11 PM; trough at 2-4 PM.
_DIURNAL_PROFILE = {
    0: 1.10, 1: 1.08, 2: 1.05, 3: 1.02, 4: 1.00, 5: 1.02,
    6: 1.08, 7: 1.15, 8: 1.22, 9: 1.20, 10: 1.12, 11: 1.02,
    12: 0.92, 13: 0.85, 14: 0.82, 15: 0.83, 16: 0.88, 17: 0.95,
    18: 1.02, 19: 1.10, 20: 1.18, 21: 1.20, 22: 1.18, 23: 1.14,
}


# ═══════════════════════════════════════════════════════════════════
# DIURNAL-AWARE FORECAST
# ═══════════════════════════════════════════════════════════════════

def _diurnal_forecast(
    values: np.ndarray,
    ist_hours: np.ndarray,
    horizon: int = 3,
    parameter: str = "",
    alpha: float = 0.35,
    phi: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forecast using diurnal profile + damped exponential smoothing.

    1. Estimate the "base level" (deseasonalised mean) from observed data.
    2. Compute short-term momentum from last few hours (damped).
    3. For each forecast hour, apply diurnal multiplier + momentum.
    4. Clamp to physical limits.

    Returns: (forecast_values, lower_bounds, upper_bounds)
    """
    n = len(values)
    if n < 2:
        v = values[-1] if n else 0
        return (np.full(horizon, v),
                np.full(horizon, v * 0.85),
                np.full(horizon, v * 1.15))

    # Step 1: Estimate base level by removing diurnal effect
    diurnal_mults = np.array([_DIURNAL_PROFILE.get(int(h) % 24, 1.0) for h in ist_hours])
    deseasonalised = values / np.maximum(diurnal_mults, 0.5)
    base_level = np.median(deseasonalised[-min(12, n):])  # median of recent 12h

    # Step 2: Short-term momentum (exponentially weighted recent trend)
    recent_n = min(6, n)
    recent = deseasonalised[-recent_n:]
    if len(recent) >= 3:
        # Exponential smoothing on deseasonalised values
        level = recent[0]
        trend = 0.0
        for t in range(1, len(recent)):
            new_level = alpha * recent[t] + (1 - alpha) * (level + phi * trend)
            trend = 0.3 * (new_level - level) + 0.7 * phi * trend
            level = new_level
        momentum_per_hour = trend
    else:
        momentum_per_hour = 0.0

    # Step 3: Build forecasts
    current_val = values[-1]
    last_ist_hour = int(ist_hours[-1]) % 24
    max_bound = _MAX_BOUNDS.get(parameter, current_val * 3)

    forecasts = np.zeros(horizon)
    for h in range(horizon):
        fut_hour = (last_ist_hour + h + 1) % 24
        diurnal_mult = _DIURNAL_PROFILE.get(fut_hour, 1.0)

        # Damped momentum: phi^h shrinks trend over time
        damped_momentum = momentum_per_hour * (phi ** (h + 1))

        # Predicted = base_level * diurnal_mult + accumulated momentum
        pred = (base_level + damped_momentum * (h + 1)) * diurnal_mult

        # Clamp: max change per hour
        ref = current_val if h == 0 else forecasts[h - 1]
        max_delta = ref * _MAX_HOURLY_CHANGE
        pred = np.clip(pred, ref - max_delta, ref + max_delta)
        pred = np.clip(pred, 0, max_bound)
        forecasts[h] = pred

    # Step 4: Confidence intervals from residual variability
    if n > 3:
        fitted = base_level * diurnal_mults
        residuals = values - fitted
        residual_std = np.std(residuals[-min(12, n):])
    else:
        residual_std = abs(current_val) * 0.08

    z = 1.65  # ~90% CI
    lower = np.array([f - z * residual_std * math.sqrt(h + 1) for h, f in enumerate(forecasts)])
    upper = np.array([f + z * residual_std * math.sqrt(h + 1) for h, f in enumerate(forecasts)])

    forecasts = np.maximum(forecasts, 0)
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, max_bound)

    return forecasts, lower, upper


# ═══════════════════════════════════════════════════════════════════
# LINEAR REGRESSION FALLBACK (very few data points)
# ═══════════════════════════════════════════════════════════════════

def _linear_forecast(
    values: np.ndarray,
    horizon: int = 3,
    parameter: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted linear regression — recent points weighted more."""
    n = len(values)
    if n < 2:
        v = values[-1] if n == 1 else 0
        return (np.full(horizon, v),
                np.full(horizon, v * 0.85),
                np.full(horizon, v * 1.15))

    x = np.arange(n)
    # Exponential weights — recent data matters more
    weights = np.exp(np.linspace(0, 2, n))
    weights /= weights.sum()

    # Weighted least squares
    w_mean_x = np.average(x, weights=weights)
    w_mean_y = np.average(values, weights=weights)
    cov = np.average((x - w_mean_x) * (values - w_mean_y), weights=weights)
    var = np.average((x - w_mean_x) ** 2, weights=weights)
    slope = cov / var if var > 0 else 0
    intercept = w_mean_y - slope * w_mean_x

    # Forecast
    future_x = np.arange(n, n + horizon)
    forecasts = intercept + slope * future_x

    # Clamp: max change per hour
    current = values[-1]
    max_bound = _MAX_BOUNDS.get(parameter, current * 3)
    for h in range(horizon):
        ref = current if h == 0 else forecasts[h - 1]
        max_delta = ref * _MAX_HOURLY_CHANGE
        forecasts[h] = np.clip(forecasts[h], ref - max_delta, ref + max_delta)
        forecasts[h] = np.clip(forecasts[h], 0, max_bound)

    # CI
    residuals = values - (intercept + slope * x)
    residual_std = np.std(residuals) if n > 2 else abs(values.mean()) * 0.1
    z = 1.65

    lower = np.array([f - z * residual_std * math.sqrt(h + 1.5) for h, f in enumerate(forecasts)])
    upper = np.array([f + z * residual_std * math.sqrt(h + 1.5) for h, f in enumerate(forecasts)])

    forecasts = np.maximum(forecasts, 0)
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, max_bound)

    return forecasts, lower, upper


# ═══════════════════════════════════════════════════════════════════
# TREND DETECTION
# ═══════════════════════════════════════════════════════════════════

def _detect_trend(values: np.ndarray) -> tuple[str, float, str]:
    """Detect trend direction from recent values.

    Returns (trend_label, pct_change_per_hour, emoji)
    """
    if len(values) < 2:
        return "stable", 0.0, "="

    # Use last 6 hours or all available
    recent = values[-min(6, len(values)):]
    first_half = recent[: len(recent) // 2].mean()
    second_half = recent[len(recent) // 2 :].mean()

    if first_half == 0:
        pct = 0.0
    else:
        pct = ((second_half - first_half) / first_half) * 100
        # Normalize to per-hour
        pct /= max(len(recent) // 2, 1)

    if pct > 3:
        return "rising", round(pct, 1), "^"
    elif pct < -3:
        return "falling", round(pct, 1), "v"
    else:
        return "stable", round(pct, 1), "="


# ═══════════════════════════════════════════════════════════════════
# CONFIDENCE ESTIMATION
# ═══════════════════════════════════════════════════════════════════

def _estimate_confidence(
    values: np.ndarray,
    forecast_vals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[float]:
    """Estimate confidence per forecast hour (0-1)."""
    n = len(values)
    # Base confidence from data availability
    data_conf = min(n / 12, 1.0)  # 12+ hours = full data confidence

    # Volatility penalty
    if n > 2:
        cv = np.std(values) / (np.mean(values) + 1e-6)
        vol_penalty = max(0, min(cv / 0.5, 0.4))  # up to 40% penalty
    else:
        vol_penalty = 0.3

    # CI width penalty (wider = less confident)
    confs = []
    for h in range(len(forecast_vals)):
        ci_width = upper[h] - lower[h]
        ci_ratio = ci_width / (forecast_vals[h] + 1e-6)
        ci_penalty = min(ci_ratio / 2, 0.3)

        # Distance penalty (further = less confident)
        dist_penalty = 0.05 * (h + 1)

        conf = max(0.1, data_conf - vol_penalty - ci_penalty - dist_penalty)
        confs.append(round(conf, 2))

    return confs


# ═══════════════════════════════════════════════════════════════════
# MAIN FORECAST FUNCTION
# ═══════════════════════════════════════════════════════════════════

def forecast_pollutant(
    df_hourly: pd.DataFrame,
    parameter: str,
    horizon: int = 3,
) -> Optional[PollutantForecast]:
    """Generate a forecast for a single pollutant.

    Args:
        df_hourly: DataFrame with columns [parameter, value, date_utc, unit]
                   Should contain hourly aggregated data.
        parameter: e.g. "pm25", "pm10", "no2"
        horizon:   hours to forecast ahead (default: 3)

    Returns:
        PollutantForecast or None if insufficient data.
    """
    sub = df_hourly[df_hourly["parameter"] == parameter].copy()
    if sub.empty:
        return None

    sub["date_utc"] = pd.to_datetime(sub["date_utc"], errors="coerce", utc=True)
    sub = sub.sort_values("date_utc").drop_duplicates("date_utc", keep="last")
    values = sub["value"].values.astype(float)
    timestamps = sub["date_utc"].values
    unit = sub["unit"].iloc[0] if "unit" in sub.columns else "µg/m³"

    if len(values) < 3:
        return None

    # Compute IST hours for diurnal profile
    ist_hours = np.array([
        (pd.Timestamp(t) + pd.Timedelta(hours=5, minutes=30)).hour
        for t in timestamps
    ], dtype=float)

    # Choose method based on data availability
    if len(values) >= 6:
        forecasts, lower, upper = _diurnal_forecast(
            values, ist_hours, horizon, parameter=parameter)
        method = "diurnal"
    else:
        forecasts, lower, upper = _linear_forecast(values, horizon, parameter=parameter)
        method = "linear"

    # Detect trend
    trend, trend_pct, trend_emoji = _detect_trend(values)

    # Estimate confidence
    confidences = _estimate_confidence(values, forecasts, lower, upper)

    # Build forecast points
    last_ts = pd.Timestamp(timestamps[-1])
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")

    points = []
    for h in range(horizon):
        ts = last_ts + pd.Timedelta(hours=h + 1)
        points.append(ForecastPoint(
            hour_offset=h + 1,
            timestamp=ts,
            value=round(float(forecasts[h]), 1),
            lower=round(float(lower[h]), 1),
            upper=round(float(upper[h]), 1),
            confidence=confidences[h],
        ))

    return PollutantForecast(
        parameter=parameter,
        unit=unit,
        trend=trend,
        trend_pct=trend_pct,
        trend_emoji=trend_emoji,
        points=points,
        current_value=round(float(values[-1]), 1),
        method=method,
    )


def forecast_aqi(
    df_hourly: pd.DataFrame,
    compute_aqi_fn,
    classify_fn,
    horizon: int = 3,
    override_current_aqi: int | None = None,
) -> AQIForecast:
    """Generate a complete AQI forecast from hourly data.

    Args:
        df_hourly:      Recent hourly data (all pollutants).
        compute_aqi_fn: Function to compute AQI from pollutant values dict.
        classify_fn:    Function to classify an AQI integer to category.
        horizon:        Hours to forecast (default 3).
        override_current_aqi: If set, use this as the display current AQI
            instead of computing from latest hourly values. Useful when the
            dashboard hero AQI is based on 24h median.

    Returns:
        AQIForecast with per-pollutant and overall AQI predictions.
    """
    parameters = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    pollutant_forecasts = {}

    for param in parameters:
        pf = forecast_pollutant(df_hourly, param, horizon)
        if pf is not None:
            pollutant_forecasts[param] = pf

    if not pollutant_forecasts:
        return AQIForecast(
            current_aqi=0,
            summary="Insufficient data for forecasting.",
            confidence_level="low",
        )

    # Compute current AQI from latest hourly values
    current_vals = {}
    for param, pf in pollutant_forecasts.items():
        current_vals[param] = (pf.current_value, pf.unit)
    current_aqi_result = compute_aqi_fn(current_vals)
    current_aqi = override_current_aqi if override_current_aqi is not None else current_aqi_result.get("aqi", 0)

    # Compute forecasted AQI for each hour
    forecasted_aqi = []
    for h in range(horizon):
        hour_vals = {}
        for param, pf in pollutant_forecasts.items():
            if h < len(pf.points):
                hour_vals[param] = (pf.points[h].value, pf.unit)
        if hour_vals:
            aqi_result = compute_aqi_fn(hour_vals)
            aqi_val = aqi_result.get("aqi", 0)
            cat = classify_fn(aqi_val)

            # Confidence: average of pollutant confidences for this hour
            confs = [pf.points[h].confidence for pf in pollutant_forecasts.values()
                     if h < len(pf.points)]
            avg_conf = sum(confs) / len(confs) if confs else 0.5

            # Compute AQI bounds
            lower_vals = {}
            upper_vals = {}
            for param, pf in pollutant_forecasts.items():
                if h < len(pf.points):
                    lower_vals[param] = (pf.points[h].lower, pf.unit)
                    upper_vals[param] = (pf.points[h].upper, pf.unit)

            lower_aqi = compute_aqi_fn(lower_vals).get("aqi", 0) if lower_vals else aqi_val
            upper_aqi = compute_aqi_fn(upper_vals).get("aqi", 0) if upper_vals else aqi_val

            # Fix: lower should actually be the min
            final_lower = min(lower_aqi, upper_aqi)
            final_upper = max(lower_aqi, upper_aqi)

            ts = list(pollutant_forecasts.values())[0].points[h].timestamp

            forecasted_aqi.append({
                "hour_offset": h + 1,
                "timestamp": ts,
                "aqi": aqi_val,
                "category": cat,
                "lower": final_lower,
                "upper": final_upper,
                "confidence": round(avg_conf, 2),
                "dominant": aqi_result.get("dominant", ""),
            })

    # Overall trend from AQI values
    if len(forecasted_aqi) >= 2:
        first_aqi = forecasted_aqi[0]["aqi"]
        last_aqi = forecasted_aqi[-1]["aqi"]
        if last_aqi > first_aqi * 1.05:
            overall_trend = "worsening"
            overall_emoji = "[UP]"
        elif last_aqi < first_aqi * 0.95:
            overall_trend = "improving"
            overall_emoji = "[DN]"
        else:
            overall_trend = "stable"
            overall_emoji = "[=]"
    else:
        overall_trend = "stable"
        overall_emoji = "[=]"

    # Confidence level
    avg_confidences = [f["confidence"] for f in forecasted_aqi]
    avg_conf = sum(avg_confidences) / len(avg_confidences) if avg_confidences else 0.5
    if avg_conf >= 0.65:
        conf_level = "high"
    elif avg_conf >= 0.4:
        conf_level = "medium"
    else:
        conf_level = "low"

    # Build summary
    summary_parts = []
    if forecasted_aqi:
        next_aqi = forecasted_aqi[0]
        summary_parts.append(
            f"AQI is expected to be **{next_aqi['aqi']}** "
            f"({next_aqi['category']}) in the next hour"
        )
        if len(forecasted_aqi) >= 3:
            h3 = forecasted_aqi[2]
            summary_parts.append(
                f"and **{h3['aqi']}** ({h3['category']}) in 3 hours"
            )
        summary_parts.append(f"\n\nOverall trend: **{overall_trend}** {overall_emoji}")

        # Add notable pollutant trends
        for param, pf in pollutant_forecasts.items():
            if pf.trend != "stable" and abs(pf.trend_pct) > 5:
                name = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2",
                        "so2": "SO2", "co": "CO", "o3": "O3"}.get(param, param)
                summary_parts.append(
                    f"\n{pf.trend_emoji} {name} is **{pf.trend}** "
                    f"({pf.trend_pct:+.1f}%/hr)"
                )

    return AQIForecast(
        current_aqi=current_aqi,
        forecasted_aqi=forecasted_aqi,
        pollutant_forecasts=pollutant_forecasts,
        overall_trend=overall_trend,
        overall_trend_emoji=overall_emoji,
        confidence_level=conf_level,
        summary=". ".join(summary_parts) if summary_parts else "Insufficient data.",
    )
