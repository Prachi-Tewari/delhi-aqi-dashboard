"""Dynamic, context-aware insights engine.

Analyzes current AQI data, historical trends, forecasts, and station
patterns to generate intelligent, actionable insights in real time.

Insight categories:
  • trend_alert    — significant AQI change in recent hours
  • health_context — contextual health advice for current conditions
  • comparison     — current vs historical / WHO / other benchmarks
  • anomaly        — unusual readings at specific stations
  • forecast_note  — highlights from the short-term forecast
  • diurnal        — time-of-day patterns (rush hour, overnight, etc.)
  • station_insight— notable differences between stations
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Insight:
    """A single generated insight."""
    category: str       # trend_alert, health_context, comparison, etc.
    severity: str       # info, warning, critical
    emoji: str
    title: str
    body: str
    priority: int = 0   # higher = show first


# ═══════════════════════════════════════════════════════════════════
# WHO + INDIA NAQI REFERENCE DATA
# ═══════════════════════════════════════════════════════════════════

WHO_24H = {"pm25": 15, "pm10": 45, "no2": 25, "so2": 40, "o3": 100, "co": 4000}
INDIA_24H = {"pm25": 60, "pm10": 100, "no2": 80, "so2": 80, "o3": 100, "co": 4000}

POLLUTANT_LABELS = {
    "pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂",
    "so2": "SO₂", "co": "CO", "o3": "O₃",
}


# ═══════════════════════════════════════════════════════════════════
# TREND ALERTS
# ═══════════════════════════════════════════════════════════════════

def _trend_insights(
    df_hourly: pd.DataFrame,
    current_aqi: int,
    aqi_category: str,
) -> list[Insight]:
    """Detect significant AQI/pollutant changes in recent hours."""
    insights = []

    if df_hourly is None or df_hourly.empty:
        return insights

    for param in ["pm25", "pm10", "no2", "o3"]:
        sub = df_hourly[df_hourly["parameter"] == param].sort_values("date_utc")
        if len(sub) < 4:
            continue

        recent_3h = sub.tail(3)["value"].mean()
        earlier_3h = sub.iloc[-6:-3]["value"].mean() if len(sub) >= 6 else sub.head(3)["value"].mean()
        name = POLLUTANT_LABELS.get(param, param)

        if earlier_3h > 0:
            pct_change = ((recent_3h - earlier_3h) / earlier_3h) * 100
        else:
            continue

        if pct_change > 25:
            insights.append(Insight(
                category="trend_alert",
                severity="warning",
                emoji="",
                title=f"{name} Spike Detected",
                body=f"{name} has risen **{pct_change:.0f}%** in the last 3 hours "
                     f"(from {earlier_3h:.0f} to {recent_3h:.0f} µg/m³). "
                     f"{'This may be due to evening traffic and boundary layer collapse.' if param in ('pm25', 'pm10') else 'Monitor conditions closely.'}",
                priority=80 + int(pct_change),
            ))
        elif pct_change < -25:
            insights.append(Insight(
                category="trend_alert",
                severity="info",
                emoji="",
                title=f"{name} Improving",
                body=f"{name} has dropped **{abs(pct_change):.0f}%** in the last 3 hours "
                     f"(from {earlier_3h:.0f} to {recent_3h:.0f} µg/m³). "
                     f"Air quality is getting better.",
                priority=40,
            ))

    return insights


# ═══════════════════════════════════════════════════════════════════
# DIURNAL PATTERN INSIGHTS
# ═══════════════════════════════════════════════════════════════════

def _diurnal_insights(df_hourly: pd.DataFrame) -> list[Insight]:
    """Insights based on time of day and typical pollution patterns."""
    insights = []
    if df_hourly is None or df_hourly.empty:
        return insights

    now_utc = datetime.now(timezone.utc)
    ist_hour = (now_utc.hour + 5) % 24  # rough IST (UTC+5:30)

    if 7 <= ist_hour <= 10:
        insights.append(Insight(
            category="diurnal",
            severity="info",
            emoji="",
            title="Morning Rush Hour",
            body="Traffic emissions typically peak between 8-10 AM. "
                 "PM2.5 and NO₂ may rise. Avoid outdoor exercise if AQI is elevated.",
            priority=50,
        ))
    elif 17 <= ist_hour <= 21:
        insights.append(Insight(
            category="diurnal",
            severity="warning",
            emoji="",
            title="Evening Pollution Peak",
            body="Evening hours see the highest pollution in Delhi due to "
                 "traffic, cooking emissions, and atmospheric boundary layer collapse "
                 "trapping pollutants near the surface.",
            priority=60,
        ))
    elif 23 <= ist_hour or ist_hour <= 4:
        insights.append(Insight(
            category="diurnal",
            severity="info",
            emoji="",
            title="Nighttime Inversion",
            body="Night-time temperature inversions can trap pollutants at ground level. "
                 "PM2.5 often stays elevated until morning winds begin.",
            priority=35,
        ))
    elif 11 <= ist_hour <= 15:
        insights.append(Insight(
            category="diurnal",
            severity="info",
            emoji="",
            title="Midday Mixing",
            body="Solar heating improves atmospheric mixing, which typically disperses "
                 "pollutants. This is usually the best window for outdoor activity.",
            priority=30,
        ))

    return insights


# ═══════════════════════════════════════════════════════════════════
# WHO / STANDARD COMPARISON INSIGHTS
# ═══════════════════════════════════════════════════════════════════

def _comparison_insights(
    pollutant_vals: dict,
    aqi_val: int,
) -> list[Insight]:
    """Compare current values against WHO and India standards."""
    insights = []

    for param, (val, unit) in pollutant_vals.items():
        name = POLLUTANT_LABELS.get(param, param)
        who_limit = WHO_24H.get(param)
        india_limit = INDIA_24H.get(param)

        if who_limit and val > who_limit:
            ratio = val / who_limit
            if ratio > 10:
                sev = "critical"
                emoji = ""
                prio = 95
            elif ratio > 5:
                sev = "warning"
                emoji = ""
                prio = 75
            else:
                sev = "info"
                emoji = ""
                prio = 50

            insights.append(Insight(
                category="comparison",
                severity=sev,
                emoji=emoji,
                title=f"{name}: {ratio:.1f}× WHO Limit",
                body=f"Current {name} is **{val:.0f} {unit}**, which is "
                     f"**{ratio:.1f}×** the WHO 24-hour guideline of {who_limit} {unit}. "
                     f"{'This level poses serious health risks.' if ratio > 5 else 'Sensitive groups should take precautions.'}",
                priority=prio,
            ))

    # Overall AQI context
    if aqi_val > 300:
        insights.append(Insight(
            category="comparison",
            severity="critical",
            emoji="",
            title="AQI Emergency Level",
            body=f"AQI **{aqi_val}** is in the emergency range. "
                 "This is equivalent to smoking **"
                 f"{round(pollutant_vals.get('pm25', (0,''))[0] / 22, 1)}** cigarettes per day. "
                 "Avoid all outdoor exposure.",
            priority=100,
        ))

    return insights


# ═══════════════════════════════════════════════════════════════════
# FORECAST-BASED INSIGHTS
# ═══════════════════════════════════════════════════════════════════

def _forecast_insights(aqi_forecast) -> list[Insight]:
    """Generate insights from the forecasting module's results."""
    insights = []

    if aqi_forecast is None or not aqi_forecast.forecasted_aqi:
        return insights

    current = aqi_forecast.current_aqi
    predicted = aqi_forecast.forecasted_aqi

    # Check for category change
    if len(predicted) >= 1:
        next_cat = predicted[0].get("category", "")
        if next_cat and predicted[0]["aqi"] > current * 1.15:
            insights.append(Insight(
                category="forecast_note",
                severity="warning",
                emoji="",
                title="AQI Expected to Worsen",
                body=f"AQI is predicted to rise from **{current}** to "
                     f"**{predicted[0]['aqi']}** ({next_cat}) in the next hour. "
                     f"Consider moving outdoor activities indoors.",
                priority=85,
            ))
        elif predicted[-1]["aqi"] < current * 0.85:
            insights.append(Insight(
                category="forecast_note",
                severity="info",
                emoji="",
                title="AQI Expected to Improve",
                body=f"AQI is predicted to drop from **{current}** to "
                     f"**{predicted[-1]['aqi']}** in the next {len(predicted)} hours. "
                     f"Outdoor conditions should get better.",
                priority=55,
            ))

    # Highlight volatile pollutants
    for param, pf in aqi_forecast.pollutant_forecasts.items():
        if pf.trend == "rising" and abs(pf.trend_pct) > 8:
            name = POLLUTANT_LABELS.get(param, param)
            insights.append(Insight(
                category="forecast_note",
                severity="warning",
                emoji="",
                title=f"{name} Rising Rapidly",
                body=f"{name} is increasing at **{pf.trend_pct:+.1f}%/hr**. "
                     f"Current: {pf.current_value} → "
                     f"Predicted: {pf.points[-1].value if pf.points else '?'} "
                     f"in {len(pf.points)}h.",
                priority=70,
            ))

    return insights


# ═══════════════════════════════════════════════════════════════════
# STATION-LEVEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════

def _station_insights(df_latest: pd.DataFrame) -> list[Insight]:
    """Find notable differences between monitoring stations."""
    insights = []

    if df_latest is None or df_latest.empty or "location" not in df_latest.columns:
        return insights

    for param in ["pm25", "pm10"]:
        sub = df_latest[df_latest["parameter"] == param]
        if sub.empty or sub["location"].nunique() < 2:
            continue

        station_means = sub.groupby("location")["value"].mean()
        if len(station_means) < 2:
            continue

        worst = station_means.idxmax()
        best = station_means.idxmin()
        worst_val = station_means[worst]
        best_val = station_means[best]
        name = POLLUTANT_LABELS.get(param, param)

        if best_val > 0 and worst_val / best_val > 2:
            insights.append(Insight(
                category="station_insight",
                severity="info",
                emoji="",
                title=f"{name} Varies Across Stations",
                body=f"**{worst}** reads {worst_val:.0f} µg/m³ while "
                     f"**{best}** reads {best_val:.0f} µg/m³ — "
                     f"a **{worst_val / best_val:.1f}×** difference. "
                     f"Hyperlocal factors (traffic, construction) may be at play.",
                priority=45,
            ))

    return insights


# ═══════════════════════════════════════════════════════════════════
# HEALTH CONTEXT INSIGHTS
# ═══════════════════════════════════════════════════════════════════

def _health_insights(aqi_val: int, aqi_category: str) -> list[Insight]:
    """Generate population-specific health advice."""
    insights = []

    if aqi_val > 200:
        insights.append(Insight(
            category="health_context",
            severity="critical" if aqi_val > 300 else "warning",
            emoji="",
            title="Respiratory Risk Elevated",
            body="People with asthma, COPD, or other respiratory conditions "
                 "should **stay indoors** with air purifiers. "
                 "Even healthy individuals may experience throat irritation and coughing.",
            priority=90 if aqi_val > 300 else 70,
        ))

    if aqi_val > 150:
        insights.append(Insight(
            category="health_context",
            severity="warning",
            emoji="",
            title="Outdoor Exercise Advisory",
            body="Avoid prolonged outdoor exercise. Heavy breathing during "
                 "exercise increases particulate intake by 5-10×. "
                 "If exercising, choose early afternoon when mixing is best.",
            priority=65,
        ))

    if aqi_val >= 100 and aqi_val <= 200:
        insights.append(Insight(
            category="health_context",
            severity="info",
            emoji="",
            title="Sensitive Groups Take Care",
            body="Children, elderly, and those with pre-existing conditions "
                 "should limit prolonged outdoor exposure. "
                 "Use N95 masks if going outdoors.",
            priority=50,
        ))

    return insights


# ═══════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════

def generate_insights(
    aqi_val: int,
    aqi_category: str,
    pollutant_vals: dict,
    df_latest: Optional[pd.DataFrame] = None,
    df_hourly: Optional[pd.DataFrame] = None,
    aqi_forecast=None,
    max_insights: int = 6,
) -> list[Insight]:
    """Generate a ranked list of context-aware insights.

    Args:
        aqi_val:        Current overall AQI.
        aqi_category:   Current AQI category string.
        pollutant_vals: Dict of {param: (value, unit)}.
        df_latest:      Per-station latest readings DataFrame.
        df_hourly:      Recent hourly data DataFrame.
        aqi_forecast:   AQIForecast object from forecasting module.
        max_insights:   Maximum number of insights to return.

    Returns:
        Sorted list of Insight objects (highest priority first).
    """
    all_insights: list[Insight] = []

    # Collect insights from all generators
    all_insights.extend(_trend_insights(df_hourly, aqi_val, aqi_category))
    all_insights.extend(_diurnal_insights(df_hourly))
    all_insights.extend(_comparison_insights(pollutant_vals, aqi_val))
    all_insights.extend(_forecast_insights(aqi_forecast))
    all_insights.extend(_station_insights(df_latest))
    all_insights.extend(_health_insights(aqi_val, aqi_category))

    # Sort by priority (descending) and trim
    all_insights.sort(key=lambda i: i.priority, reverse=True)
    return all_insights[:max_insights]


def format_insights_markdown(insights: list[Insight]) -> str:
    """Format insights as a Markdown string for display."""
    if not insights:
        return "*No notable insights at this time.*"

    parts = []
    for ins in insights:
        severity_badge = {
            "critical": "[CRITICAL]",
            "warning": "[WARNING]",
            "info": "[INFO]",
        }.get(ins.severity, "")

        parts.append(
            f"**{ins.title}** {severity_badge}\n\n"
            f"{ins.body}"
        )

    return "\n\n---\n\n".join(parts)
