"""Tool-calling layer for live AQI API retrieval.

When the query classifier detects a 'live_data' intent, or the user asks
about current conditions, this module fetches fresh data from the OpenAQ API
and formats it as context for the LLM.
"""

from __future__ import annotations

import os
import json
from datetime import datetime


TOOL_DEFINITIONS = [
    {
        "name": "get_current_aqi",
        "description": "Fetch the latest real-time AQI readings for Delhi from OpenAQ sensors.",
        "parameters": {
            "city": "Delhi",
            "country_iso": "IN",
        },
    },
    {
        "name": "get_historical_trend",
        "description": "Fetch historical daily AQI data for a given number of past days.",
        "parameters": {
            "city": "Delhi",
            "days": 7,
        },
    },
]


def execute_tool(tool_name: str, params: dict | None = None) -> dict:
    """Execute a tool and return structured data.

    Returns:
        {
          "tool": str,
          "success": bool,
          "data": dict | str,
          "timestamp": str,
        }
    """
    params = params or {}

    if tool_name == "get_current_aqi":
        return _tool_current_aqi(params)
    elif tool_name == "get_historical_trend":
        return _tool_historical(params)
    else:
        return {
            "tool": tool_name,
            "success": False,
            "data": f"Unknown tool: {tool_name}",
            "timestamp": _now(),
        }


def auto_tool_call(query: str, snapshot: dict | None = None) -> dict | None:
    """Automatically decide and execute a tool based on query intent.

    If the query needs live data, fetch it. Otherwise return None.
    """
    from rag.query_classifier import classify_query

    intent = classify_query(query)
    config = intent["config"]

    if not config.get("needs_api", False):
        return None

    # If we already have a snapshot (from app.py), package it
    if snapshot:
        return {
            "tool": "cached_snapshot",
            "success": True,
            "data": snapshot,
            "timestamp": _now(),
            "note": "Using cached live data from current session",
        }

    # Otherwise fetch fresh
    return execute_tool("get_current_aqi")


def format_tool_result(result: dict) -> str:
    """Format a tool result as context text for the LLM prompt."""
    if not result or not result.get("success"):
        return ""

    lines = [f"**[Live API Data — {result.get('tool', '?')} @ {result.get('timestamp', '?')}]**"]

    data = result.get("data", {})
    if isinstance(data, dict):
        for k, v in sorted(data.items()):
            if isinstance(v, dict):
                for kk, vv in v.items():
                    lines.append(f"  • {k}.{kk}: {vv}")
            else:
                lines.append(f"  • {k}: {v}")
    else:
        lines.append(str(data))

    return "\n".join(lines)


# ── internal tool implementations ───────────────────────────────────

def _tool_current_aqi(params: dict) -> dict:
    """Fetch current AQI using the OpenAQ client."""
    try:
        from api.openaq_client import get_latest_city_measurements
        from visualization.plots import compute_aqi

        city = params.get("city", "Delhi")
        country = params.get("country_iso", "IN")
        api_key = params.get("api_key", "")

        df = get_latest_city_measurements(city=city, country_iso=country,
                                          api_key=api_key)
        if df.empty:
            return {"tool": "get_current_aqi", "success": False,
                    "data": "No data available", "timestamp": _now()}

        pollutant_vals = {}
        for param in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
            sub = df[df["parameter"] == param]
            if not sub.empty:
                avg = sub["value"].mean()
                unit = sub["unit"].iloc[0] if "unit" in sub.columns else ""
                pollutant_vals[param] = (avg, unit)

        aqi_r = compute_aqi(pollutant_vals)

        snapshot = {}
        for p, (v, u) in pollutant_vals.items():
            snapshot[p] = round(float(v), 1)
        snapshot["AQI"] = aqi_r["aqi"]
        snapshot["category"] = aqi_r["category"]
        snapshot["dominant"] = aqi_r["dominant"]
        snapshot["stations"] = df["location"].nunique() if "location" in df.columns else 0

        return {
            "tool": "get_current_aqi",
            "success": True,
            "data": snapshot,
            "timestamp": _now(),
        }
    except Exception as e:
        return {
            "tool": "get_current_aqi",
            "success": False,
            "data": str(e),
            "timestamp": _now(),
        }


def _tool_historical(params: dict) -> dict:
    """Fetch historical trend data."""
    try:
        from api.openaq_client import get_historical_city

        city = params.get("city", "Delhi")
        days = params.get("days", 7)
        api_key = params.get("api_key", "")

        df = get_historical_city(city=city, days=days, api_key=api_key)
        if df.empty:
            return {"tool": "get_historical_trend", "success": False,
                    "data": "No historical data", "timestamp": _now()}

        summary = {}
        for param in df["parameter"].unique():
            sub = df[df["parameter"] == param]
            summary[param] = {
                "mean": round(float(sub["value"].mean()), 1),
                "max": round(float(sub["value"].max()), 1),
                "min": round(float(sub["value"].min()), 1),
                "readings": len(sub),
            }

        return {
            "tool": "get_historical_trend",
            "success": True,
            "data": {"days": days, "pollutants": summary},
            "timestamp": _now(),
        }
    except Exception as e:
        return {
            "tool": "get_historical_trend",
            "success": False,
            "data": str(e),
            "timestamp": _now(),
        }


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
