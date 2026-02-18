"""Visualization & AQI computation — India NAQI standard.

All chart functions return **Plotly** figure objects for interactive,
modern-looking visuals. AQI logic (breakpoints, sub-index calculation,
unit conversion) remains unchanged.
"""

import math
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# INDIA NAQI BREAKPOINTS
# ═══════════════════════════════════════════════════════════════════════
_I_BREAKS = [0, 50, 100, 200, 300, 400, 500]

_C_BREAKS = {
    "pm25": [0, 30, 60, 90, 120, 250, 380],
    "pm10": [0, 50, 100, 250, 350, 430, 510],
    "no2":  [0, 40, 80, 180, 280, 400, 520],
    "so2":  [0, 40, 80, 380, 800, 1600, 2100],
    "co":   [0, 1.0, 2.0, 10.0, 17.0, 34.0, 46.0],
    "o3":   [0, 50, 100, 168, 208, 748, 940],
}

PPB_TO_UGM3 = {"no2": 1.88, "so2": 2.62, "co": 1.145e-3, "o3": 1.96}

# OpenAQ v3 often reports CO in **ppm** but labels the unit as "ppb".
# Real CO ppb values for urban sites are 500–10 000; values < 100 "ppb"
# are almost certainly ppm.  We convert ppm → mg/m³ using 1.145.
_CO_PPM_THRESHOLD = 100          # if CO "ppb" < this → treat as ppm
_CO_PPM_TO_MGM3   = 1.145        # 1 ppm CO ≈ 1.145 mg/m³

AQI_BANDS = [
    ("Good",         0,  50,  "#009966"),
    ("Satisfactory", 51, 100, "#58B453"),
    ("Moderate",     101, 200, "#FFDE33"),
    ("Poor",         201, 300, "#FF9933"),
    ("Very Poor",    301, 400, "#CC0033"),
    ("Severe",       401, 500, "#7E0023"),
]

HEALTH_ADVICE = {
    "Good":         "Air quality is ideal. Enjoy outdoor activities freely.",
    "Satisfactory": "Acceptable air quality. Sensitive individuals should limit prolonged outdoor exertion.",
    "Moderate":     "Sensitive groups (children, elderly, asthmatics) may feel mild discomfort. Reduce heavy outdoor exercise.",
    "Poor":         "Health effects possible for everyone. Reduce outdoor activity. Sensitive groups should stay indoors.",
    "Very Poor":    "Health alert — serious effects. Avoid outdoor activity. Close windows. Use an air purifier.",
    "Severe":       "Health emergency. Stay indoors, seal gaps, use N95 mask if outside. Seek medical attention if symptomatic.",
}

WHO_GUIDELINES = {
    "pm25": {"limit": 15,   "unit": "µg/m³", "label": "PM2.5",  "period": "24-h"},
    "pm10": {"limit": 45,   "unit": "µg/m³", "label": "PM10",   "period": "24-h"},
    "no2":  {"limit": 25,   "unit": "µg/m³", "label": "NO₂",    "period": "24-h"},
    "so2":  {"limit": 40,   "unit": "µg/m³", "label": "SO₂",    "period": "24-h"},
    "o3":   {"limit": 100,  "unit": "µg/m³", "label": "O₃",     "period": "8-h"},
    "co":   {"limit": 4.0,  "unit": "mg/m³", "label": "CO",     "period": "24-h"},
}

# ═══════════════════════════════════════════════════════════════════════
# AQI COMPUTATION (unchanged)
# ═══════════════════════════════════════════════════════════════════════

def _convert_to_ugm3(param: str, value: float, unit: str) -> float:
    """Convert a pollutant reading to the unit used by India-NAQI breakpoints.

    For CO the breakpoints are in **mg/m³**; every other param is in **µg/m³**.
    """
    unit_l = (unit or "").lower().strip()

    # ── CO special handling ──────────────────────────────────────────
    if param == "co":
        # Already mg/m³ → use directly
        if unit_l in ("mg/m³", "mg/m3"):
            return value
        # µg/m³ → mg/m³
        if unit_l in ("µg/m³", "ug/m3", "µg/m3", "μg/m³"):
            return value / 1000.0
        # ppm → mg/m³
        if unit_l == "ppm":
            return value * _CO_PPM_TO_MGM3
        # "ppb" — but OpenAQ often really means ppm for CO
        if unit_l == "ppb":
            if value < _CO_PPM_THRESHOLD:          # looks like ppm in disguise
                return value * _CO_PPM_TO_MGM3     # ppm → mg/m³
            else:                                  # genuine ppb
                return value * PPB_TO_UGM3["co"]   # ppb → mg/m³
        # Fallback: assume µg/m³ → mg/m³
        return value / 1000.0

    # ── Every other pollutant — target is µg/m³ ─────────────────────
    if unit_l in ("µg/m³", "ug/m3", "µg/m3", "μg/m³"):
        return value
    if unit_l in ("mg/m³", "mg/m3"):
        return value * 1000.0
    if unit_l == "ppb" and param in PPB_TO_UGM3:
        return value * PPB_TO_UGM3[param]
    if unit_l == "ppm" and param in PPB_TO_UGM3:
        return value * PPB_TO_UGM3[param] * 1000.0
    return value                                   # already µg/m³


def convert_for_display(param: str, value: float, unit: str):
    """Return (converted_value, display_unit_str) suitable for the UI.

    Everything is shown in µg/m³ except CO which is in mg/m³.
    """
    if param == "co":
        mg = _convert_to_ugm3(param, value, unit)
        return round(mg, 2), "mg/m³"

    ugm3 = _convert_to_ugm3(param, value, unit)
    return round(ugm3, 1), "µg/m³"


def compute_sub_index(param: str, concentration: float) -> float:
    if param not in _C_BREAKS:
        return -1
    C, I = _C_BREAKS[param], _I_BREAKS
    c = max(concentration, 0)
    if c <= C[0]:
        return 0
    for i in range(1, len(C)):
        if c <= C[i]:
            return ((I[i] - I[i-1]) / (C[i] - C[i-1])) * (c - C[i-1]) + I[i-1]
    return 500


def compute_aqi(pollutant_values: dict) -> dict:
    sub = {}
    for param, (val, unit) in pollutant_values.items():
        conc = _convert_to_ugm3(param, val, unit)
        si = compute_sub_index(param, conc)
        if si >= 0:
            sub[param] = round(si)
    if not sub:
        return {"aqi": 0, "category": "Unknown", "color": "#999",
                "dominant": "", "sub_indices": {}}
    aqi = max(sub.values())
    dominant = max(sub, key=sub.get)
    cat = classify_aqi_value(aqi)
    return {"aqi": aqi, "category": cat, "color": aqi_color(cat),
            "dominant": dominant, "sub_indices": sub}


def classify_aqi_value(aqi: int) -> str:
    for label, lo, hi, _ in AQI_BANDS:
        if aqi <= hi:
            return label
    return "Severe"

def classify_aqi(pm25_ugm3: float) -> str:
    return classify_aqi_value(int(compute_sub_index("pm25", pm25_ugm3)))

def aqi_color(category: str) -> str:
    for label, _, _, color in AQI_BANDS:
        if label == category:
            return color
    return "#999999"

def health_advice(category: str) -> str:
    return HEALTH_ADVICE.get(category, "")


# ═══════════════════════════════════════════════════════════════════════
# PLOTLY CHART THEME
# ═══════════════════════════════════════════════════════════════════════
_LAYOUT = dict(
    font=dict(family="Inter, sans-serif", size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=50, b=20),
    hoverlabel=dict(bgcolor="#1a1a2e", font_color="white", font_size=12),
)


# ═══════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════

def _hex_to_rgba(hex_color: str, opacity: float = 1.0) -> str:
    """Convert #RRGGBB to rgba(r,g,b,opacity) string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


def aqi_gauge(aqi_value: int, category: str = ""):
    """Speedometer-style AQI gauge with gradient sectors."""
    if not category:
        category = classify_aqi_value(aqi_value)
    color = aqi_color(category)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        number=dict(font=dict(size=52, color=color, family="Inter, sans-serif")),
        gauge=dict(
            axis=dict(range=[0, 500], tickwidth=1, tickcolor="#ddd",
                      tickfont=dict(size=10)),
            bar=dict(color=color, thickness=0.3),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[lo, hi], color=_hex_to_rgba(c, 0.2))
                for _, lo, hi, c in AQI_BANDS
            ],
            threshold=dict(
                line=dict(color="#1a1a2e", width=3),
                thickness=0.8,
                value=aqi_value,
            ),
        ),
        title=dict(text=f"<b>{category}</b>", font=dict(size=16, color=color)),
    ))
    fig.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k != "margin"},
        height=260,
        margin=dict(l=30, r=30, t=60, b=10),
    )
    return fig


def sub_index_chart(sub_indices: dict):
    """Horizontal bar chart — each pollutant's sub-index with AQI colour."""
    if not sub_indices:
        return None
    # Sort by value descending so dominant is on top
    items = sorted(sub_indices.items(), key=lambda x: x[1], reverse=True)
    params = [p for p, _ in items]
    values = [v for _, v in items]
    labels = [WHO_GUIDELINES.get(p, {}).get("label", p.upper()) for p in params]
    colors = [aqi_color(classify_aqi_value(v)) for v in values]
    cats   = [classify_aqi_value(v) for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=values,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
            cornerradius=6,
        ),
        text=[f"  {v} — {c}" for v, c in zip(values, cats)],
        textposition="outside",
        textfont=dict(size=12, color="#444"),
        hovertemplate="<b>%{y}</b><br>Sub-index: %{x}<br>Category: %{text}<extra></extra>",
    ))

    # WHO-threshold lines  (faint reference)
    fig.update_layout(
        **_LAYOUT,
        title=dict(text="<b>Pollutant Sub-Index Breakdown</b>", x=0.02, font=dict(size=16)),
        xaxis=dict(title="Sub-Index (0–500)", showgrid=True,
                   gridcolor="rgba(0,0,0,.06)", zeroline=False,
                   range=[0, max(max(values) * 1.25, 120)]),
        yaxis=dict(showgrid=False, autorange="reversed"),
        height=max(200, len(params) * 55 + 80),
        bargap=0.3,
    )
    return fig


def pollutant_vs_who(df: pd.DataFrame):
    """Grouped bars: measured average vs WHO guideline for each pollutant."""
    if df is None or df.empty or "parameter" not in df.columns:
        return None

    rows = []
    for _, row in df.iterrows():
        p = row["parameter"]
        if p not in WHO_GUIDELINES:
            continue
        v = row["value"]
        u = row.get("unit", "")
        conv = _convert_to_ugm3(p, v, u)
        if p == "co":
            conv *= 1000
        rows.append({"parameter": p, "value_ugm3": conv})
    if not rows:
        return None

    cdf = pd.DataFrame(rows)
    agg = cdf.groupby("parameter")["value_ugm3"].mean()
    params = [p for p in ["pm25", "pm10", "no2", "so2", "o3"] if p in agg.index]
    if not params:
        return None

    labels = [WHO_GUIDELINES[p]["label"] for p in params]
    measured = [round(agg[p], 1) for p in params]
    limits   = [WHO_GUIDELINES[p]["limit"] for p in params]
    bar_colors = ["#ef4444" if m > l else "#22c55e" for m, l in zip(measured, limits)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Measured", x=labels, y=measured,
        marker=dict(color=bar_colors, cornerradius=5),
        text=[f"{v}" for v in measured],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.add_trace(go.Scatter(
        name="WHO Limit", x=labels, y=limits,
        mode="markers+text",
        marker=dict(symbol="diamond", size=12, color="#1a1a2e",
                    line=dict(width=2, color="white")),
        text=[f"{l}" for l in limits],
        textposition="top center",
        textfont=dict(size=10, color="#555"),
    ))
    fig.update_layout(
        **_LAYOUT,
        title=dict(text="<b>Measured vs WHO 2021 Guidelines</b> (µg/m³)",
                   x=0.02, font=dict(size=16)),
        yaxis=dict(title="µg/m³", showgrid=True, gridcolor="rgba(0,0,0,.06)",
                   zeroline=False),
        xaxis=dict(showgrid=False),
        height=370,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        bargap=0.35,
    )
    return fig


def timeseries_plot(df: pd.DataFrame, parameter: str = "pm25"):
    """Area/line chart with AQI-coloured background bands."""
    if df is None or df.empty:
        return None
    if not {"parameter", "date_utc", "value"}.issubset(df.columns):
        return None
    sub = (df[df["parameter"] == parameter]
           .sort_values("date_utc").dropna(subset=["date_utc", "value"]))
    if sub.empty:
        return None

    label_map = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂",
                 "so2": "SO₂", "co": "CO", "o3": "O₃"}
    name = label_map.get(parameter, parameter.upper())
    unit = (sub["unit"].iloc[0] if "unit" in sub.columns
            and not sub["unit"].isna().all() else "")

    fig = go.Figure()

    # Colour bands (using concentration breakpoints)
    if parameter in _C_BREAKS:
        breaks = _C_BREAKS[parameter]
        ymax = max(sub["value"].max() * 1.15, breaks[2] * 1.1)
        for i, (label, _, _, color) in enumerate(AQI_BANDS):
            lo_c = breaks[i]
            hi_c = breaks[i + 1] if i + 1 < len(breaks) else ymax
            if lo_c >= ymax:
                break
            fig.add_hrect(
                y0=lo_c, y1=min(hi_c, ymax),
                fillcolor=color, opacity=0.07,
                line_width=0,
                annotation_text=label if hi_c < ymax else "",
                annotation_position="top left",
                annotation=dict(font=dict(size=9, color=color)),
            )
    else:
        ymax = sub["value"].max() * 1.15

    # Main line + area fill
    fig.add_trace(go.Scatter(
        x=sub["date_utc"], y=sub["value"],
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2.5, shape="spline"),
        marker=dict(size=4, color="#3b82f6"),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
        name=name,
        hovertemplate=f"<b>{name}</b><br>"
                      f"Date: %{{x|%b %d, %H:%M}}<br>"
                      f"Value: %{{y:.1f}} {unit}<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT,
        title=dict(text=f"<b>{name} Trend</b>", x=0.02, font=dict(size=15)),
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(title=unit or "Concentration", showgrid=True,
                   gridcolor="rgba(0,0,0,.06)", zeroline=False,
                   range=[0, ymax]),
        height=300,
        showlegend=False,
    )
    return fig


def pollutant_radar(sub_indices: dict):
    """Radar/spider chart of pollutant sub-indices — great overview."""
    if not sub_indices or len(sub_indices) < 3:
        return None
    params = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    available = [p for p in params if p in sub_indices]
    if len(available) < 3:
        return None

    labels = [WHO_GUIDELINES.get(p, {}).get("label", p.upper()) for p in available]
    values = [sub_indices[p] for p in available]
    colors = [aqi_color(classify_aqi_value(v)) for v in values]

    # Close the polygon
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(59,130,246,0.12)",
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=8, color=colors + [colors[0]],
                    line=dict(width=2, color="white")),
        hovertemplate="<b>%{theta}</b><br>Sub-index: %{r}<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(max(values) * 1.2, 150)],
                            gridcolor="rgba(0,0,0,.06)", tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=12, color="#333")),
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(text="<b>Pollutant Footprint</b>", x=0.02, font=dict(size=16)),
        height=350,
        showlegend=False,
    )
    return fig


def aqi_scale_bar():
    """Horizontal AQI scale as a Plotly figure."""
    fig = go.Figure()
    for label, lo, hi, color in AQI_BANDS:
        fig.add_trace(go.Bar(
            x=[hi - lo], y=["AQI"], orientation="h",
            base=lo,
            marker=dict(color=color, cornerradius=2),
            text=f"{label}<br>{lo}–{hi}",
            textposition="inside",
            textfont=dict(size=10, color="white" if lo >= 101 else "#333"),
            hoverinfo="skip",
            showlegend=False,
        ))
    fig.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k != "margin"},
        height=70,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[0, 500]),
        yaxis=dict(visible=False),
        barmode="stack",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 24-HOUR DAILY AQI TREND
# ═══════════════════════════════════════════════════════════════════════

def daily_aqi_trend(
    df_hourly,
    compute_aqi_fn,
    aqi_forecast=None,
    current_aqi: int = 0,
    target_date=None,
):
    """Hour-by-hour AQI trend for a specific date (0–24 h).

    Shows a solid line for observed hours and optionally a dashed line
    for the forecast portion (only when target_date is today).
    AQI band colours appear in the background.

    Args:
        df_hourly:      Hourly DataFrame (parameter, value, unit, date_utc).
        compute_aqi_fn: Function to compute AQI dict from pollutant values.
        aqi_forecast:   Optional AQIForecast for predicted hours (today only).
        current_aqi:    Current live AQI for the "Now" marker (today only).
        target_date:    A datetime.date for the date to display.
                        Defaults to today (IST).
    """
    import datetime as _dt

    if df_hourly is None or df_hourly.empty:
        return None

    df = df_hourly.copy()
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce", utc=True)
    if df["date_utc"].isna().all():
        return None

    # IST offset
    IST = pd.Timedelta(hours=5, minutes=30)

    # Resolve target date (IST)
    now_utc = pd.Timestamp.now("UTC")
    now_ist = now_utc + IST
    if target_date is None:
        target_date = now_ist.date()
    elif hasattr(target_date, "date"):
        target_date = target_date.date()

    is_today = target_date == now_ist.date()

    # Convert to IST and filter to the target calendar day
    df["ist"] = df["date_utc"] + IST
    df["ist_date"] = df["ist"].dt.date
    df = df[df["ist_date"] == target_date]

    # Compute hourly AQI
    if "hour" not in df.columns:
        df["hour"] = df["date_utc"].dt.floor("h")

    hours_in_data = sorted(df["hour"].unique())
    past_hours_utc = []
    past_aqi = []

    for h in hours_in_data:
        hour_df = df[df["hour"] == h]
        vals = {}
        for _, row in hour_df.iterrows():
            p = row["parameter"]
            vals[p] = (row["value"], row.get("unit", "µg/m³"))
        if vals:
            result = compute_aqi_fn(vals)
            past_hours_utc.append(pd.Timestamp(h))
            past_aqi.append(result.get("aqi", 0))

    if not past_hours_utc:
        return None

    # IST integer hours (0-23) for x-axis
    def _ist_hour(ts_utc):
        return (pd.Timestamp(ts_utc) + IST).hour

    past_ist_hours = [_ist_hour(h) for h in past_hours_utc]

    # Full 0-24 x-axis tick labels
    hour_labels = [f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}" for h in range(25)]

    # Build forecast portion (only for today)
    forecast_ist_hours = []
    forecast_aqi_vals = []
    forecast_lower = []
    forecast_upper = []
    if is_today and aqi_forecast and aqi_forecast.forecasted_aqi:
        for f in aqi_forecast.forecasted_aqi:
            ts = f["timestamp"]
            if hasattr(ts, "tz_localize") and ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            fh = _ist_hour(ts)
            # Only include forecast hours that belong to the same calendar day
            f_ist = pd.Timestamp(ts) + IST
            if f_ist.date() == target_date:
                forecast_ist_hours.append(fh)
                forecast_aqi_vals.append(f["aqi"])
                forecast_lower.append(f["lower"])
                forecast_upper.append(f["upper"])

    fig = go.Figure()

    # y-axis ceiling
    all_vals = past_aqi + forecast_aqi_vals + ([current_aqi] if is_today else [])
    ymax = max(max(all_vals) if all_vals else 100, 150) * 1.2

    # AQI band backgrounds
    for label_b, lo, hi, color in AQI_BANDS:
        if lo >= ymax:
            break
        fig.add_hrect(
            y0=lo, y1=min(hi, ymax),
            fillcolor=color, opacity=0.06,
            line_width=0,
            annotation_text=label_b if hi < ymax else "",
            annotation_position="top right",
            annotation=dict(font=dict(size=9, color=color)),
        )

    # ── Past hours: solid line ──
    past_colors = [aqi_color(classify_aqi_value(v)) for v in past_aqi]
    fig.add_trace(go.Scatter(
        x=past_ist_hours, y=past_aqi,
        mode="lines+markers",
        line=dict(color="#3b82f6", width=3, shape="spline"),
        marker=dict(size=6, color=past_colors,
                    line=dict(width=1.5, color="white")),
        name="Observed",
        hovertemplate="<b>%{text}</b><br>AQI: %{y}<extra></extra>",
        text=[hour_labels[h] for h in past_ist_hours],
    ))

    # ── "Now" marker (today only) ──
    if is_today and past_ist_hours:
        now_h = past_ist_hours[-1]
        now_val = past_aqi[-1]
        now_color = aqi_color(classify_aqi_value(now_val))
        fig.add_trace(go.Scatter(
            x=[now_h], y=[now_val],
            mode="markers+text",
            marker=dict(size=16, color=now_color,
                        line=dict(width=3, color="white"),
                        symbol="circle"),
            text=[f"Now: {now_val}"],
            textposition="top center",
            textfont=dict(size=11, color=now_color, family="Inter, sans-serif"),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Forecast: dashed line with confidence band (today only) ──
    if forecast_ist_hours and past_ist_hours:
        bridge_h = [past_ist_hours[-1]] + forecast_ist_hours
        bridge_v = [past_aqi[-1]] + forecast_aqi_vals
        bridge_lo = [past_aqi[-1]] + forecast_lower
        bridge_hi = [past_aqi[-1]] + forecast_upper

        fig.add_trace(go.Scatter(
            x=bridge_h, y=bridge_hi,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=bridge_h, y=bridge_lo,
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(249,115,22,0.10)",
            showlegend=False, hoverinfo="skip",
        ))

        fc_colors = [aqi_color(classify_aqi_value(v)) for v in forecast_aqi_vals]
        fig.add_trace(go.Scatter(
            x=bridge_h, y=bridge_v,
            mode="lines+markers",
            line=dict(color="#f97316", width=2.5, dash="dot", shape="spline"),
            marker=dict(size=7, color=["#3b82f6"] + fc_colors,
                        line=dict(width=1.5, color="white")),
            name="Forecast",
            hovertemplate="<b>%{text}</b><br>AQI: %{y} (predicted)<extra></extra>",
            text=[hour_labels[h] for h in bridge_h],
        ))

    # Date label for the title
    date_str = target_date.strftime("%A, %d %B %Y")   # e.g. "Monday, 06 January 2025"
    title_prefix = "Today" if is_today else date_str
    title_text = f"<b>{title_prefix}'s AQI Trend</b> — {date_str}"

    fig.update_layout(
        **_LAYOUT,
        title=dict(text=title_text, x=0.02, font=dict(size=16)),
        xaxis=dict(
            title="Hour (IST)", showgrid=False,
            range=[-0.5, 24.5],
            tickmode="array",
            tickvals=list(range(0, 25, 2)),
            ticktext=[hour_labels[h] for h in range(0, 25, 2)],
            tickangle=-45,
            tickfont=dict(size=10),
        ),
        yaxis=dict(title="AQI", showgrid=True,
                   gridcolor="rgba(0,0,0,.06)", zeroline=False,
                   range=[0, ymax]),
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# FORECAST CHARTS
# ═══════════════════════════════════════════════════════════════════════

def forecast_chart(aqi_forecast, current_aqi: int = 0):
    """AQI forecast line chart with confidence band and category background.

    Args:
        aqi_forecast: AQIForecast object from forecasting module.
        current_aqi:  Current AQI to anchor the "now" point.
    """
    if not aqi_forecast or not aqi_forecast.forecasted_aqi:
        return None

    forecasts = aqi_forecast.forecasted_aqi

    # Build arrays: now + forecast hours
    timestamps = [pd.Timestamp.now("UTC").floor("h")]  # "Now"
    aqi_vals = [current_aqi or aqi_forecast.current_aqi]
    lower_vals = [current_aqi or aqi_forecast.current_aqi]
    upper_vals = [current_aqi or aqi_forecast.current_aqi]
    confidences = [1.0]

    for f in forecasts:
        timestamps.append(f["timestamp"])
        aqi_vals.append(f["aqi"])
        lower_vals.append(f["lower"])
        upper_vals.append(f["upper"])
        confidences.append(f["confidence"])

    # Convert to IST-like labels
    labels = ["Now"] + [f"+{f['hour_offset']}h" for f in forecasts]

    fig = go.Figure()

    # AQI band backgrounds
    ymax = max(max(upper_vals) * 1.2, 200)
    for label_b, lo, hi, color in AQI_BANDS:
        if lo >= ymax:
            break
        fig.add_hrect(
            y0=lo, y1=min(hi, ymax),
            fillcolor=color, opacity=0.06,
            line_width=0,
            annotation_text=label_b if hi < ymax else "",
            annotation_position="top right",
            annotation=dict(font=dict(size=9, color=color)),
        )

    # Confidence band (filled area)
    fig.add_trace(go.Scatter(
        x=labels, y=upper_vals,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=lower_vals,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(59,130,246,0.12)",
        name="Confidence Band",
        hoverinfo="skip",
    ))

    # Main forecast line
    colors = [aqi_color(classify_aqi_value(v)) for v in aqi_vals]
    fig.add_trace(go.Scatter(
        x=labels, y=aqi_vals,
        mode="lines+markers+text",
        line=dict(color="#3b82f6", width=3, shape="spline"),
        marker=dict(size=12, color=colors,
                    line=dict(width=2, color="white")),
        text=[str(v) for v in aqi_vals],
        textposition="top center",
        textfont=dict(size=11, color="#333"),
        name="Predicted AQI",
        hovertemplate="<b>%{x}</b><br>AQI: %{y}<br>"
                      "Range: %{customdata[0]}–%{customdata[1]}<br>"
                      "Confidence: %{customdata[2]:.0%}<extra></extra>",
        customdata=list(zip(lower_vals, upper_vals, confidences)),
    ))

    # "Now" marker (larger)
    fig.add_trace(go.Scatter(
        x=["Now"], y=[aqi_vals[0]],
        mode="markers",
        marker=dict(size=18, color=colors[0],
                    line=dict(width=3, color="white"),
                    symbol="circle"),
        showlegend=False, hoverinfo="skip",
    ))

    trend_text = f"Trend: {aqi_forecast.overall_trend.title()} {aqi_forecast.overall_trend_emoji}"
    conf_text = f"Confidence: {aqi_forecast.confidence_level.title()}"

    fig.update_layout(
        **_LAYOUT,
        title=dict(
            text=f"<b>AQI Forecast — Next {len(forecasts)} Hours</b>"
                 f"<br><sup>{trend_text} · {conf_text}</sup>",
            x=0.02, font=dict(size=16),
        ),
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(title="AQI", showgrid=True,
                   gridcolor="rgba(0,0,0,.06)", zeroline=False,
                   range=[0, ymax]),
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05,
                    xanchor="right", x=1),
    )
    return fig


def pollutant_forecast_sparklines(aqi_forecast):
    """Small multiples — sparkline forecast for each pollutant.

    Returns a single figure with subplots.
    """
    if not aqi_forecast or not aqi_forecast.pollutant_forecasts:
        return None

    from plotly.subplots import make_subplots

    pf_items = list(aqi_forecast.pollutant_forecasts.items())
    n = len(pf_items)
    if n == 0:
        return None

    cols = min(n, 3)
    rows = math.ceil(n / cols)

    labels_map = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂",
                  "so2": "SO₂", "co": "CO", "o3": "O₃"}

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{labels_map.get(p, p)} {pf.trend_emoji}"
                        for p, pf in pf_items],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for idx, (param, pf) in enumerate(pf_items):
        r = idx // cols + 1
        c = idx % cols + 1

        # Historical (current value as anchor)
        x_vals = ["Now"] + [f"+{pt.hour_offset}h" for pt in pf.points]
        y_vals = [pf.current_value] + [pt.value for pt in pf.points]
        y_upper = [pf.current_value] + [pt.upper for pt in pf.points]
        y_lower = [pf.current_value] + [pt.lower for pt in pf.points]

        # Confidence band
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_upper, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_lower, mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(59,130,246,0.1)",
            showlegend=False, hoverinfo="skip",
        ), row=r, col=c)

        # Main line
        trend_color = {"rising": "#ef4444", "falling": "#22c55e",
                       "stable": "#3b82f6"}.get(pf.trend, "#3b82f6")
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers",
            line=dict(color=trend_color, width=2.5, shape="spline"),
            marker=dict(size=6, color=trend_color),
            name=labels_map.get(param, param),
            showlegend=False,
            hovertemplate=f"<b>{labels_map.get(param, param)}</b><br>"
                          f"%{{x}}: %{{y:.1f}} {pf.unit}<extra></extra>",
        ), row=r, col=c)

    fig.update_layout(
        **_LAYOUT,
        height=200 * rows + 60,
        title=dict(text="<b>Pollutant Forecasts</b>", x=0.02,
                   font=dict(size=15)),
    )

    # Clean up axes
    for i in range(1, rows * cols + 1):
        fig.update_xaxes(showgrid=False, row=(i - 1) // cols + 1,
                         col=(i - 1) % cols + 1)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,.04)",
                         row=(i - 1) // cols + 1, col=(i - 1) % cols + 1)

    return fig


# ═══════════════════════════════════════════════════════════════════════
# STATION-WISE CHARTS
# ═══════════════════════════════════════════════════════════════════════

def station_comparison_chart(df_latest: pd.DataFrame, parameter: str = "pm25"):
    """Bar chart comparing a pollutant across monitoring stations."""
    if df_latest is None or df_latest.empty:
        return None
    if "location" not in df_latest.columns:
        return None

    sub = df_latest[df_latest["parameter"] == parameter].copy()
    if sub.empty:
        return None

    labels_map = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂",
                  "so2": "SO₂", "co": "CO", "o3": "O₃"}
    name = labels_map.get(parameter, parameter.upper())
    unit = sub["unit"].iloc[0] if "unit" in sub.columns else "µg/m³"

    agg = sub.groupby("location")["value"].mean().sort_values(ascending=True)
    if agg.empty:
        return None

    stations = agg.index.tolist()
    values = agg.values

    # Color by AQI category at that station's value
    if parameter in _C_BREAKS:
        colors = [aqi_color(classify_aqi_value(int(compute_sub_index(parameter, v))))
                  for v in values]
    else:
        colors = ["#3b82f6"] * len(values)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=stations, x=values,
        orientation="h",
        marker=dict(color=colors, cornerradius=6),
        text=[f"  {v:.0f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, color="#444"),
        hovertemplate=f"<b>%{{y}}</b><br>{name}: %{{x:.1f}} {unit}<extra></extra>",
    ))

    # WHO line
    who_limit = {
        "pm25": 15, "pm10": 45, "no2": 25, "so2": 40, "o3": 100
    }.get(parameter)
    if who_limit:
        fig.add_vline(x=who_limit, line_dash="dash",
                      line_color="#1a1a2e", line_width=1.5,
                      annotation_text=f"WHO: {who_limit}",
                      annotation_position="top right",
                      annotation_font=dict(size=9, color="#555"))

    fig.update_layout(
        **_LAYOUT,
        title=dict(text=f"<b>{name} by Station</b> ({unit})",
                   x=0.02, font=dict(size=15)),
        xaxis=dict(title=f"{name} ({unit})", showgrid=True,
                   gridcolor="rgba(0,0,0,.06)", zeroline=False),
        yaxis=dict(showgrid=False),
        height=max(200, len(stations) * 40 + 100),
        bargap=0.25,
    )
    return fig


def station_aqi_heatmap(df_latest: pd.DataFrame):
    """Heatmap of pollutant values across stations.

    Rows = stations, Columns = pollutants, Color = sub-index severity.
    """
    if df_latest is None or df_latest.empty or "location" not in df_latest.columns:
        return None

    params = ["pm25", "pm10", "no2", "so2", "o3", "co"]
    labels = ["PM2.5", "PM10", "NO₂", "SO₂", "O₃", "CO"]
    available_params = [p for p in params
                        if p in df_latest["parameter"].values]
    if not available_params:
        return None

    stations = sorted(df_latest["location"].unique())
    if len(stations) < 2:
        return None

    # Build matrix of sub-indices
    matrix = []
    text_matrix = []
    for station in stations:
        row = []
        text_row = []
        for param in available_params:
            sub = df_latest[(df_latest["location"] == station) &
                            (df_latest["parameter"] == param)]
            if sub.empty:
                row.append(0)
                text_row.append("N/A")
            else:
                val = sub["value"].mean()
                si = compute_sub_index(param, val)
                row.append(si)
                text_row.append(f"{val:.0f} (SI:{si:.0f})")
        matrix.append(row)
        text_matrix.append(text_row)

    z = np.array(matrix)
    col_labels = [labels[params.index(p)] for p in available_params]

    # Custom AQI colorscale
    colorscale = [
        [0.0, "#009966"],     # Good
        [0.1, "#009966"],
        [0.1, "#58bc2b"],     # Satisfactory
        [0.2, "#58bc2b"],
        [0.2, "#ffbf00"],     # Moderate
        [0.4, "#ffbf00"],
        [0.4, "#ff5722"],     # Poor
        [0.6, "#ff5722"],
        [0.6, "#960032"],     # Very Poor
        [0.8, "#960032"],
        [0.8, "#7E0023"],     # Severe
        [1.0, "#7E0023"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=col_labels,
        y=stations,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale=colorscale,
        zmin=0, zmax=500,
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        colorbar=dict(title="Sub-Index", tickvals=[50, 100, 200, 300, 400],
                      ticktext=["Good", "Satisfactory", "Moderate", "Poor", "V.Poor"]),
    ))

    fig.update_layout(
        **_LAYOUT,
        title=dict(text="<b>Station × Pollutant Heatmap</b>",
                   x=0.02, font=dict(size=15)),
        height=max(250, len(stations) * 35 + 100),
        xaxis=dict(side="top"),
    )
    return fig


def station_detail_chart(df_station: pd.DataFrame, station_name: str = ""):
    """Detailed view for a single selected station — bar chart of all pollutants."""
    if df_station is None or df_station.empty:
        return None

    labels_map = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂",
                  "so2": "SO₂", "co": "CO", "o3": "O₃"}
    params_order = ["pm25", "pm10", "no2", "so2", "o3", "co"]

    records = []
    for param in params_order:
        sub = df_station[df_station["parameter"] == param]
        if sub.empty:
            continue
        val = sub["value"].mean()
        si = compute_sub_index(param, val)
        cat = classify_aqi_value(int(si))
        clr = aqi_color(cat)
        records.append({
            "param": param,
            "label": labels_map.get(param, param),
            "value": val,
            "sub_index": si,
            "category": cat,
            "color": clr,
        })

    if not records:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[r["label"] for r in records],
        y=[r["sub_index"] for r in records],
        marker=dict(
            color=[r["color"] for r in records],
            cornerradius=6,
        ),
        text=[f"{r['sub_index']:.0f}<br>({r['value']:.0f})" for r in records],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{x}</b><br>Sub-index: %{y:.0f}<br>"
                      "Category: %{customdata}<extra></extra>",
        customdata=[r["category"] for r in records],
    ))

    # AQI band backgrounds
    ymax = max(r["sub_index"] for r in records) * 1.3
    for label_b, lo, hi, color in AQI_BANDS:
        if lo >= ymax:
            break
        fig.add_hrect(y0=lo, y1=min(hi, ymax),
                      fillcolor=color, opacity=0.05, line_width=0)

    title = f"<b>{station_name}</b> — Pollutant Breakdown" if station_name else "<b>Station Detail</b>"
    fig.update_layout(
        **_LAYOUT,
        title=dict(text=title, x=0.02, font=dict(size=15)),
        yaxis=dict(title="Sub-Index", showgrid=True,
                   gridcolor="rgba(0,0,0,.06)", range=[0, ymax]),
        xaxis=dict(showgrid=False),
        height=320,
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# AQI FORECAST CHART
# ═══════════════════════════════════════════════════════════════════════
def forecast_chart(forecast_obj, current_aqi: float = None):
    """
    Plotly chart: 6-hour AQI forecast with confidence bands
    and colour-coded AQI category background.
    """
    if not forecast_obj or not forecast_obj.forecasted_aqi:
        return None

    points = forecast_obj.forecasted_aqi
    hours = [0] + [p["hour_offset"] for p in points]
    aqi_vals = [current_aqi or forecast_obj.current_aqi] + [p["aqi"] for p in points]
    lower_b = [current_aqi or forecast_obj.current_aqi] + [p["lower"] for p in points]
    upper_b = [current_aqi or forecast_obj.current_aqi] + [p["upper"] for p in points]
    labels = ["Now"] + [f"+{p['hour_offset']}h" for p in points]

    fig = go.Figure()

    # AQI category background bands
    band_data = [
        (0, 50, "Good", "#2ecc71", 0.08),
        (50, 100, "Satisfactory", "#27ae60", 0.06),
        (100, 200, "Moderate", "#f1c40f", 0.06),
        (200, 300, "Poor", "#e67e22", 0.06),
        (300, 400, "Very Poor", "#e74c3c", 0.06),
        (400, 500, "Severe", "#8e44ad", 0.06),
    ]
    y_max = max(upper_b) * 1.15
    for lo, hi, lbl, clr, opa in band_data:
        if lo < y_max:
            fig.add_hrect(
                y0=lo, y1=min(hi, y_max),
                fillcolor=clr, opacity=opa, line_width=0,
                annotation_text=lbl if hi <= y_max else "",
                annotation_position="right",
                annotation_font=dict(size=9, color=clr),
            )

    # Confidence band
    fig.add_trace(go.Scatter(
        x=hours, y=upper_b, mode="lines",
        line=dict(width=0), showlegend=False, name="Upper",
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=lower_b, mode="lines",
        fill="tonexty", fillcolor="rgba(59,130,246,0.12)",
        line=dict(width=0), showlegend=True, name="Confidence Band",
    ))

    # Main forecast line
    colors = [aqi_color(classify_aqi_value(v)) for v in aqi_vals]
    fig.add_trace(go.Scatter(
        x=hours, y=aqi_vals, mode="lines+markers",
        name="Predicted AQI",
        line=dict(color="#3b82f6", width=3),
        marker=dict(size=10, color=colors, line=dict(width=2, color="#fff")),
        text=[f"AQI: {v:.0f}<br>{classify_aqi_value(v)}" for v in aqi_vals],
        hovertemplate="%{text}<extra></extra>",
    ))

    # Current AQI diamond
    fig.add_trace(go.Scatter(
        x=[0], y=[aqi_vals[0]], mode="markers",
        marker=dict(size=14, color=aqi_color(classify_aqi_value(aqi_vals[0])),
                    symbol="diamond", line=dict(width=2, color="#fff")),
        name="Current AQI",
        text=[f"Current: {aqi_vals[0]:.0f}"],
        hovertemplate="%{text}<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT,
        title=dict(text="<b>6-Hour AQI Forecast</b>", x=0.02,
                   font=dict(size=16)),
        xaxis=dict(tickvals=hours, ticktext=labels, title="", showgrid=False),
        yaxis=dict(title="AQI", showgrid=True,
                   gridcolor="rgba(0,0,0,.06)", range=[0, y_max]),
        height=380,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig
