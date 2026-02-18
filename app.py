"""
Delhi AQI Intelligence Platform
================================
Professional air quality analytics dashboard:
  - Live AQI monitoring   (India NAQI standard, 24h median)
  - Station-wise analysis  (15+ monitoring stations across Delhi/NCR)
  - ML-powered 6-hour AQI forecast (LightGBM, regime-aware)
  - Dynamic context-aware insights engine
  - AI expert analyst      (RAG + Llama 3.3 70B via Groq)
  - Interactive Plotly visualizations
"""

import streamlit as st
import pandas as pd
import os, sys
import datetime as _dtmod

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.openaq_client import (
    get_latest_city_measurements,
    get_historical_city,
    get_hourly_data,
    list_city_stations,
    get_station_latest,
)
from rag.prompt_template import build_prompt
from visualization.plots import (
    compute_aqi, classify_aqi_value, aqi_color, health_advice,
    aqi_gauge, sub_index_chart, pollutant_vs_who, timeseries_plot,
    pollutant_radar, aqi_scale_bar,
    station_comparison_chart, station_aqi_heatmap, station_detail_chart,
    convert_for_display, forecast_chart,
    AQI_BANDS, WHO_GUIDELINES,
)
from insights.engine import generate_insights
from forecasting.inference import forecast_next_6_hours

# ─── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delhi AQI Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Professional Analytics CSS ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.block-container {
    padding-top: .5rem; padding-bottom: 1rem; max-width: 1280px;
}
header[data-testid="stHeader"] { background: transparent; }

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: .3; }
}
@keyframes barGrow {
    from { width: 0; }
    to   { width: var(--bar-w); }
}
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* ── Navigation Bar ── */
.nav-bar {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e2e8f0; padding: 10px 24px; border-radius: 14px;
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 16px; font-size: .82rem;
    border: 1px solid rgba(255,255,255,.06);
    box-shadow: 0 4px 20px rgba(0,0,0,.12);
    animation: fadeIn .4s ease-out;
}
.nav-bar .nav-title {
    font-weight: 700; font-size: .95rem;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: .3px;
}
.nav-bar .nav-meta {
    display: flex; align-items: center; gap: 16px; font-size: .78rem;
    color: #94a3b8;
}
.nav-bar .live-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #34d399;
    animation: pulse 1.8s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(52,211,153,.5);
    display: inline-block; margin-right: 4px;
}

/* ── Hero Banner ── */
.hero {
    text-align: center; padding: 40px 28px 34px; border-radius: 24px;
    color: #fff; overflow: hidden; position: relative;
    box-shadow: 0 12px 48px rgba(0,0,0,.18);
    animation: fadeInUp .5s ease-out;
}
.hero::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(circle at 25% 15%, rgba(255,255,255,.15) 0%, transparent 55%),
                radial-gradient(circle at 75% 85%, rgba(0,0,0,.12) 0%, transparent 45%);
    pointer-events: none;
}
.hero::after {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(180deg, transparent 60%, rgba(0,0,0,.08) 100%);
    pointer-events: none;
}
.hero .live-badge {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(255,255,255,.12); padding: 5px 18px;
    border-radius: 24px; font-size: .72rem; font-weight: 600;
    letter-spacing: .8px; margin-bottom: 16px;
    backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,.1);
    text-transform: uppercase;
}
.hero .live-badge .pulse-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #4ade80;
    animation: pulse 1.6s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(74,222,128,.6);
}
.hero .aqi-number {
    font-size: 6rem; font-weight: 900; line-height: 1; margin: 0;
    text-shadow: 0 4px 24px rgba(0,0,0,.2);
    letter-spacing: -3px;
}
.hero .aqi-label {
    font-size: .88rem; font-weight: 500; margin: 8px 0 0;
    opacity: .75; letter-spacing: 2px; text-transform: uppercase;
}
.hero .hero-tags {
    font-size: .88rem; margin-top: 14px;
    display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;
}
.hero .hero-tags .tag {
    background: rgba(255,255,255,.1); padding: 5px 16px; border-radius: 10px;
    font-size: .8rem; font-weight: 500; backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,.08);
    letter-spacing: .3px;
}

/* ── AQI Scale Strip ── */
.scale-strip {
    display: flex; border-radius: 10px; overflow: hidden;
    height: 7px; margin: 14px 0 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}
.scale-strip .seg { flex: 1; transition: opacity .2s; }
.scale-strip:hover .seg { opacity: .85; }
.scale-labels {
    display: flex; justify-content: space-between;
    font-size: .6rem; color: #94a3b8; padding: 0 2px; margin-bottom: 12px;
    font-weight: 500;
}

/* ── Section Header ── */
.section-header {
    font-size: 1.1rem; font-weight: 700; margin: 28px 0 14px;
    padding-bottom: 8px; position: relative; color: #1e293b;
    letter-spacing: -.2px;
}
.section-header::after {
    content: ''; position: absolute; bottom: 0; left: 0;
    width: 40px; height: 3px; border-radius: 2px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
}

/* ── Pollutant Card ── */
.poll-card {
    background: #fff; border-radius: 16px; padding: 20px 14px 16px;
    text-align: center; position: relative; overflow: hidden;
    box-shadow: 0 1px 8px rgba(0,0,0,.04), 0 4px 16px rgba(0,0,0,.02);
    border: 1px solid rgba(0,0,0,.04);
    transition: all .25s cubic-bezier(.4,0,.2,1);
    animation: fadeInUp .45s ease-out both;
}
.poll-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--accent); border-radius: 3px 3px 0 0;
}
.poll-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 8px 32px rgba(0,0,0,.1);
}
.poll-card .p-value {
    font-size: 2rem; font-weight: 800; line-height: 1;
    letter-spacing: -.5px;
    font-family: 'JetBrains Mono', monospace;
}
.poll-card .p-name {
    font-size: .78rem; color: #64748b; margin-top: 6px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .8px;
}
.poll-card .p-unit {
    font-size: .65rem; color: #94a3b8; margin-top: 3px;
    font-family: 'JetBrains Mono', monospace;
}
.poll-card .p-bar-track {
    height: 4px; border-radius: 4px; margin-top: 12px;
    background: #f1f5f9; overflow: hidden;
}
.poll-card .p-bar-fill {
    height: 100%; border-radius: 4px;
    animation: barGrow .8s ease-out both;
    width: var(--bar-w);
}

/* ── KPI Card ── */
.kpi-card {
    background: #fff; border-radius: 16px; padding: 24px 18px;
    text-align: center; position: relative; overflow: hidden;
    box-shadow: 0 1px 8px rgba(0,0,0,.04), 0 4px 16px rgba(0,0,0,.02);
    border: 1px solid rgba(0,0,0,.04);
    animation: fadeInUp .5s ease-out both;
    transition: all .25s cubic-bezier(.4,0,.2,1);
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 28px rgba(0,0,0,.08);
}
.kpi-card .kpi-icon {
    width: 36px; height: 36px; border-radius: 10px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: .75rem; font-weight: 700; margin-bottom: 10px;
    color: #fff; font-family: 'JetBrains Mono', monospace;
}
.kpi-card .kpi-value {
    font-size: 2.5rem; font-weight: 800; line-height: 1;
    letter-spacing: -1px;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-card .kpi-label {
    font-size: .78rem; color: #64748b; margin-top: 8px; line-height: 1.4;
    font-weight: 500;
}
.kpi-card .kpi-sub {
    font-size: .65rem; color: #94a3b8; margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Health Card ── */
.health-card {
    background: #fff; border-radius: 14px; padding: 18px 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,.04);
    border-left: 4px solid var(--severity-color, #94a3b8);
    margin-bottom: 8px; animation: fadeInUp .4s ease-out both;
    transition: all .2s ease;
}
.health-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,.08);
}
.health-card .hc-title {
    font-weight: 700; font-size: .9rem; margin-bottom: 4px;
    display: flex; align-items: center; gap: 8px;
}
.health-card .hc-title .severity-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}
.health-card .hc-body {
    font-size: .82rem; color: #64748b; line-height: 1.5;
}

/* ── Action Pill ── */
.action-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: #f8fafc; border-radius: 10px; padding: 8px 16px;
    border: 1px solid #e2e8f0; font-size: .82rem; font-weight: 500;
    color: #334155; transition: all .2s;
}
.action-pill:hover { background: #f1f5f9; }
.action-pill .pill-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}

/* ── Tab Styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; border-bottom: 2px solid #f1f5f9;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0; font-weight: 600;
    font-size: .85rem; padding: 8px 20px;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    border-radius: 12px !important;
    border-color: #e2e8f0 !important;
}

/* ── Data Table ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 600 !important; font-size: .88rem !important;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .hero .aqi-number { font-size: 4rem; }
    .nav-bar { flex-direction: column; gap: 8px; text-align: center; }
    .kpi-card .kpi-value { font-size: 2rem; }
}
</style>
""", unsafe_allow_html=True)

# ─── Helper: resolve secrets (env var > Streamlit secrets > "") ─────
def _get_secret(name: str) -> str:
    """Resolve a secret from env var or Streamlit Cloud secrets."""
    val = os.environ.get(name, "")
    if not val:
        try:
            val = st.secrets.get(name, "")
        except Exception:
            val = ""
    return val

# ─── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")
    openaq_key = st.text_input(
        "OpenAQ API Key",
        value=_get_secret("OPENAQ_API_KEY"),
        type="password",
    )
    if openaq_key:
        os.environ["OPENAQ_API_KEY"] = openaq_key

    groq_key = st.text_input(
        "Groq API Key (Llama 3.3 70B)",
        value=_get_secret("GROQ_API_KEY"),
        type="password",
        help="Free key at https://console.groq.com",
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    st.markdown("---")
    city = st.text_input("City", value="Delhi")
    country = st.text_input("ISO Code", value="IN")
    hist_days = st.slider("History (days)", 1, 30, 7)

    if st.button("Clear chat history"):
        st.session_state.pop("chat_messages", None)
        st.rerun()

# ─── Fetch Data ─────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner="Fetching live data from OpenAQ...")
def fetch_data(city, country_iso, days, api_key):
    try:
        latest = get_latest_city_measurements(
            city=city, country_iso=country_iso, limit=200, api_key=api_key)
    except Exception:
        latest = pd.DataFrame()
    try:
        hist = get_historical_city(
            city=city, country_iso=country_iso, days=days, limit=3000,
            api_key=api_key)
    except Exception:
        hist = pd.DataFrame()
    return latest, hist

@st.cache_data(ttl=600, show_spinner="Fetching hourly data...")
def fetch_hourly(city, country_iso, api_key, _current_values_hash=""):
    try:
        return get_hourly_data(city=city, country_iso=country_iso,
                               hours=24, api_key=api_key)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner="Loading station list...")
def fetch_stations(city, country_iso, api_key):
    try:
        return list_city_stations(city=city, country_iso=country_iso,
                                  api_key=api_key)
    except Exception:
        return []

df_latest, df_hist = fetch_data(city, country, hist_days, openaq_key)
df_hourly = fetch_hourly(city, country, openaq_key)
stations_list = fetch_stations(city, country, openaq_key)

if df_latest.empty:
    st.error("No data available. Check your OpenAQ API key in the sidebar.")
    st.stop()

# ─── Compute AQI — India NAQI (24h median across stations) ─────────

def _compute_24h_averages(df_hourly_data, df_latest_data):
    """Compute India-NAQI compliant pollutant concentrations.

    Uses 24h MEDIAN across stations for all pollutants.
    Falls back to median of latest readings when hourly data is absent.
    """
    result = {}
    has_hourly = (df_hourly_data is not None and not df_hourly_data.empty
                  and "location" in df_hourly_data.columns
                  and df_hourly_data["location"].iloc[0] != "synthetic")

    for param in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        if has_hourly:
            sub = df_hourly_data[df_hourly_data["parameter"] == param]
            if not sub.empty:
                unit = sub["unit"].iloc[0] if "unit" in sub.columns else ""
                val = float(sub["value"].median())
                result[param] = (val, unit)
                continue
        sub_latest = df_latest_data[df_latest_data["parameter"] == param]
        if not sub_latest.empty:
            val = float(sub_latest["value"].median())
            unit = sub_latest["unit"].iloc[0] if "unit" in sub_latest.columns else ""
            result[param] = (val, unit)
    return result

pollutant_vals = _compute_24h_averages(df_hourly, df_latest)

aqi_r    = compute_aqi(pollutant_vals)
aqi_val  = aqi_r["aqi"]
aqi_cat  = aqi_r["category"]
aqi_clr  = aqi_r["color"]
dominant = aqi_r["dominant"]
sub_idx  = aqi_r["sub_indices"]
pm25_avg = pollutant_vals.get("pm25", (0, ""))[0]
n_stations = df_latest["location"].nunique() if "location" in df_latest.columns else "?"

# Ensure hourly data is anchored on live values
if df_hourly.empty or (
    "location" in df_hourly.columns
    and df_hourly["location"].iloc[0] == "synthetic"
):
    from api.openaq_client import _demo_hourly
    df_hourly = _demo_hourly(hours=24, current_values=pollutant_vals)

# ─── Build snapshot for chat context ────────────────────────────────
_snapshot = {}
for p, (v, u) in pollutant_vals.items():
    dv, du = convert_for_display(p, v, u)
    _snapshot[f"{p}"] = dv
    _snapshot[f"{p}_unit"] = du
_snapshot["AQI"] = aqi_val
_snapshot["AQI_category"] = aqi_cat
_snapshot["dominant_pollutant"] = dominant
for p, si in sub_idx.items():
    _snapshot[f"{p}_subindex"] = si
if pm25_avg:
    _snapshot["cigarette_equivalent_per_day"] = round(pm25_avg / 22, 1)

# ─── Generate Dynamic Insights ─────────────────────────────────────
try:
    _insights = generate_insights(
        aqi_val=aqi_val,
        aqi_category=aqi_cat,
        pollutant_vals=pollutant_vals,
        df_latest=df_latest,
        df_hourly=df_hourly if not df_hourly.empty else None,
        aqi_forecast=None,
        max_insights=6,
    )
except Exception:
    _insights = []

# ─── Compute 6-Hour Forecast ────────────────────────────────────────
try:
    _aqi_forecast = forecast_next_6_hours(
        df_hourly, current_aqi=aqi_val, pollutant_vals={
            k: v for k, (v, _u) in pollutant_vals.items()
        }, horizon=6,
    )
except Exception as _fc_err:
    _aqi_forecast = None

# Add forecast to snapshot for chat context
if _aqi_forecast and _aqi_forecast.forecasted_aqi:
    _snapshot["forecast_trend"] = _aqi_forecast.overall_trend
    _snapshot["forecast_confidence"] = _aqi_forecast.confidence_level
    for f in _aqi_forecast.forecasted_aqi:
        _snapshot[f"aqi_+{f['hour_offset']}h"] = f["aqi"]


# ═════════════════════════════════════════════════════════════════════
# NAVIGATION BAR
# ═════════════════════════════════════════════════════════════════════
_now = _dtmod.datetime.now().strftime("%b %d, %Y  %I:%M %p")
st.markdown(
    f'<div class="nav-bar">'
    f'  <span class="nav-title">Delhi AQI Intelligence</span>'
    f'  <div class="nav-meta">'
    f'    <span><span class="live-dot"></span> LIVE</span>'
    f'    <span>{n_stations} stations</span>'
    f'    <span>{_now}</span>'
    f'  </div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════════════
# HERO BANNER
# ═════════════════════════════════════════════════════════════════════
_grad2 = {"Good": "#00805a", "Satisfactory": "#3a8a3b", "Moderate": "#e6a800",
          "Poor": "#d4700a", "Very Poor": "#9a0028", "Severe": "#500018"}
_clr2 = _grad2.get(aqi_cat, aqi_clr)

st.markdown(
    f'<div class="hero" style="background: linear-gradient(160deg, {aqi_clr}, {_clr2});">'
    f'  <div class="live-badge"><span class="pulse-dot"></span> LIVE MONITORING</div>'
    f'  <p class="aqi-number">{aqi_val}</p>'
    f'  <p class="aqi-label">India NAQI  //  24-Hour Average</p>'
    f'  <div class="hero-tags">'
    f'    <span class="tag">{aqi_cat}</span>'
    f'    <span class="tag">PM2.5 {pm25_avg:.0f} ug/m3</span>'
    f'    <span class="tag">Dominant: {dominant.upper()}</span>'
    f'  </div>'
    f'</div>',
    unsafe_allow_html=True,
)

# Scale strip
scale_html = '<div class="scale-strip">'
for _, lo, hi, color in AQI_BANDS:
    scale_html += f'<div class="seg" style="background:{color}"></div>'
scale_html += '</div><div class="scale-labels">'
for label, lo, hi, _ in AQI_BANDS:
    short = label.split()[0] if len(label) > 8 else label
    scale_html += f'<span>{short} {lo}-{hi}</span>'
scale_html += '</div>'
st.markdown(scale_html, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════
# POLLUTANT CARDS
# ═════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Air Pollutant Concentrations</div>',
            unsafe_allow_html=True)

PARAM_META = {
    "pm25": "PM2.5", "pm10": "PM10",
    "no2": "NO2", "so2": "SO2",
    "co": "CO", "o3": "O3",
}
display_params = [p for p in ["pm25", "pm10", "no2", "so2", "co", "o3"]
                  if p in pollutant_vals]
cols = st.columns(min(len(display_params), 6))
for i, p in enumerate(display_params):
    val, unit = pollutant_vals[p]
    disp_val, disp_unit = convert_for_display(p, val, unit)
    si = sub_idx.get(p, 0)
    cat_p = classify_aqi_value(si)
    clr_p = aqi_color(cat_p)
    name = PARAM_META.get(p, p.upper())
    bar_pct = min(si / 500 * 100, 100)
    delay = i * 0.06
    with cols[i]:
        st.markdown(
            f'<div class="poll-card" style="--accent:{clr_p}; animation-delay:{delay}s">'
            f'  <div class="p-value" style="color:{clr_p}">{disp_val}</div>'
            f'  <div class="p-name">{name}</div>'
            f'  <div class="p-unit">{disp_unit}  //  Sub-index {si}</div>'
            f'  <div class="p-bar-track">'
            f'    <div class="p-bar-fill" style="--bar-w:{bar_pct}%; background:linear-gradient(90deg,{clr_p},{clr_p}bb)"></div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════════════════════════════
# KEY METRICS
# ═════════════════════════════════════════════════════════════════════
cig_per_day = round(pm25_avg / 22, 1) if pm25_avg else 0
cig_weekly = round(cig_per_day * 7, 1)
who_pm25 = WHO_GUIDELINES["pm25"]["limit"]
times_who = round(pm25_avg / who_pm25, 1) if pm25_avg and who_pm25 else "?"

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        f'<div class="kpi-card" style="animation-delay:.1s">'
        f'  <div class="kpi-icon" style="background:linear-gradient(135deg,#f97316,#ea580c)">CIG</div>'
        f'  <div class="kpi-value" style="color:#f97316">{cig_per_day}</div>'
        f'  <div class="kpi-label">Cigarettes/day equivalent</div>'
        f'  <div class="kpi-sub">{cig_weekly}/week breathing Delhi air</div>'
        f'</div>', unsafe_allow_html=True)
with m2:
    st.markdown(
        f'<div class="kpi-card" style="animation-delay:.15s">'
        f'  <div class="kpi-icon" style="background:linear-gradient(135deg,{aqi_clr},{_clr2})">AQI</div>'
        f'  <div class="kpi-value" style="color:{aqi_clr}">{aqi_val}</div>'
        f'  <div class="kpi-label">{aqi_cat}</div>'
        f'  <div class="kpi-sub">India NAQI Standard</div>'
        f'</div>', unsafe_allow_html=True)
with m3:
    _who_clr = "#ef4444" if (isinstance(times_who, (int, float)) and times_who > 3) else "#f97316"
    st.markdown(
        f'<div class="kpi-card" style="animation-delay:.2s">'
        f'  <div class="kpi-icon" style="background:linear-gradient(135deg,{_who_clr},#b91c1c)">WHO</div>'
        f'  <div class="kpi-value" style="color:{_who_clr}">{times_who}x</div>'
        f'  <div class="kpi-label">Above WHO PM2.5 Guideline</div>'
        f'  <div class="kpi-sub">{pm25_avg:.0f} ug/m3 vs {who_pm25} ug/m3 limit</div>'
        f'</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════
# HEALTH ADVISORY
# ═════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Health Advisory</div>',
            unsafe_allow_html=True)

adv = health_advice(aqi_cat)
st.markdown(
    f'<div class="health-card" style="--severity-color:{aqi_clr}">'
    f'  <div class="hc-title" style="color:{aqi_clr}">'
    f'    <span class="severity-dot" style="background:{aqi_clr}"></span>'
    f'    Current Advisory  //  {aqi_cat}'
    f'  </div>'
    f'  <div class="hc-body">{adv}</div>'
    f'</div>', unsafe_allow_html=True)

HEALTH_RISKS = {
    "Poor":      [("Asthma", "Moderate risk. Reduce outdoor exertion.", "#e74c3c"),
                  ("Cardiac", "Mild risk. Avoid strenuous outdoor activity.", "#e74c3c"),
                  ("Allergies", "Elevated. Particulate matter worsens symptoms.", "#e74c3c"),
                  ("Respiratory", "Reduced immunity. Take precautions.", "#e74c3c")],
    "Very Poor": [("Asthma", "High risk. Stay indoors, use air purifier.", "#c0392b"),
                  ("Cardiac", "Serious risk. PM2.5 increases cardiac events.", "#c0392b"),
                  ("Allergies", "Severe. Seal windows, use N95 mask.", "#c0392b"),
                  ("COPD", "Critical. Avoid all outdoor exposure.", "#c0392b")],
    "Severe":    [("Asthma", "Emergency level. Indoor only, purifier on max.", "#7E0023"),
                  ("Cardiac", "Emergency level. Avoid exertion, monitor BP.", "#7E0023"),
                  ("Allergies", "Emergency. Complete indoor isolation.", "#7E0023"),
                  ("COPD", "Life-threatening. Seek medical help if symptomatic.", "#7E0023")],
    "Moderate":  [("Asthma", "Mild risk. Sensitive individuals should take care.", "#e67e22"),
                  ("Cardiac", "Low risk. Normal activity, monitor symptoms.", "#e67e22")],
}
risks = HEALTH_RISKS.get(aqi_cat, [])
if risks:
    rcols = st.columns(min(len(risks), 4))
    for i, (title, body, clr) in enumerate(risks):
        with rcols[i]:
            st.markdown(
                f'<div class="health-card" style="--severity-color:{clr}">'
                f'  <div class="hc-title">'
                f'    <span class="severity-dot" style="background:{clr}"></span>'
                f'    {title}'
                f'  </div>'
                f'  <div class="hc-body">{body}</div>'
                f'</div>', unsafe_allow_html=True)

# Recommended Actions
sol_data = {
    "Good":        [("Enjoy outdoors", "#27ae60")],
    "Satisfactory": [("Normal activity", "#27ae60"), ("Mask optional", "#94a3b8")],
    "Moderate":    [("Use N95 Mask", "#e67e22"), ("Limit outdoor time", "#e67e22")],
    "Poor":        [("N95 Required", "#e74c3c"), ("Stay Indoor", "#e74c3c"),
                    ("Air Purifier On", "#e74c3c")],
    "Very Poor":   [("N95 Required", "#c0392b"), ("Stay Indoor", "#c0392b"),
                    ("Air Purifier Max", "#c0392b"), ("Seal Windows", "#c0392b")],
    "Severe":      [("N95 Required", "#7E0023"), ("Seal Indoors", "#7E0023"),
                    ("Purifier Max", "#7E0023"), ("Seek Medical Help", "#7E0023")],
}
pills = sol_data.get(aqi_cat, [])
if pills:
    st.markdown("**Recommended Actions:**  " + "  ".join(
        f'<span class="action-pill">'
        f'<span class="pill-dot" style="background:{c}"></span>{txt}</span>'
        for txt, c in pills
    ), unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════
# DYNAMIC INSIGHTS
# ═════════════════════════════════════════════════════════════════════
if _insights:
    st.markdown('<div class="section-header">Live Insights</div>',
                unsafe_allow_html=True)
    n_cols = min(len(_insights), 3)
    for row_start in range(0, len(_insights), n_cols):
        row_items = _insights[row_start:row_start + n_cols]
        cols_ins = st.columns(len(row_items))
        for ci, ins in enumerate(row_items):
            sev_color = {"critical": "#ef4444", "warning": "#f59e0b",
                         "info": "#3b82f6"}.get(ins.severity, "#6b7280")
            with cols_ins[ci]:
                st.markdown(
                    f'<div class="health-card" style="--severity-color:{sev_color}">'
                    f'  <div class="hc-title" style="color:{sev_color}">'
                    f'    <span class="severity-dot" style="background:{sev_color}"></span>'
                    f'    {ins.title}'
                    f'  </div>'
                    f'  <div class="hc-body">{ins.body}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ═════════════════════════════════════════════════════════════════════
# AQI FORECAST (ML-based, 6-hour recursive)
# ═════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">6-Hour AQI Forecast</div>',
            unsafe_allow_html=True)

if _aqi_forecast and _aqi_forecast.forecasted_aqi:
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        _trend_clr = {"worsening": "#ef4444", "improving": "#22c55e",
                      "stable": "#3b82f6"}.get(_aqi_forecast.overall_trend, "#6b7280")
        _trend_lbl = {"worsening": "UP", "improving": "DN",
                      "stable": "="}.get(_aqi_forecast.overall_trend, "=")
        st.markdown(
            f'<div class="kpi-card">'
            f'  <div class="kpi-icon" style="background:{_trend_clr}">{_trend_lbl}</div>'
            f'  <div class="kpi-value" style="color:{_trend_clr}">'
            f'    {_aqi_forecast.overall_trend.title()}'
            f'  </div>'
            f'  <div class="kpi-label">Overall Trend</div>'
            f'  <div class="kpi-sub">Next 6 hours</div>'
            f'</div>', unsafe_allow_html=True)
    with fc2:
        _next = _aqi_forecast.forecasted_aqi[0]
        _next_clr = aqi_color(classify_aqi_value(_next["aqi"]))
        st.markdown(
            f'<div class="kpi-card">'
            f'  <div class="kpi-icon" style="background:{_next_clr}">+1h</div>'
            f'  <div class="kpi-value" style="color:{_next_clr}">'
            f'    {_next["aqi"]}'
            f'  </div>'
            f'  <div class="kpi-label">AQI in 1 Hour</div>'
            f'  <div class="kpi-sub">{_next["category"]}  /  '
            f'    Range: {_next["lower"]}-{_next["upper"]}</div>'
            f'</div>', unsafe_allow_html=True)
    with fc3:
        _conf_clr = {"high": "#22c55e", "medium": "#f59e0b",
                     "low": "#ef4444"}.get(_aqi_forecast.confidence_level, "#6b7280")
        st.markdown(
            f'<div class="kpi-card">'
            f'  <div class="kpi-icon" style="background:{_conf_clr}">C</div>'
            f'  <div class="kpi-value" style="color:{_conf_clr}">'
            f'    {_aqi_forecast.confidence_level.title()}'
            f'  </div>'
            f'  <div class="kpi-label">Forecast Confidence</div>'
            f'  <div class="kpi-sub">Based on data quality & model: {_aqi_forecast.model_name}</div>'
            f'</div>', unsafe_allow_html=True)

    fig_fc = forecast_chart(_aqi_forecast, current_aqi=aqi_val)
    if fig_fc:
        st.plotly_chart(fig_fc, width="stretch", key="forecast_main")

    with st.expander("Forecast Details", expanded=False):
        fc_data = []
        for fp in _aqi_forecast.forecasted_aqi:
            fc_data.append({
                "Hour": f"+{fp['hour_offset']}h",
                "AQI": fp["aqi"],
                "Category": fp["category"],
                "Lower": fp["lower"],
                "Upper": fp["upper"],
            })
        st.dataframe(pd.DataFrame(fc_data), hide_index=True, width="stretch")

    if _aqi_forecast.summary:
        st.info(f"**Forecast Summary:** {_aqi_forecast.summary}")
else:
    st.info("Insufficient hourly data for ML-based forecasting. "
            "Requires at least 6 hours of recent data.")

# ═════════════════════════════════════════════════════════════════════
# STATION-WISE VIEW
# ═════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Station-Wise Monitoring</div>',
            unsafe_allow_html=True)

tab_cmp, tab_hm, tab_det = st.tabs([
    "Station Comparison", "Station Heatmap", "Station Detail"
])

with tab_cmp:
    compare_param = st.selectbox(
        "Select pollutant to compare across stations",
        ["pm25", "pm10", "no2", "so2", "o3", "co"],
        format_func=lambda p: {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2",
                               "so2": "SO2", "co": "CO", "o3": "O3"}.get(p, p),
        key="compare_param",
    )
    fig_cmp = station_comparison_chart(df_latest, compare_param)
    if fig_cmp:
        st.plotly_chart(fig_cmp, width="stretch", key="station_compare")
    else:
        st.info("Not enough station data for comparison.")

with tab_hm:
    fig_hm = station_aqi_heatmap(df_latest)
    if fig_hm:
        st.plotly_chart(fig_hm, width="stretch", key="station_heatmap")
        st.caption("Color intensity shows sub-index severity. "
                   "Values: concentration (Sub-Index).")
    else:
        st.info("Need data from at least 2 stations for heatmap.")

with tab_det:
    if stations_list:
        station_names = {s["name"]: s["id"] for s in stations_list}
        selected_station = st.selectbox(
            "Select a station to explore",
            list(station_names.keys()),
            key="station_select",
        )
        if selected_station:
            station_df = df_latest[df_latest["location"] == selected_station]
            if station_df.empty:
                sid = station_names[selected_station]
                station_df = get_station_latest(sid, api_key=openaq_key)
            if not station_df.empty:
                fig_sd = station_detail_chart(station_df, selected_station)
                if fig_sd:
                    st.plotly_chart(fig_sd, width="stretch",
                                   key="station_detail")
                _s_vals = {}
                for param in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
                    _s_sub = station_df[station_df["parameter"] == param]
                    if not _s_sub.empty:
                        _s_vals[param] = (
                            _s_sub["value"].mean(),
                            _s_sub["unit"].iloc[0]
                            if "unit" in _s_sub.columns else ""
                        )
                if _s_vals:
                    _s_aqi = compute_aqi(_s_vals)
                    _s_clr = aqi_color(_s_aqi["category"])
                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        st.markdown(
                            f'<div class="kpi-card">'
                            f'  <div class="kpi-icon" style="background:{_s_clr}">AQI</div>'
                            f'  <div class="kpi-value" style="color:{_s_clr}">{_s_aqi["aqi"]}</div>'
                            f'  <div class="kpi-label">{_s_aqi["category"]}</div>'
                            f'  <div class="kpi-sub">Station AQI</div>'
                            f'</div>', unsafe_allow_html=True)
                    with sc2:
                        _s_dom = _s_aqi.get("dominant", "?")
                        st.markdown(
                            f'<div class="kpi-card">'
                            f'  <div class="kpi-icon" style="background:{_s_clr}">DOM</div>'
                            f'  <div class="kpi-value" style="color:{_s_clr}">{_s_dom.upper()}</div>'
                            f'  <div class="kpi-label">Dominant Pollutant</div>'
                            f'  <div class="kpi-sub">At this station</div>'
                            f'</div>', unsafe_allow_html=True)
                    with sc3:
                        _diff = _s_aqi["aqi"] - aqi_val
                        _diff_str = f"+{_diff}" if _diff > 0 else str(_diff)
                        _diff_clr = ("#ef4444" if _diff > 20
                                     else "#22c55e" if _diff < -20
                                     else "#f59e0b")
                        _diff_arrow = ("^" if _diff > 0
                                       else "v" if _diff < 0 else "=")
                        st.markdown(
                            f'<div class="kpi-card">'
                            f'  <div class="kpi-icon" style="background:{_diff_clr}">{_diff_arrow}</div>'
                            f'  <div class="kpi-value" style="color:{_diff_clr}">{_diff_str}</div>'
                            f'  <div class="kpi-label">vs City Average</div>'
                            f'  <div class="kpi-sub">City AQI: {aqi_val}</div>'
                            f'</div>', unsafe_allow_html=True)
            else:
                st.warning(f"No data available for {selected_station}.")
    else:
        st.info("No station list available. Ensure OpenAQ API key is configured.")

# ═════════════════════════════════════════════════════════════════════
# DETAILED ANALYSIS CHARTS
# ═════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Detailed Analysis</div>',
            unsafe_allow_html=True)

tab_overview, tab_who, tab_trend, tab_data = st.tabs([
    "Overview", "WHO Comparison", "Trends", "Raw Data"
])

with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        fig = aqi_gauge(aqi_val, aqi_cat)
        st.plotly_chart(fig, width="stretch", key="gauge")
    with c2:
        fig = pollutant_radar(sub_idx)
        if fig:
            st.plotly_chart(fig, width="stretch", key="radar")
        else:
            fig = sub_index_chart(sub_idx)
            if fig:
                st.plotly_chart(fig, width="stretch", key="sub_bar_alt")
    fig = sub_index_chart(sub_idx)
    if fig:
        st.plotly_chart(fig, width="stretch", key="sub_full")
        st.caption("Overall AQI = highest sub-index. "
                   "India NAQI breakpoint interpolation.")

with tab_who:
    fig = pollutant_vs_who(df_latest)
    if fig:
        st.plotly_chart(fig, width="stretch", key="who")
        st.caption("Values exceeding WHO 2021 guidelines highlighted in red.")
    else:
        st.info("Insufficient data for WHO comparison.")

with tab_trend:
    tc1, tc2 = st.columns(2)
    with tc1:
        fig = timeseries_plot(df_hist, "pm25")
        if fig:
            st.plotly_chart(fig, width="stretch", key="ts_pm25")
        else:
            st.info("No PM2.5 history available.")
    with tc2:
        fig = timeseries_plot(df_hist, "pm10")
        if fig:
            st.plotly_chart(fig, width="stretch", key="ts_pm10")
        else:
            st.info("No PM10 history available.")
    with st.expander("More pollutant trends"):
        for p in ("no2", "o3", "so2", "co"):
            fig = timeseries_plot(df_hist, p)
            if fig:
                st.plotly_chart(fig, width="stretch", key=f"ts_{p}")

with tab_data:
    if "parameter" in df_latest.columns:
        summary = (df_latest.groupby("parameter")["value"]
                   .agg(["mean", "min", "max", "count"]).round(1))
        summary.columns = ["Average", "Min", "Max", "Readings"]
        st.dataframe(summary, width="stretch")

with st.expander("AQI Scale Reference"):
    fig = aqi_scale_bar()
    st.plotly_chart(fig, width="stretch", key="scale")
    st.markdown("""
| AQI | Category | Health Guidance |
|-----|----------|-----------------|
| 0-50 | **Good** | Safe for outdoor activities |
| 51-100 | **Satisfactory** | Sensitive groups limit prolonged exertion |
| 101-200 | **Moderate** | Reduce heavy outdoor exercise |
| 201-300 | **Poor** | Everyone should reduce outdoor activity |
| 301-400 | **Very Poor** | Avoid outdoor activity, use air purifier |
| 401-500 | **Severe** | Stay indoors, N95 mask, seek medical help |
""")

# ═════════════════════════════════════════════════════════════════════
# AI CHAT — Expert Delhi AQI Analyst
# ═════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">AI Air Quality Analyst</div>',
            unsafe_allow_html=True)
st.caption("Expert analysis powered by Llama 3.3 70B + Hybrid RAG. "
           "Ask about Delhi air quality, health impacts, pollution science, "
           "CPCB standards, or any environmental topic.")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "conv_memory" not in st.session_state:
    from rag.memory import ConversationMemory
    st.session_state.conv_memory = ConversationMemory(
        max_tokens=6000, keep_recent=6, groq_api_key=groq_key,
    )

# Helper: RAG retrieval
def _retrieve_rag(question: str) -> tuple[list, dict]:
    try:
        from rag.retriever import Retriever
        from rag.query_classifier import classify_query
        intent = classify_query(question)
        config = intent.get("config", {})
        if not config.get("needs_rag", True) and intent["confidence"] > 0.5:
            return [], intent
        ret = Retriever("embeddings/vector_store")
        top_k = config.get("top_k", 5)
        results = ret.retrieve(question, top_k=top_k,
                               use_reranker=True, intent_config=config)
        ret.save_cache()
        return results, intent
    except Exception:
        return [], {"intent": "general", "confidence": 0.3, "config": {}}

def _execute_tool_call(question: str) -> dict | None:
    try:
        from rag.tool_calling import auto_tool_call
        return auto_tool_call(question, snapshot=_snapshot)
    except Exception:
        return None

def _rule_answer():
    cat = aqi_r["category"]
    lines = [f"## Air Quality: AQI {aqi_r['aqi']} -- {cat}\n"]
    lines.append(f"**Dominant pollutant**: {aqi_r['dominant'].upper()}\n")
    lines.append("### Sub-Indices")
    for p, si in sorted(aqi_r["sub_indices"].items(), key=lambda x: -x[1]):
        label = WHO_GUIDELINES.get(p, {}).get("label", p.upper())
        lines.append(f"- **{label}**: {si} ({classify_aqi_value(si)})")
    lines.append(f"\n### Health Advisory\n{health_advice(cat)}")
    if pm25_avg:
        lines.append(
            f"\n> Breathing this air is equivalent to "
            f"**{round(pm25_avg / 22, 1)} cigarettes/day**"
        )
    lines.append("\n---\n*Add your free Groq API key in the sidebar "
                 "for detailed AI analysis.*")
    return "\n".join(lines)

# Render chat history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("confidence_badge"):
            st.caption(msg["confidence_badge"])

# Chat input
if prompt := st.chat_input(
    "Ask about Delhi air quality, health impacts, pollution science..."
):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            retrieved, intent = _retrieve_rag(prompt)
            intent_type = intent.get("intent", "general")
            tool_result = _execute_tool_call(prompt)

            from rag.llm_pipeline import LLMPipeline, SYSTEM_MESSAGE
            from rag.confidence import format_confidence_badge

            llm = LLMPipeline(groq_api_key=groq_key)

            context_block = "\n".join(
                f"  - {k}: {v}" for k, v in sorted(_snapshot.items())
            )
            insights_block = ""
            if _insights:
                insights_block = "\n\n## Current Insights\n"
                for ins in _insights[:4]:
                    insights_block += f"  - {ins.title}: {ins.body[:120]}\n"

            forecast_block = ""
            if _aqi_forecast and _aqi_forecast.forecasted_aqi:
                forecast_block = (
                    "\n\n## 6-Hour AQI Forecast (ML Model)\n"
                    f"  - Trend: {_aqi_forecast.overall_trend} "
                    f"({_aqi_forecast.confidence_level} confidence)\n"
                )
                for fp in _aqi_forecast.forecasted_aqi:
                    forecast_block += (
                        f"  - +{fp['hour_offset']}h: AQI {fp['aqi']} "
                        f"({fp['category']}, range {fp['lower']}-{fp['upper']})\n"
                    )

            system_with_context = (
                SYSTEM_MESSAGE + "\n\n"
                "## Current Live AQI Data (Delhi)\n" + context_block
                + forecast_block + insights_block
            )

            groq_messages = [
                {"role": "system", "content": system_with_context}
            ]
            for msg in st.session_state.chat_messages[:-1]:
                groq_messages.append({
                    "role": msg["role"], "content": msg["content"]
                })

            if retrieved or tool_result:
                current_msg = build_prompt(
                    prompt, retrieved, _snapshot,
                    intent=intent, tool_result=tool_result,
                )
            else:
                current_msg = prompt

            groq_messages.append({"role": "user", "content": current_msg})
            groq_messages = st.session_state.conv_memory.prepare_messages(
                groq_messages
            )

            result = llm.chat_with_guard(
                groq_messages,
                query=prompt,
                retrieved_chunks=retrieved,
                intent=intent_type,
                max_length=1500,
            )

            answer = result["answer"]
            confidence = result["confidence"]
            hall_check = result["hallucination_check"]
            regenerated = result["regenerated"]

            is_error = (answer.startswith("[Groq")
                        or answer.startswith("[LLM")
                        or answer.startswith("[API"))
            if is_error:
                if "API key" in answer or "401" in answer:
                    st.warning(
                        "Groq API key missing or invalid. "
                        "Get a free key at https://console.groq.com"
                    )
                answer = _rule_answer()
                confidence = {
                    "score": 0.3, "grade": "low",
                    "breakdown": {}, "explanation": "Fallback response",
                }

        st.markdown(answer)

        badge = format_confidence_badge(confidence)
        st.caption(badge)

        with st.expander("Response Details"):
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.markdown(
                    f"**Intent:** {intent_type} "
                    f"({intent.get('confidence', 0):.0%})"
                )
                st.markdown(f"**Retrieved:** {len(retrieved)} chunks")
                if regenerated:
                    st.markdown(
                        "*Response was regenerated "
                        "(hallucination guard triggered)*"
                    )
            with dcol2:
                bd = confidence.get("breakdown", {})
                for k, v in bd.items():
                    st.markdown(
                        f"**{k.replace('_', ' ').title()}:** {v:.0%}"
                    )
                st.markdown(
                    f"**Hallucination risk:** "
                    f"{hall_check.get('hallucination_risk', '?')}"
                )

        if retrieved:
            with st.expander("Sources"):
                for i, ref in enumerate(retrieved, 1):
                    score = ref.get("final_score",
                                    ref.get("rerank_score", "?"))
                    score_str = (f" -- relevance: {score:.2f}"
                                 if isinstance(score, float) else "")
                    topic = ref.get("topic", "")
                    topic_str = f" [{topic}]" if topic else ""
                    st.caption(
                        f"**[Ref {i}]** {ref.get('doc_name', '?')} "
                        f"(p.{ref.get('page', '?')}) "
                        f"{topic_str}{score_str}"
                    )

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": answer,
        "confidence_badge": badge if not is_error else None,
    })

# Starter suggestions
if not st.session_state.chat_messages:
    st.markdown("**Try asking:**")
    suggestions = [
        "How does Delhi's AQI compare to other megacities?",
        "Is it safe to exercise outdoors right now?",
        "What are the main sources of Delhi's pollution?",
        "Explain the health impact of current PM2.5 levels",
    ]
    scols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        with scols[i]:
            if st.button(s, key=f"sug_{i}", width="stretch"):
                st.session_state.chat_messages.append(
                    {"role": "user", "content": s}
                )
                st.rerun()

# ─── Footer ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#94a3b8; font-size:.75rem; '
    'padding:8px 0;">'
    'Data: <b>OpenAQ</b>  //  Standard: <b>India NAQI</b>  //  '
    'WHO 2021 Guidelines  //  '
    'AI: <b>Llama 3.3 70B</b> via Groq  //  '
    'Built with Streamlit + RAG + Plotly'
    '</div>',
    unsafe_allow_html=True,
)
