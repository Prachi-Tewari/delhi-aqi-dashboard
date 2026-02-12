"""OpenAQ v3 API client for fetching real-time AQI data.

OpenAQ v3 requires a free API key (X-API-Key header).
Get one at: https://explore.openaq.org  ->  Sign up  ->  API Keys

Key insight from v3:
  /locations/{id}/latest returns rows with sensorsId but NO parameter name.
  We must build a sensorId -> param mapping from the /locations response.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone


# ---------- constants ----------
BASE = "https://api.openaq.org/v3"

# Delhi (India Gate) coordinates
DELHI_LAT = 28.6139
DELHI_LON = 77.2090

POLLUTANTS = {"pm25", "pm10", "no2", "so2", "co", "o3"}


# ---------- low-level ----------

def _key(api_key: str = "") -> str:
    return api_key or os.environ.get("OPENAQ_API_KEY", "")


def _headers(api_key: str = "") -> dict:
    k = _key(api_key)
    h = {"Accept": "application/json"}
    if k:
        h["X-API-Key"] = k
    return h


def _get(endpoint: str, params: dict = None, api_key: str = "",
         timeout: int = 25) -> dict:
    url = f"{BASE}/{endpoint.lstrip('/')}"
    r = requests.get(url, params=params or {}, headers=_headers(api_key),
                     timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------- locations ----------

def list_locations(city: str = "Delhi", country_iso: str = "IN",
                   lat: float = None, lon: float = None,
                   radius: int = 25000, limit: int = 100,
                   api_key: str = "") -> list:
    """Return monitoring locations near <lat,lon> (default: Delhi).

    Each item includes a ``sensors`` list with ``{id, parameter, units}``.
    Only locations with data from the last 30 days are included.
    """
    api_key = _key(api_key)
    if not api_key:
        return []

    lat = lat or DELHI_LAT
    lon = lon or DELHI_LON

    params = {
        "coordinates": f"{lat},{lon}",
        "radius": radius,
        "limit": limit,
    }
    if country_iso:
        params["iso"] = country_iso

    try:
        data = _get("locations", params, api_key=api_key)
    except Exception:
        return []

    # Only keep locations active in the last 30 days
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    locations = []
    for loc in data.get("results", []):
        dl = loc.get("datetimeLast") or {}
        last_str = dl.get("utc") if isinstance(dl, dict) else None
        if last_str:
            try:
                last_dt = datetime.fromisoformat(last_str.replace("Z", "+00:00"))
                if last_dt < cutoff:
                    continue
            except Exception:
                pass
        else:
            continue  # skip locations with no last-update

        sensors_raw = loc.get("sensors") or []
        sensors = []
        for s in sensors_raw:
            if not isinstance(s, dict):
                continue
            p = s.get("parameter") or {}
            sensors.append({
                "id": s.get("id"),
                "parameter": p.get("name", ""),
                "units": p.get("units", ""),
            })

        locations.append({
            "id": loc.get("id"),
            "name": loc.get("name", ""),
            "locality": loc.get("locality", ""),
            "country": (loc.get("country") or {}).get("code", ""),
            "coordinates": loc.get("coordinates", {}),
            "sensors": sensors,
            "parameters": list({s["parameter"] for s in sensors if s["parameter"]}),
            "last_updated": last_str,
        })
    return locations


# ---------- latest ----------

def get_latest_city_measurements(
    city: str = "Delhi",
    country_iso: str = "IN",
    parameters: set = None,
    limit: int = 100,
    api_key: str = "",
) -> pd.DataFrame:
    """Most-recent value from each sensor at nearby active locations.

    /locations/{id}/latest returns {sensorsId, value, datetime} but
    *no* parameter name, so we build a sensorId->param map from the
    location's sensors list.
    """
    if parameters is None:
        parameters = POLLUTANTS

    api_key = _key(api_key)
    if not api_key:
        return _demo_latest()

    locations = list_locations(city, country_iso, api_key=api_key, limit=limit)
    if not locations:
        return _demo_latest()

    rows = []
    for loc in locations[:15]:
        loc_id = loc.get("id")
        if not loc_id:
            continue

        # Build sensorId -> {parameter, units} map
        sensor_map = {}
        for s in loc.get("sensors", []):
            sid = s.get("id")
            if sid:
                sensor_map[sid] = {
                    "parameter": s.get("parameter", ""),
                    "units": s.get("units", ""),
                }

        try:
            data = _get(f"locations/{loc_id}/latest",
                        {"limit": 100}, api_key=api_key)
        except Exception:
            continue

        for item in data.get("results", []):
            sid = item.get("sensorsId")
            info = sensor_map.get(sid, {})
            param = info.get("parameter", "")
            if param not in parameters:
                continue

            dt = item.get("datetime") or {}
            rows.append({
                "parameter": param,
                "value": item.get("value"),
                "unit": info.get("units", ""),
                "date_utc": dt.get("utc") if isinstance(dt, dict) else dt,
                "location": loc.get("name", ""),
            })

    if not rows:
        return _demo_latest()

    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce", utc=True)

    # Keep only the most recent reading per parameter+location
    # (locations may have old + new sensors for the same pollutant)
    df = (
        df.sort_values("date_utc", ascending=False)
        .groupby(["parameter", "location"], as_index=False)
        .first()
    )
    return df


# ---------- historical ----------

def get_historical_city(
    city: str = "Delhi",
    country_iso: str = "IN",
    days: int = 7,
    parameters: set = None,
    limit: int = 3000,
    api_key: str = "",
) -> pd.DataFrame:
    """Daily-aggregated data from /sensors/{id}/days for each sensor."""
    if parameters is None:
        parameters = POLLUTANTS

    api_key = _key(api_key)
    if not api_key:
        return _demo_historical(days=days)

    locations = list_locations(city, country_iso, api_key=api_key, limit=50)
    if not locations:
        return _demo_historical(days=days)

    date_from = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    date_to = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    rows = []
    api_calls = 0
    max_calls = 40

    for loc in locations[:10]:
        for sensor in loc.get("sensors", []):
            if api_calls >= max_calls:
                break
            param = sensor.get("parameter", "")
            sid = sensor.get("id")
            if param not in parameters or not sid:
                continue
            api_calls += 1
            try:
                data = _get(
                    f"sensors/{sid}/days",
                    {"date_from": date_from, "date_to": date_to, "limit": 100},
                    api_key=api_key, timeout=30,
                )
            except Exception:
                continue
            for rec in data.get("results", []):
                val = rec.get("value")
                if val is None:
                    continue
                period = rec.get("period") or {}
                dt_from = period.get("datetimeFrom") or {}
                date_utc = dt_from.get("utc") if isinstance(dt_from, dict) else dt_from
                rec_param = (rec.get("parameter") or {}).get("name", param)
                rec_unit = (rec.get("parameter") or {}).get("units", sensor.get("units", ""))
                rows.append({
                    "parameter": rec_param,
                    "value": val,
                    "unit": rec_unit,
                    "date_utc": date_utc,
                    "location": loc.get("name", ""),
                })

    if not rows:
        return _demo_historical(days=days)

    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")
    return df


# ---------- station-level latest ----------

def get_station_latest(
    location_id: int,
    parameters: set = None,
    api_key: str = "",
) -> pd.DataFrame:
    """Latest readings for a single station by its location ID."""
    if parameters is None:
        parameters = POLLUTANTS
    api_key = _key(api_key)
    if not api_key:
        return pd.DataFrame()

    # Get location details for sensor map
    try:
        loc_data = _get(f"locations/{location_id}", api_key=api_key)
    except Exception:
        return pd.DataFrame()

    loc = loc_data.get("results", [{}])
    if isinstance(loc, list):
        loc = loc[0] if loc else {}

    sensor_map = {}
    for s in loc.get("sensors", []):
        sid = s.get("id")
        p = s.get("parameter") or {}
        if sid:
            sensor_map[sid] = {
                "parameter": p.get("name", ""),
                "units": p.get("units", ""),
            }

    try:
        data = _get(f"locations/{location_id}/latest",
                    {"limit": 100}, api_key=api_key)
    except Exception:
        return pd.DataFrame()

    rows = []
    for item in data.get("results", []):
        sid = item.get("sensorsId")
        info = sensor_map.get(sid, {})
        param = info.get("parameter", "")
        if param not in parameters:
            continue
        dt = item.get("datetime") or {}
        rows.append({
            "parameter": param,
            "value": item.get("value"),
            "unit": info.get("units", ""),
            "date_utc": dt.get("utc") if isinstance(dt, dict) else dt,
            "location": loc.get("name", ""),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce", utc=True)
    return df


# ---------- hourly data (for forecasting) ----------

def get_hourly_data(
    city: str = "Delhi",
    country_iso: str = "IN",
    hours: int = 24,
    parameters: set = None,
    api_key: str = "",
    current_values: dict = None,
) -> pd.DataFrame:
    """Fetch recent hourly data from sensors using /sensors/{id}/hours endpoint.

    Args:
        current_values: Optional dict of {param: (value, unit)} from live data.
                        Used to anchor fallback synthetic data so forecasts
                        are consistent with current dashboard readings.

    Returns a DataFrame with columns:
        parameter, value, unit, date_utc, location
    Each row is one hour's average for one sensor.
    """
    if parameters is None:
        parameters = POLLUTANTS
    api_key = _key(api_key)
    if not api_key:
        return _demo_hourly(hours=hours, current_values=current_values)

    locations = list_locations(city, country_iso, api_key=api_key, limit=50)
    if not locations:
        return _demo_hourly(hours=hours, current_values=current_values)

    date_from = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    date_to = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = []
    api_calls = 0
    max_calls = 30  # rate-limit guard

    for loc in locations[:8]:
        for sensor in loc.get("sensors", []):
            if api_calls >= max_calls:
                break
            param = sensor.get("parameter", "")
            sid = sensor.get("id")
            if param not in parameters or not sid:
                continue
            api_calls += 1
            try:
                data = _get(
                    f"sensors/{sid}/hours",
                    {"date_from": date_from, "date_to": date_to, "limit": 200},
                    api_key=api_key, timeout=30,
                )
            except Exception:
                continue
            for rec in data.get("results", []):
                val = rec.get("value")
                if val is None:
                    continue
                period = rec.get("period") or {}
                dt_from = period.get("datetimeFrom") or {}
                ts = dt_from.get("utc") if isinstance(dt_from, dict) else dt_from
                rec_param = (rec.get("parameter") or {}).get("name", param)
                rec_unit = (rec.get("parameter") or {}).get("units",
                            sensor.get("units", ""))
                rows.append({
                    "parameter": rec_param,
                    "value": val,
                    "unit": rec_unit,
                    "date_utc": ts,
                    "location": loc.get("name", ""),
                })

    if not rows:
        return _demo_hourly(hours=hours, current_values=current_values)

    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce", utc=True)
    # Average across stations per hour per parameter
    df["hour"] = df["date_utc"].dt.floor("h")
    agg = (df.groupby(["hour", "parameter"])
           .agg(value=("value", "mean"), unit=("unit", "first"),
                location=("location", lambda x: f"{x.nunique()} stations"))
           .reset_index()
           .rename(columns={"hour": "date_utc"}))
    return agg.sort_values(["parameter", "date_utc"]).reset_index(drop=True)


def list_city_stations(
    city: str = "Delhi",
    country_iso: str = "IN",
    api_key: str = "",
) -> list[dict]:
    """Return a simplified list of stations for UI selectors.

    Each item: {id, name, locality, parameters, coordinates, last_updated}
    """
    locations = list_locations(city, country_iso, api_key=api_key, limit=100)
    stations = []
    for loc in locations:
        stations.append({
            "id": loc["id"],
            "name": loc.get("name", f"Station {loc['id']}"),
            "locality": loc.get("locality", ""),
            "parameters": loc.get("parameters", []),
            "lat": (loc.get("coordinates") or {}).get("latitude"),
            "lon": (loc.get("coordinates") or {}).get("longitude"),
            "last_updated": loc.get("last_updated", ""),
        })
    # Sort by name
    stations.sort(key=lambda s: s["name"])
    return stations


# ---------- demo / fallback ----------

def _demo_hourly(hours: int = 24, current_values: dict = None) -> pd.DataFrame:
    """Generate realistic hourly data anchored on actual live readings.

    If ``current_values`` is provided (dict of {param: (value, unit)}),
    the synthetic series ends at those values so that the forecast is
    consistent with the rest of the dashboard.
    """
    rng = np.random.default_rng(99)
    dates = pd.date_range(
        end=pd.Timestamp.now("UTC").floor("h"),
        periods=hours, freq="h",
    )

    # Use live values if provided; otherwise fall back to Delhi-typical
    default_base = {"pm25": 150, "pm10": 250, "no2": 40,
                    "so2": 12, "co": 1500, "o3": 30}
    if current_values:
        base = {}
        for param, default_val in default_base.items():
            if param in current_values:
                v, _ = current_values[param]
                base[param] = max(float(v), 1.0)
            else:
                base[param] = default_val
    else:
        base = default_base

    rows = []
    for param, b in base.items():
        # Diurnal pattern: peaks at morning and evening rush hours
        hour_of_day = np.array([d.hour for d in dates])
        local_hour = (hour_of_day + 5) % 24  # rough IST
        diurnal = b * 0.12 * (
            np.exp(-0.5 * ((local_hour - 8) / 2.5) ** 2) +
            np.exp(-0.5 * ((local_hour - 20) / 2.5) ** 2)
        )
        # Small noise (±8% of base)
        noise = rng.normal(0, b * 0.08, size=hours)
        # Gentle trend so the last value ≈ base (the live value)
        values_raw = b + noise + diurnal
        # Shift so that the LAST value equals the live reading
        shift = b - values_raw[-1]
        values = np.maximum(values_raw + shift, 1)

        for dt, val in zip(dates, values):
            rows.append({
                "parameter": param, "value": round(float(val), 1),
                "unit": "µg/m³", "date_utc": dt,
                "location": "synthetic",
            })
    return pd.DataFrame(rows)


def _demo_latest() -> pd.DataFrame:
    now = pd.Timestamp.now("UTC")
    return pd.DataFrame([
        {"parameter": "pm25", "value": 185.0, "unit": "µg/m³",
         "date_utc": now, "location": "Delhi Demo Station"},
        {"parameter": "pm10", "value": 290.0, "unit": "µg/m³",
         "date_utc": now, "location": "Delhi Demo Station"},
        {"parameter": "no2",  "value": 48.0,  "unit": "µg/m³",
         "date_utc": now, "location": "Delhi Demo Station"},
        {"parameter": "so2",  "value": 15.0,  "unit": "µg/m³",
         "date_utc": now, "location": "Delhi Demo Station"},
        {"parameter": "co",   "value": 1800.0, "unit": "µg/m³",
         "date_utc": now, "location": "Delhi Demo Station"},
        {"parameter": "o3",   "value": 32.0,  "unit": "µg/m³",
         "date_utc": now, "location": "Delhi Demo Station"},
    ])


def _demo_historical(days: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    hours = days * 24
    dates = pd.date_range(end=pd.Timestamp.now("UTC"), periods=hours, freq="h")
    base = {"pm25": 150, "pm10": 250, "no2": 40,
            "so2": 12, "co": 1500, "o3": 30}
    rows = []
    for param, b in base.items():
        noise = rng.normal(0, b * 0.2, size=hours)
        diurnal = b * 0.15 * np.sin(np.linspace(0, days * 2 * np.pi, hours))
        values = np.maximum(b + noise + diurnal, 0)
        for dt, val in zip(dates, values):
            rows.append({
                "parameter": param, "value": round(float(val), 1),
                "unit": "µg/m³", "date_utc": dt,
                "location": "Delhi Demo Station",
            })
    return pd.DataFrame(rows)


# ---------- CLI test ----------

if __name__ == "__main__":
    key = _key()
    if not key:
        print("No OPENAQ_API_KEY set — showing demo data.")
        print(_demo_latest().to_string(index=False))
    else:
        print("Listing active Delhi locations ...")
        locs = list_locations("Delhi", api_key=key)
        for loc in locs[:5]:
            print(f"  {loc['name']}  (id={loc['id']}, "
                  f"params={loc['parameters']}, last={loc['last_updated']})")

        print("\nFetching latest ...")
        df = get_latest_city_measurements("Delhi", api_key=key)
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("  (no data)")
