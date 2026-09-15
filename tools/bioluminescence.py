"""
Noctiluca scintillans (sea sparkle) bioluminescence likelihood forecast.

Uses SST, wind, waves, season, and moon illumination. Returns a 0–100
score with a plain-language label.

Sources (all already used elsewhere in ORCA):
  - SST, waves : Copernicus Marine (via copernicus_service)
  - wind       : Open-Meteo (via weather_service)
  - moon       : local moon_service
  - season     : calendar
"""
from __future__ import annotations

from datetime import datetime

from tools.copernicus_service import get_copernicus_marine_snapshot
from tools.weather_service import get_weather_conditions
from tools.moon_service import get_moon_phase


def get_bioluminescence_forecast(latitude: float, longitude: float) -> dict:
    """Score the likelihood of visible Noctiluca bioluminescence tonight."""
    snap = get_copernicus_marine_snapshot(latitude, longitude)
    obs = snap.get("observations", {}) or {}

    sst = (obs.get("temperature", {}) or {}).get("value")
    waves = (obs.get("waves", {}) or {}).get("significant_wave_height_m")

    weather = get_weather_conditions(latitude, longitude, forecast_days=1)
    current_w = weather.get("current", {}) if isinstance(weather, dict) else {}
    wind = current_w.get("wind_speed_10m")

    moon = get_moon_phase()

    score = 0.0
    factors: dict[str, str] = {}

    # SST (0–30)
    if sst is not None:
        if 22 <= sst <= 30:
            score += 30; factors["sst"] = "favorable (22–30 °C)"
        elif 18 <= sst < 22:
            score += 15; factors["sst"] = "marginal"
        else:
            factors["sst"] = "unfavorable"
    else:
        factors["sst"] = "no data"

    # Wind (0–25) — calm is best
    if wind is not None:
        if wind < 10:
            score += 25; factors["wind"] = "calm — ideal"
        elif wind < 15:
            score += 15; factors["wind"] = "light"
        else:
            factors["wind"] = "too windy"
    else:
        factors["wind"] = "no data"

    # Waves (0–15)
    if waves is not None:
        if waves < 0.5:
            score += 15; factors["waves"] = "very calm"
        elif waves < 1.0:
            score += 10; factors["waves"] = "calm"
        else:
            factors["waves"] = "too rough"
    else:
        factors["waves"] = "no data"

    # Season (0–25) — post-monsoon on Indian coast
    month = datetime.utcnow().month
    if month in (10, 11, 12, 1, 2):
        score += 25; factors["season"] = "post-monsoon — peak season"
    elif month in (9, 3):
        score += 10; factors["season"] = "shoulder season"
    else:
        factors["season"] = "off-season"

    # Moon (0–20) — dark nights favoured
    illum = moon["illumination_pct"]
    if illum < 20:
        score += 20; factors["moon"] = "new/dark — ideal"
    elif illum < 40:
        score += 12; factors["moon"] = "moderate darkness"
    elif illum < 70:
        score += 5; factors["moon"] = "bright"
    else:
        factors["moon"] = "full moon — washed out"

    score = min(score, 100)
    if score > 65:
        label = "HIGH"
    elif score > 40:
        label = "MODERATE"
    else:
        label = "LOW"

    return {
        "status": "FORECAST",
        "source": "ORCA bioluminescence heuristic",
        "organism": "Noctiluca scintillans (sea sparkle)",
        "query_point": {"latitude": latitude, "longitude": longitude},
        "likelihood_score": round(score, 1),
        "likelihood_label": label,
        "factors": factors,
        "disclaimer": (
            "Environmental-condition forecast only. Bioluminescence is never "
            "guaranteed — the organisms must be present. Best viewing after "
            "sunset on a dark beach around a new moon."
        ),
    }