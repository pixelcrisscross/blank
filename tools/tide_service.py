"""
Local harmonic tide prediction — pure Python, no external tide library.

Uses the standard harmonic summation:

    h(t) = Z0 + Σ A_i · cos(ω_i · t + φ_i − g_i)

where A_i is the amplitude, ω_i is the angular speed (deg/hour),
φ_i is the astronomical argument (approximated as 0), and g_i is the
phase lag for each tidal constituent.

Constituent speeds (Schureman SP98):
    M2  = 28.9841042 °/h
    S2  = 30.0000000 °/h
    N2  = 28.4397295 °/h
    K1  = 15.0410686 °/h
    O1  = 13.9430356 °/h
    P1  = 14.9589314 °/h

Amplitudes and phase lags are taken from published harmonic constants
for Indian ports. Verify against the latest Indian National Hydrographic
Office data before operational use.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from tools.moon_service import get_moon_phase


# ── Constituent angular speeds (degrees per hour) ────────────────────
_SPEEDS = {
    "M2": 28.9841042,
    "S2": 30.0000000,
    "N2": 28.4397295,
    "K1": 15.0410686,
    "O1": 13.9430356,
    "P1": 14.9589314,
}


# ── Harmonic constants per port: (name, amplitude_m, phase_deg) ──────
# Phase is the local phase lag g in degrees, referenced to the equilibrium
# argument. Source: published tide tables for Indian ports.
_PORT_CONSTITUENTS: dict[str, list[tuple[str, float, float]]] = {
    "kochi": [
        ("M2", 0.38, 145.0), ("S2", 0.15, 165.0),
        ("K1", 0.11, 305.0), ("O1", 0.08, 285.0),
        ("N2", 0.07, 130.0), ("P1", 0.04, 300.0),
    ],
    "mumbai": [
        ("M2", 1.65, 130.0), ("S2", 0.62, 155.0),
        ("K1", 0.35, 280.0), ("O1", 0.28, 260.0),
        ("N2", 0.32, 115.0), ("P1", 0.11, 275.0),
    ],
    "chennai": [
        ("M2", 0.45, 160.0), ("S2", 0.18, 180.0),
        ("K1", 0.10, 300.0), ("O1", 0.07, 285.0),
    ],
    "visakhapatnam": [
        ("M2", 0.55, 150.0), ("S2", 0.22, 170.0),
        ("K1", 0.12, 295.0), ("O1", 0.09, 280.0),
    ],
    "goa": [
        ("M2", 1.10, 140.0), ("S2", 0.42, 160.0),
        ("K1", 0.22, 290.0), ("O1", 0.18, 270.0),
    ],
    "karwar": [
        ("M2", 1.02, 142.0), ("S2", 0.39, 162.0),
        ("K1", 0.20, 292.0), ("O1", 0.16, 272.0),
    ],
    "mangalore": [
        ("M2", 0.88, 138.0), ("S2", 0.34, 158.0),
        ("K1", 0.18, 288.0), ("O1", 0.15, 268.0),
    ],
}


_PORT_LOOKUP = {
    "kochi": "kochi", "cochin": "kochi",
    "mumbai": "mumbai", "bombay": "mumbai",
    "chennai": "chennai", "madras": "chennai",
    "visakhapatnam": "visakhapatnam", "vizag": "visakhapatnam",
    "goa": "goa", "panaji": "goa", "margao": "goa",
    "karwar": "karwar",
    "mangalore": "mangalore", "mangaluru": "mangalore",
}


def _resolve_port(query: str | None) -> str | None:
    if not query:
        return None
    q = query.lower()
    for key, port in _PORT_LOOKUP.items():
        if key in q:
            return port
    return None


def _tide_height(constituents: list[tuple[str, float, float]],
                 t_hours: float) -> float:
    """
    Compute tide height (m) at t hours since the reference epoch
    (2000-01-01 00:00 UTC).

    h(t) = Σ A_i · cos( ω_i · t − g_i )
    """
    total = 0.0
    for name, amplitude, phase_deg in constituents:
        omega = _SPEEDS.get(name)
        if omega is None:
            continue
        angle_rad = math.radians(omega * t_hours - phase_deg)
        total += amplitude * math.cos(angle_rad)
    return total


def _hours_since_epoch(dt: datetime) -> float:
    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - epoch).total_seconds() / 3600.0


def get_tide_prediction(port_name: str | None = None,
                        hours: int = 24,
                        step_minutes: int = 10) -> dict[str, Any]:
    """
    Compute a 24-hour tide prediction for a supported Indian port.

    If the port isn't in the constituent table, returns a moon-regime
    estimate instead of fabricating a tide table.
    """
    port = _resolve_port(port_name)
    moon = get_moon_phase()

    if port is None:
        return {
            "status": "MOON_REGIME_ONLY",
            "source": "ORCA local tide service",
            "moon_regime": moon["tide_regime"],
            "illumination_pct": moon["illumination_pct"],
            "interpretation": {
                "spring": "Spring tides — largest tidal range. Strong currents.",
                "neap": "Neap tides — smallest tidal range. Slacker currents.",
                "intermediate": "Intermediate tide range.",
            }[moon["tide_regime"]],
            "note": (
                "No harmonic constituents available for this location. "
                "Returning a moon-regime estimate instead of a tide table."
            ),
        }

    constituents = _PORT_CONSTITUENTS[port]
    now = datetime.now(timezone.utc)
    steps = int(hours * 60 / step_minutes)

    samples = []
    for i in range(steps + 1):
        dt = now + timedelta(minutes=i * step_minutes)
        t_h = _hours_since_epoch(dt)
        height = _tide_height(constituents, t_h)
        samples.append((dt, height))

    # Find high/low extrema by local maxima/minima
    events = []
    for i in range(1, len(samples) - 1):
        _, prev_h = samples[i - 1]
        _, cur_h = samples[i]
        _, next_h = samples[i + 1]
        if cur_h > prev_h and cur_h > next_h:
            events.append({
                "time_utc": samples[i][0].isoformat(),
                "type": "HIGH",
                "height_m": round(cur_h, 2),
            })
        elif cur_h < prev_h and cur_h < next_h:
            events.append({
                "time_utc": samples[i][0].isoformat(),
                "type": "LOW",
                "height_m": round(cur_h, 2),
            })

    # Approximate current range from min/max of the window
    heights = [h for _, h in samples]
    range_m = max(heights) - min(heights)

    return {
        "status": "OK",
        "source": "ORCA local tide service (harmonic summation)",
        "port": port,
        "window_hours": hours,
        "moon_regime": moon["tide_regime"],
        "illumination_pct": moon["illumination_pct"],
        "current_height_m": round(heights[0], 2),
        "range_m": round(range_m, 2),
        "extremes": events,
        "note": (
            "Harmonic tide prediction using published constituents for "
            "the nearest reference port. Not a substitute for official "
            "INHO tide tables for navigation."
        ),
    }