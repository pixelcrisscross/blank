"""
Local moon-phase and illumination calculator.

Uses the synodic-month approximation. Accuracy is ~1–2% illumination,
which is more than sufficient for bioluminescence and solunar guidance.

For arcsecond precision, swap in Skyfield with a JPL ephemeris — the
public API here stays the same.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

# Reference new moon: 2000-01-06 18:14 UTC
_KNOWN_NEW_MOON = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
_SYNODIC_MONTH = 29.530588853  # days


def _phase_name(age: float) -> str:
    if age < 1.85:
        return "New Moon"
    if age < 5.54:
        return "Waxing Crescent"
    if age < 9.23:
        return "First Quarter"
    if age < 12.91:
        return "Waxing Gibbous"
    if age < 16.61:
        return "Full Moon"
    if age < 20.30:
        return "Waning Gibbous"
    if age < 23.99:
        return "Last Quarter"
    return "Waning Crescent"


def get_moon_phase(dt: datetime | None = None) -> dict:
    """Compute moon phase and illumination locally."""
    dt = dt or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    days_since = (dt - _KNOWN_NEW_MOON).total_seconds() / 86400.0
    phase_age = days_since % _SYNODIC_MONTH
    illumination = (1 - math.cos(phase_age * 2 * math.pi / _SYNODIC_MONTH)) / 2

    # Spring tides at new/full moon (≈0 or ≈14.77 days); neap at quarters
    distance_from_spring = min(
        abs(phase_age - 0),
        abs(phase_age - _SYNODIC_MONTH / 2),
        abs(phase_age - _SYNODIC_MONTH),
    )
    tide_regime = "spring" if distance_from_spring < 2.0 else (
        "neap" if abs(phase_age - _SYNODIC_MONTH / 4) < 2.0
        or abs(phase_age - 3 * _SYNODIC_MONTH / 4) < 2.0
        else "intermediate"
    )

    return {
        "status": "OK",
        "source": "ORCA local moon calculator (synodic approximation)",
        "timestamp_utc": dt.isoformat(),
        "phase_name": _phase_name(phase_age),
        "phase_age_days": round(phase_age, 2),
        "illumination_pct": round(illumination * 100, 1),
        "tide_regime": tide_regime,  # spring | neap | intermediate
        "is_dark_night": illumination < 0.30,
    }