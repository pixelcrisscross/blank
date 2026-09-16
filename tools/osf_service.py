"""
INCOIS Ocean State Forecast freshness check.

Returns the latest available forecast date for each parameter, so
downstream consumers can flag stale data.

Source:
    https://samudra.incois.gov.in/incoismobileappdata/rest/incois/latestavailableosfforecastdates
"""
from __future__ import annotations

import time
from typing import Any

import httpx

_BASE = "https://samudra.incois.gov.in/incoismobileappdata/rest/incois"
_OSF_URL = f"{_BASE}/latestavailableosfforecastdates"

_CACHE: dict[str, Any] = {}
_CACHE_TTL = 3600


def get_osf_freshness() -> dict[str, Any]:
    now = time.time()
    entry = _CACHE.get("osf")
    if entry and (now - entry["ts"]) < _CACHE_TTL:
        return entry["data"]

    try:
        r = httpx.get(_OSF_URL, timeout=15)
        r.raise_for_status()
        raw = r.json()
    except Exception as exc:
        return {
            "status": "ERROR",
            "source": "INCOIS OSF",
            "error": str(exc),
        }

    result = {
        "status": "OK",
        "source": "INCOIS Ocean State Forecast",
        "forecast_dates": raw,
        "note": (
            "Latest available forecast date per parameter. "
            "If today is later than these dates, the data is stale."
        ),
    }

    _CACHE["osf"] = {"ts": now, "data": result}
    return result