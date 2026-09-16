"""
INCOIS storm surge / cyclone alert service.

Source: https://samudra.incois.gov.in/incoismobileappdata/rest/incois/getlateststormsurge
"""
from __future__ import annotations

import time
from typing import Any

import httpx

_BASE = "https://samudra.incois.gov.in/incoismobileappdata/rest/incois"
_CYCLONE_URL = f"{_BASE}/getlateststormsurge"

_CACHE: dict[str, Any] = {}
_CACHE_TTL = 600  # 10 minutes


def get_cyclone_alerts() -> dict[str, Any]:
    now = time.time()
    entry = _CACHE.get("cyclone")
    if entry and (now - entry["ts"]) < _CACHE_TTL:
        return entry["data"]

    try:
        r = httpx.get(_CYCLONE_URL, timeout=20)
        r.raise_for_status()
        raw = r.json()
    except Exception as exc:
        return {
            "status": "ERROR",
            "source": "INCOIS Storm Surge",
            "error": str(exc),
        }

    alerts = raw.get("ActiveCycloneAlerts") or []

    result = {
        "status": "OK",
        "source": "INCOIS Storm Surge / Cyclone",
        "threshold_time": raw.get("ThresholdTime"),
        "active_alerts": alerts,
        "active_count": len(alerts),
        "has_active_cyclone": len(alerts) > 0,
        "note": (
            "No active cyclone alerts."
            if not alerts
            else f"{len(alerts)} active cyclone alert(s). Check IMD for track and intensity."
        ),
    }

    _CACHE["cyclone"] = {"ts": now, "data": result}
    return result