"""
INCOIS Tsunami Early Warning service.

Fetches recent earthquake/tsunami bulletins from the ITEWS Decision
Support System. The key field is EVALUATION, which states whether the
event poses a tsunami threat to India.

Source: https://tsunami.incois.gov.in/itews/DSSProducts/OPR/past90days.json
"""
from __future__ import annotations

import time
from typing import Any

import httpx
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_TSUNAMI_URL = (
    "https://tsunami.incois.gov.in/itews/DSSProducts/OPR/past90days.json"
)

_CACHE: dict[str, Any] = {}
_CACHE_TTL = 900


def get_tsunami_bulletins() -> dict[str, Any]:
    now = time.time()
    entry = _CACHE.get("tsunami")
    if entry and (now - entry["ts"]) < _CACHE_TTL:
        return entry["data"]

    data = None
    last_error = None

    # Attempt 1: strict TLS
    try:
        r = httpx.get(_TSUNAMI_URL, timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        last_error = exc

    # Attempt 2: skip verification (INCOIS host has an incomplete chain)
    if data is None:
        try:
            r = httpx.get(_TSUNAMI_URL, timeout=25, verify=False)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            last_error = exc

    if data is None:
        return {
            "status": "ERROR",
            "source": "INCOIS ITEWS",
            "error": str(last_error),
        }

    features = data.get("features", [])
    events = []
    for f in features:
        props = f.get("properties", {}) or {}
        events.append({
            "event_id": props.get("EVID"),
            "magnitude": props.get("MAGNITUDE"),
            "origin_time": props.get("OT"),
            "latitude": props.get("LATITUDE"),
            "longitude": props.get("LONGITUDE"),
            "depth_km": props.get("DEPTH"),
            "region": props.get("REGIONNAME"),
            "evaluation": props.get("EVALUATION"),
            "detail_url": props.get("detail"),
        })

    events.sort(key=lambda e: e.get("origin_time") or "", reverse=True)

    threat_events = [
        e for e in events
        if e.get("evaluation")
        and "does not exist" not in (e["evaluation"] or "").lower()
        and "no threat" not in (e["evaluation"] or "").lower()
    ]

    result = {
        "status": "OK",
        "source": "INCOIS Tsunami Early Warning (ITEWS)",
        "event_count": len(events),
        "recent_events": events[:10],
        "threat_to_india": len(threat_events) > 0,
        "threat_events": threat_events[:5],
        "note": (
            "Recent ITEWS earthquake bulletins. The EVALUATION field states "
            "the official threat assessment for India."
        ),
    }

    _CACHE["tsunami"] = {"ts": now, "data": result}
    return result