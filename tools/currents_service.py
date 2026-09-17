"""
INCOIS Ocean Current Watch service.

Fetches active ocean-current alerts from INCOIS SAMUDRA API (v2).

IMPORTANT: when no state_hint is available, this tool returns an
empty result with status NO_STATE_HINT. It does NOT fall back to
dumping India-wide alerts.

Source:
    https://samudra.incois.gov.in/incoismobileappdata2/rest/incois/currentslatestdatalang/en
"""
from __future__ import annotations

import json
import time
from typing import Any

import httpx

_BASE = "https://samudra.incois.gov.in/incoismobileappdata2/rest/incois"
_CURRENTS_URL = f"{_BASE}/currentslatestdatalang/en"

_CACHE: dict[str, Any] = {}
_CACHE_TTL = 900  # 15 minutes


def _fetch_currents() -> dict:
    now = time.time()
    entry = _CACHE.get("currents")
    if entry and (now - entry["ts"]) < _CACHE_TTL:
        return entry["data"]

    r = httpx.get(_CURRENTS_URL, timeout=20)
    r.raise_for_status()
    raw = r.json()

    alerts: list = []
    try:
        cj = raw.get("CurrentsJson") or "[]"
        alerts = json.loads(cj) if isinstance(cj, str) else cj
    except Exception:
        pass

    result = {
        "LatestCurrentsDate": raw.get("LatestCurrentsDate"),
        "alerts": alerts if isinstance(alerts, list) else [],
    }
    _CACHE["currents"] = {"ts": now, "data": result}
    return result


def _severity_order(color: str) -> int:
    c = (color or "").lower()
    if c == "orange":
        return 3
    if c == "yellow":
        return 2
    if c == "green":
        return 1
    return 0


def _state_matches(alert: dict, state_hint: str) -> bool:
    alert_state = (alert.get("STATE") or "").upper().strip()
    hint = state_hint.upper().strip()
    if not alert_state or not hint:
        return False
    return hint in alert_state or alert_state in hint


def get_ocean_current_alerts(
    latitude: float,
    longitude: float,
    state_hint: str | None = None,
) -> dict[str, Any]:
    """
    Return active Ocean Current advisories for a state.

    If no state_hint is provided, returns an empty result with status
    NO_STATE_HINT.
    """
    if not state_hint or not state_hint.strip():
        return {
            "status": "NO_STATE_HINT",
            "source": "INCOIS Ocean Current Watch",
            "active_alerts": [],
            "max_severity": "NONE",
            "note": (
                "No state hint available — cannot filter India-wide "
                "current advisories to the query location."
            ),
        }

    try:
        data = _fetch_currents()
    except Exception as exc:
        return {
            "status": "ERROR",
            "source": "INCOIS Ocean Current Watch",
            "error": str(exc),
        }

    alerts = data.get("alerts") or []
    matched = [a for a in alerts if _state_matches(a, state_hint)]

    if not matched:
        return {
            "status": "OK",
            "source": "INCOIS Ocean Current Watch",
            "state_hint": state_hint,
            "active_alerts": [],
            "max_severity": "NONE",
            "note": f"No active Ocean Current advisories for {state_hint}.",
        }

    matched.sort(key=lambda a: _severity_order(a.get("Color")), reverse=True)
    top = matched[0]
    color = (top.get("Color") or "Green").upper()
    severity = (
        "ALERT" if color == "ORANGE"
        else "WATCH" if color == "YELLOW"
        else "NONE"
    )

    return {
        "status": "OK",
        "source": "INCOIS Ocean Current Watch",
        "state_hint": state_hint,
        "data_date": data.get("LatestCurrentsDate"),
        "active_alert_count": len(matched),
        "max_severity": severity,
        "max_color": color,
        "alert_type": top.get("AlertLang") or top.get("Alert"),
        "district": top.get("District"),
        "state": top.get("STATE"),
        "message": top.get("MessageLang") or top.get("Message"),
        "issue_date": top.get("Issue Date"),
        "all_alerts": [
            {
                "alert": a.get("AlertLang") or a.get("Alert"),
                "color": a.get("Color"),
                "district": a.get("District"),
                "state": a.get("STATE"),
                "message": a.get("MessageLang") or a.get("Message"),
                "issue_date": a.get("Issue Date"),
            }
            for a in matched[:10]
        ],
        "note": (
            "Official INCOIS Ocean Current advisory. "
            "Orange = Alert (caution). Yellow = Watch (monitor)."
        ),
    }