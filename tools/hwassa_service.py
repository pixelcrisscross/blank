"""
INCOIS High Wave & Swell Surge Alert service.

Fetches active HWA/SSA alerts from INCOIS SAMUDRA API.
Alerts are district-level and forecast-based.

Severity:
    YELLOW  = Watch (monitor conditions)
    ORANGE  = Alert (be careful, restrictions likely)

Source: https://samudra.incois.gov.in/incoismobileappdata/rest/incois/hwassalatestdata
"""
from __future__ import annotations

import json
import time
from typing import Any

import httpx

_BASE = "https://samudra.incois.gov.in/incoismobileappdata/rest/incois"
_HWASSA_URL = f"{_BASE}/hwassalatestdata"

_CACHE: dict[str, Any] = {}
_CACHE_TTL = 900  # 15 minutes


def _fetch_alerts() -> dict:
    now = time.time()
    entry = _CACHE.get("hwassa")
    if entry and (now - entry["ts"]) < _CACHE_TTL:
        return entry["data"]

    r = httpx.get(_HWASSA_URL, timeout=20)
    r.raise_for_status()
    raw = r.json()

    hwa_alerts: list = []
    ssa_alerts: list = []

    try:
        hwa_raw = raw.get("HWAJson") or "[]"
        hwa_alerts = json.loads(hwa_raw) if isinstance(hwa_raw, str) else hwa_raw
    except Exception:
        pass

    try:
        ssa_raw = raw.get("SSAJson") or "[]"
        ssa_alerts = json.loads(ssa_raw) if isinstance(ssa_raw, str) else ssa_raw
    except Exception:
        pass

    result = {
        "LatestHWADate": raw.get("LatestHWADate"),
        "LatestSSADate": raw.get("LatestSSADate"),
        "hwa_alerts": hwa_alerts if isinstance(hwa_alerts, list) else [],
        "ssa_alerts": ssa_alerts if isinstance(ssa_alerts, list) else [],
    }
    _CACHE["hwassa"] = {"ts": now, "data": result}
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


def _matches_location(alert: dict, state_hint: str | None) -> bool:
    if not state_hint:
        return False
    alert_state = (alert.get("STATE") or "").upper()
    hint = state_hint.upper()
    return hint in alert_state or alert_state in hint


def get_hwassa_alerts(
    latitude: float,
    longitude: float,
    state_hint: str | None = None,
) -> dict[str, Any]:
    try:
        data = _fetch_alerts()
    except Exception as exc:
        return {
            "status": "ERROR",
            "source": "INCOIS HWA/SSA",
            "error": str(exc),
        }

    hwa = data.get("hwa_alerts") or []
    ssa = data.get("ssa_alerts") or []
    all_alerts = [("HWA", a) for a in hwa] + [("SSA", a) for a in ssa]

    matched = [
        (kind, alert) for kind, alert in all_alerts
        if _matches_location(alert, state_hint)
    ]

    india_wide = False
    if not matched and all_alerts:
        matched = all_alerts
        india_wide = True

    if not matched:
        return {
            "status": "OK",
            "source": "INCOIS HWA/SSA",
            "active_alerts": [],
            "max_severity": "NONE",
            "note": "No active High Wave or Swell Surge alerts for this location.",
        }

    matched.sort(key=lambda x: _severity_order(x[1].get("Color")), reverse=True)
    top_kind, top_alert = matched[0]
    color = (top_alert.get("Color") or "Green").upper()
    severity = (
        "ALERT" if color == "ORANGE"
        else "WATCH" if color == "YELLOW"
        else "NONE"
    )

    return {
        "status": "OK",
        "source": "INCOIS HWA/SSA",
        "data_date": data.get("LatestHWADate") or data.get("LatestSSADate"),
        "active_alert_count": len(matched),
        "max_severity": severity,
        "max_color": color,
        "alert_type": top_alert.get("Alert"),
        "district": top_alert.get("District"),
        "state": top_alert.get("STATE"),
        "message": top_alert.get("Message"),
        "issue_date": top_alert.get("Issue Date"),
        "all_alerts": [
            {
                "kind": kind,
                "alert": a.get("Alert"),
                "color": a.get("Color"),
                "district": a.get("District"),
                "state": a.get("STATE"),
                "message": a.get("Message"),
                "issue_date": a.get("Issue Date"),
            }
            for kind, a in matched[:10]
        ],
        "india_wide": india_wide,
        "note": (
            "Official INCOIS High Wave / Swell Surge advisory. "
            "Orange = Alert (be careful). Yellow = Watch (monitor)."
            if not india_wide
            else "No state-specific match; showing India-wide active alerts."
        ),
    }