"""
INCOIS High Wave & Swell Surge Alert service.

Fetches active HWA/SSA alerts from INCOIS SAMUDRA API.
Alerts are district-level and forecast-based.

IMPORTANT: when no state_hint is available, this tool returns an
empty result with status NO_STATE_HINT. It does NOT fall back to
dumping India-wide alerts — that behavior caused the pipeline to
report Andaman & Nicobar warnings for queries about Karnataka or
Kerala.

Source: https://samudra.incois.gov.in/incoismobileappdata/rest/incois/hwassalatestdata
"""
from __future__ import annotations

import json
import re
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


def _state_matches(alert: dict, state_hint: str) -> bool:
    alert_state = (alert.get("STATE") or "").upper().strip()
    hint = state_hint.upper().strip()
    if not alert_state or not hint:
        return False
    return hint in alert_state or alert_state in hint


def _parse_swell_from_message(message: str) -> dict:
    out: dict = {}
    if not message:
        return out

    m = re.search(
        r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(?:sec|s)\b",
        message, flags=re.IGNORECASE,
    )
    if m:
        out["swell_period_s_range"] = [float(m.group(1)), float(m.group(2))]

    m = re.search(
        r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(?:m\b|meters?)",
        message, flags=re.IGNORECASE,
    )
    if m:
        out["wave_height_m_range"] = [float(m.group(1)), float(m.group(2))]

    if out.get("swell_period_s_range"):
        max_period = max(out["swell_period_s_range"])
        if max_period >= 12.0:
            out["rip_current_risk"] = True
            out["rip_current_note"] = (
                f"Long-period swell ({out['swell_period_s_range'][0]:.0f}-"
                f"{out['swell_period_s_range'][1]:.0f} s) can generate strong "
                f"rip currents near beaches even though wave heights are "
                f"modest. Swimmers and casual beachgoers should exercise "
                f"caution."
            )

    return out


def get_hwassa_alerts(
    latitude: float,
    longitude: float,
    state_hint: str | None = None,
) -> dict[str, Any]:
    """
    Return active HWA/SSA alerts for a state.

    If no state_hint is provided, returns an empty result with status
    NO_STATE_HINT. This is critical — the previous version dumped every
    India-wide alert when the state was unknown, which caused misleading
    warnings for a query about a location in Karnataka.
    """
    # Guard: without a state hint we cannot filter safely.
    if not state_hint or not state_hint.strip():
        return {
            "status": "NO_STATE_HINT",
            "source": "INCOIS HWA/SSA",
            "active_alerts": [],
            "max_severity": "NONE",
            "note": (
                "No state hint available — cannot filter India-wide "
                "alerts to the query location."
            ),
        }

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
        if _state_matches(alert, state_hint)
    ]

    if not matched:
        return {
            "status": "OK",
            "source": "INCOIS HWA/SSA",
            "state_hint": state_hint,
            "active_alerts": [],
            "max_severity": "NONE",
            "note": f"No active HWA/SSA alerts for {state_hint}.",
        }

    matched.sort(key=lambda x: _severity_order(x[1].get("Color")), reverse=True)
    top_kind, top_alert = matched[0]
    color = (top_alert.get("Color") or "Green").upper()
    severity = (
        "ALERT" if color == "ORANGE"
        else "WATCH" if color == "YELLOW"
        else "NONE"
    )

    swell = _parse_swell_from_message(top_alert.get("Message") or "")

    return {
        "status": "OK",
        "source": "INCOIS HWA/SSA",
        "state_hint": state_hint,
        "data_date": data.get("LatestHWADate") or data.get("LatestSSADate"),
        "active_alert_count": len(matched),
        "max_severity": severity,
        "max_color": color,
        "alert_type": top_alert.get("Alert"),
        "district": top_alert.get("District"),
        "state": top_alert.get("STATE"),
        "message": top_alert.get("Message"),
        "issue_date": top_alert.get("Issue Date"),
        **swell,
        "all_alerts": [
            {
                "kind": kind,
                "alert": a.get("Alert"),
                "color": a.get("Color"),
                "district": a.get("District"),
                "state": a.get("STATE"),
                "message": a.get("Message"),
                "issue_date": a.get("Issue Date"),
                **_parse_swell_from_message(a.get("Message") or ""),
            }
            for kind, a in matched[:10]
        ],
        "note": (
            "Official INCOIS High Wave / Swell Surge advisory. "
            "Orange = Alert (be careful). Yellow = Watch (monitor)."
        ),
    }