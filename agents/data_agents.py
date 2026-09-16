from __future__ import annotations

import asyncio
import json
from typing import ClassVar

from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types

from state.schemas import key

from tools.location import resolve_location_query
from tools.nearest_coast import nearest_coast_info
from tools.copernicus_service import get_copernicus_marine_snapshot
from tools.weather_service import get_weather_conditions
from tools.geofence import check_geofence
from tools.marine_risk import calculate_marine_risk
from tools.pfz_service import get_pfz_advisory
from tools.coral_service import get_coral_bleaching_alert
from tools.tide_service import get_tide_prediction
from tools.bioluminescence import get_bioluminescence_forecast
from tools.algal_bloom import get_algal_bloom_risk
from tools.imd_service import get_imd_coastal_bulletin
from tools.hwassa_service import get_hwassa_alerts
from tools.cyclone_service import get_cyclone_alerts
from tools.tsunami_service import get_tsunami_bulletins
from tools.osf_service import get_osf_freshness
from tools.currents_service import get_ocean_current_alerts


def _json_block(payload) -> str:
    try:
        return json.dumps(payload, indent=None, default=str)[:4000]
    except Exception:
        return str(payload)[:4000]


# ═════════════════════════════════════════════════════════════════════
# DOMAIN SUMMARISERS
# ═════════════════════════════════════════════════════════════════════

def _fmt_ocean(r: dict) -> str:
    obs = r.get("observations", {}) or {}
    sst = (obs.get("temperature") or {}).get("value")
    cur = obs.get("currents") or {}
    wav = obs.get("waves") or {}
    chl = (obs.get("chlorophyll") or {}).get("value")
    return (
        f"Ocean data (Copernicus Marine), status={r.get('status')}:\n"
        f"- SST (OBSERVED): {sst} °C\n"
        f"- Current speed: {cur.get('speed_ms')} m/s, "
        f"direction {cur.get('direction_deg')}°\n"
        f"- Significant wave height: {wav.get('significant_wave_height_m')} m, "
        f"period {wav.get('mean_wave_period_s')} s\n"
        f"- Chlorophyll-a: {chl} mg/m³\n"
        f"- Observation time: "
        f"{(obs.get('temperature') or {}).get('observation_time')}"
    )


def _fmt_weather(r: dict) -> str:
    cur = r.get("current", {}) or {}
    units = r.get("current_units", {}) or {}
    return (
        f"Weather (Open-Meteo), status={r.get('status')}:\n"
        f"- Wind speed: {cur.get('wind_speed_10m')} "
        f"{units.get('wind_speed_10m', '')}\n"
        f"- Wind gust: {cur.get('wind_gusts_10m')} "
        f"{units.get('wind_gusts_10m', '')}\n"
        f"- Precipitation: {cur.get('precipitation')} mm\n"
        f"- Weather code: {cur.get('weather_code')}\n"
        f"- Pressure: {cur.get('pressure_msl')} hPa"
    )


def _fmt_geofence(r: dict) -> str:
    return (
        f"Geofence check, status={r.get('status')}:\n"
        f"- Inside restricted zone: {r.get('inside_restricted_zone')}\n"
        f"- Matches: {r.get('matches')}\n"
        f"- Note: {r.get('warning')}"
    )


def _fmt_pfz(r: dict) -> str:
    lines = r.get("pfz_lines", []) or []
    lcs = r.get("landing_centres", []) or []
    s = f"PFZ advisory (INCOIS), status={r.get('status')}:\n"
    if lines:
        n = lines[0]
        s += (
            f"- Nearest PFZ line: {n.get('distance_km')} km away at "
            f"({n.get('nearest_lat')}, {n.get('nearest_lon')}), "
            f"state={n.get('state')}\n"
            f"- Total PFZ lines within radius: {len(lines)}\n"
        )
    else:
        s += "- No PFZ lines within radius.\n"
    if lcs:
        lc = lcs[0]
        s += (
            f"- Nearest landing centre: {lc.get('name')} "
            f"({lc.get('sector')}), {lc.get('distance_km')} km away\n"
            f"- Total landing centres within radius: {len(lcs)}\n"
        )
    else:
        s += "- No landing centres within radius.\n"
    s += f"- Note: {r.get('note')}"
    return s


def _fmt_coral(r: dict) -> str:
    return (
        f"Coral bleaching (NOAA CRW), status={r.get('status')}:\n"
        f"- Bleaching alert: {r.get('bleaching_alert')}\n"
        f"- Level code: {r.get('bleaching_alert_level')}\n"
        f"- Reef region: {r.get('reef_region')}\n"
        f"- Distance to reef: {r.get('distance_to_reef_km')} km\n"
        f"- Interpretation: {r.get('interpretation')}\n"
        f"- Note: {r.get('note')}"
    )


def _fmt_tides(r: dict) -> str:
    return (
        f"Tides (local harmonic prediction), status={r.get('status')}:\n"
        f"- Port: {r.get('port')}\n"
        f"- Moon regime: {r.get('moon_regime')}, "
        f"illumination {r.get('illumination_pct')}%\n"
        f"- Range over window: {r.get('range_m')} m\n"
        f"- Extremes: {r.get('extremes')}\n"
        f"- Note: {r.get('note')}"
    )


def _fmt_biolum(r: dict) -> str:
    return (
        f"Bioluminescence forecast (FORECAST): {r.get('likelihood_label')} "
        f"({r.get('likelihood_score')}/100)\n"
        f"- Factors: {r.get('factors')}\n"
        f"- Disclaimer: {r.get('disclaimer')}"
    )


def _fmt_algal(r: dict) -> str:
    return (
        f"Algal bloom risk, status={r.get('status')}:\n"
        f"- Chlorophyll-a: {r.get('chlorophyll_a_mg_m3')} mg/m³\n"
        f"- Risk level: {r.get('risk_level')}\n"
        f"- Interpretation: {r.get('interpretation')}"
    )


def _fmt_imd(r: dict) -> str:
    status = r.get("status")
    if status == "NO_REGION":
        return "IMD coastal bulletin: NO_REGION (point not covered)."
    if status == "PARSE_FAILED":
        return (
            "IMD coastal bulletin: PARSE_FAILED — page fetched but fields "
            "could not be extracted."
        )
    if status == "ERROR":
        return f"IMD coastal bulletin: ERROR — {r.get('error')}"
    if status != "OK":
        return f"IMD coastal bulletin: {status}"

    return (
        f"IMD Coastal Weather Bulletin ({r.get('centre')}), "
        f"issued {r.get('issued_at')}:\n"
        f"- Region: {r.get('region')}\n"
        f"- Valid: {r.get('valid_from')} → {r.get('valid_to')}\n"
        f"- Wind: {r.get('wind')}\n"
        f"- Weather: {r.get('weather')}\n"
        f"- Visibility: {r.get('visibility')}\n"
        f"- Sea condition: {r.get('sea_condition')}\n"
        f"- PORT SIGNAL: {r.get('port_signal')} "
        f"(active={r.get('port_signal_active')})\n"
        f"- STORM SURGE / TIDAL WARNING: "
        f"{r.get('storm_surge_warning')} "
        f"(active={r.get('storm_surge_active')})\n"
        f"- Attribution: {r.get('attribution')}"
    )


def _fmt_hwassa(r: dict) -> str:
    status = r.get("status")
    if status != "OK":
        return f"INCOIS HWA/SSA: {status} — {r.get('error', '')}"

    sev = r.get("max_severity", "NONE")
    if sev == "NONE":
        return "INCOIS HWA/SSA: No active High Wave or Swell Surge alerts."

    return (
        f"INCOIS HWA/SSA ALERT — severity={sev} ({r.get('max_color')}):\n"
        f"- Type: {r.get('alert_type')}\n"
        f"- District: {r.get('district')}, {r.get('state')}\n"
        f"- Issued: {r.get('issue_date')}\n"
        f"- Message: {r.get('message')}\n"
        f"- Note: {r.get('note')}"
    )


def _fmt_cyclone(r: dict) -> str:
    if r.get("status") != "OK":
        return f"INCOIS Cyclone: {r.get('status')} — {r.get('error', '')}"
    if not r.get("has_active_cyclone"):
        return "INCOIS Cyclone: No active cyclone alerts."
    return (
        f"INCOIS Cyclone — {r.get('active_count')} active alert(s):\n"
        f"- {r.get('active_alerts')}\n"
        f"- Note: {r.get('note')}"
    )


def _fmt_tsunami(r: dict) -> str:
    if r.get("status") != "OK":
        return f"INCOIS Tsunami: {r.get('status')} — {r.get('error', '')}"
    if not r.get("threat_to_india"):
        return (
            f"INCOIS Tsunami: {r.get('event_count')} recent events, "
            f"no threat to India."
        )
    return (
        f"INCOIS Tsunami THREAT TO INDIA:\n"
        f"- {len(r.get('threat_events', []))} threatening event(s)\n"
        f"- Details: {r.get('threat_events')}\n"
        f"- Note: {r.get('note')}"
    )


def _fmt_osf(r: dict) -> str:
    if r.get("status") != "OK":
        return f"INCOIS OSF freshness: {r.get('status')}"
    return (
        f"INCOIS OSF forecast dates: {r.get('forecast_dates', {})}\n"
        f"- Note: {r.get('note')}"
    )


def _fmt_currents(r: dict) -> str:
    status = r.get("status")
    if status != "OK":
        return f"INCOIS Ocean Currents: {status} — {r.get('error', '')}"

    sev = r.get("max_severity", "NONE")
    if sev == "NONE":
        return "INCOIS Ocean Currents: No active advisories."

    return (
        f"INCOIS Ocean Current ALERT — severity={sev} ({r.get('max_color')}):\n"
        f"- Type: {r.get('alert_type')}\n"
        f"- District: {r.get('district')}, {r.get('state')}\n"
        f"- Issued: {r.get('issue_date')}\n"
        f"- Message: {r.get('message')}\n"
        f"- Note: {r.get('note')}"
    )


_SUMMARISERS = {
    "ocean": _fmt_ocean,
    "weather": _fmt_weather,
    "geofence": _fmt_geofence,
    "pfz": _fmt_pfz,
    "coral": _fmt_coral,
    "tides": _fmt_tides,
    "biolum": _fmt_biolum,
    "algal_bloom": _fmt_algal,
    "imd": _fmt_imd,
    "hwassa": _fmt_hwassa,
    "cyclone": _fmt_cyclone,
    "tsunami": _fmt_tsunami,
    "osf_freshness": _fmt_osf,
    "currents": _fmt_currents,
}


# ═════════════════════════════════════════════════════════════════════
# LOCATION RESOLUTION
# ═════════════════════════════════════════════════════════════════════

class ResolveLocationAgent(BaseAgent):
    """
    Resolve the location and compute the marine query point.

    Writes:
      - orca_location        : the raw query point
      - orca_marine_location : the point marine tools should query.
                               Equal to the query point when coastal,
                               otherwise the nearest ocean point within
                               a 200 km radius. None when deep inland.
    """

    async def _run_async_impl(self, ctx: InvocationContext):
        plan_raw = ctx.session.state.get(key("plan"), "")
        location_hint = ""
        coords = None
        try:
            plan = json.loads(plan_raw) if isinstance(plan_raw, str) else plan_raw
            if isinstance(plan, dict):
                location_hint = str(plan.get("location") or "").strip()
                coords = plan.get("coordinates")
        except Exception:
            location_hint = str(plan_raw).strip()

        if (
            isinstance(coords, dict)
            and coords.get("latitude") is not None
            and coords.get("longitude") is not None
        ):
            result = {
                "status": "FOUND",
                "name": "Explicit coordinates",
                "latitude": float(coords["latitude"]),
                "longitude": float(coords["longitude"]),
                "source": "Plan coordinates",
            }
        elif location_hint:
            result = await asyncio.to_thread(resolve_location_query, location_hint)
        else:
            result = {"status": "NOT_FOUND", "error": "No location in plan."}

        ctx.session.state[key("location")] = result

        # ── Compute the marine query point ─────────────────────────
        marine_location = None
        marine_note = ""

        if result.get("status") == "FOUND":
            lat = float(result["latitude"])
            lon = float(result["longitude"])
            coast_info = await asyncio.to_thread(nearest_coast_info, lat, lon)

            if coast_info["is_coastal"]:
                marine_location = {
                    "latitude": lat,
                    "longitude": lon,
                    "name": result.get("name"),
                    "admin1": result.get("admin1"),
                    "country": result.get("country"),
                    "source": result.get("source"),
                    "distance_from_query_km": 0.0,
                }
                marine_note = "Query point is coastal."
            elif coast_info["within_range"]:
                marine_location = {
                    "latitude": coast_info["coastal_point"]["latitude"],
                    "longitude": coast_info["coastal_point"]["longitude"],
                    "name": f"nearest coast to {result.get('name')}",
                    "admin1": result.get("admin1"),
                    "country": result.get("country"),
                    "source": "ORCA nearest-coast lookup",
                    "distance_from_query_km": coast_info["distance_km"],
                }
                marine_note = coast_info["note"]
            else:
                marine_note = coast_info["note"]

        ctx.session.state[key("marine_location")] = marine_location

        # ── Event ─────────────────────────────────────────────────
        if result.get("status") == "FOUND":
            lines = [
                f"Location resolved: {result.get('name')}, "
                f"{result.get('admin1')}, {result.get('country')}. "
                f"Latitude={result.get('latitude')}, "
                f"Longitude={result.get('longitude')}."
            ]
            if marine_location is not None:
                dist = marine_location.get("distance_from_query_km", 0.0)
                if dist and dist > 0:
                    lines.append(
                        f"Marine query point: "
                        f"{marine_location['latitude']:.4f}, "
                        f"{marine_location['longitude']:.4f} "
                        f"({dist:.1f} km from query point). {marine_note}"
                    )
                else:
                    lines.append(f"Marine query point = query point. {marine_note}")
            else:
                lines.append(f"No marine query point available. {marine_note}")
            text = "\n".join(lines)
        else:
            text = f"Location resolution failed: {_json_block(result)}"

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=text)]),
        )


# ═════════════════════════════════════════════════════════════════════
# CAPABILITY AGENT
# ═════════════════════════════════════════════════════════════════════

class CapabilityAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        plan_raw = ctx.session.state.get(key("plan"), "{}")
        try:
            plan = json.loads(plan_raw) if isinstance(plan_raw, str) else plan_raw
        except Exception:
            plan = {}

        if not isinstance(plan, dict) or plan.get("intent") != "meta":
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="Not a meta-query; skipping.")],
                ),
            )
            return

        capabilities = """
ORCA — marine intelligence platform. Capabilities:

- Marine observations (Copernicus Marine): SST, currents, waves, chlorophyll-a
- Weather & forecast (Open-Meteo): wind, gusts, precipitation, visibility
- Fishery zones (INCOIS PFZ): official PFZ lines and landing centres
- Coral reefs (NOAA Coral Reef Watch): bleaching alerts
- Tides (local harmonic prediction): high/low extremes
- Bioluminescence (heuristic): Noctiluca forecast for tourism
- Algal blooms (Copernicus chlorophyll): bloom risk classification
- IMD Coastal Bulletin: official port signals, storm surge warnings
- INCOIS HWA/SSA: official High Wave & Swell Surge alerts
- INCOIS Ocean Current Watch: surface current advisories
- INCOIS ITEWS: tsunami threat bulletins
- INCOIS Cyclone: storm surge alerts
- Safety (deterministic): risk score with hard blockers
- Location resolution: place names or explicit coordinates

Note: marine domains use the nearest coast to the query location when
the query is inland (up to 200 km). Beyond that, only weather applies.

Example questions:
- "What is the SST off Goa right now?"
- "Where can I fish near Kochi tomorrow?"
- "Are there coral reefs near Lakshadweep?"
- "What's the tide at Mumbai tonight?"
- "Are there any port warnings off Visakhapatnam?"
"""
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=capabilities)],
            ),
        )


# ═════════════════════════════════════════════════════════════════════
# DYNAMIC DATA COLLECTION
# ═════════════════════════════════════════════════════════════════════

class DynamicDataCollectionAgent(BaseAgent):
    """
    Read domains_needed from the plan, run only the required collectors.

    Marine tools (ocean, pfz, coral, biolum, algal_bloom, geofence) run
    at the marine query point — which is either the raw query point
    (when coastal) or the nearest coast within 200 km.

    Global tools (weather, IMD) run at the raw query point.

    Hint-based tools (hwassa, currents, tides, cyclone, tsunami,
    osf_freshness) don't need coordinates — they use the state or
    centre name.
    """

    MARINE_DOMAINS: ClassVar[set[str]] = {
        "ocean", "pfz", "coral", "biolum", "algal_bloom", "geofence",
    }
    GLOBAL_DOMAINS: ClassVar[set[str]] = {
        "weather", "imd",
    }
    HINT_DOMAINS: ClassVar[set[str]] = {
        "hwassa", "currents", "tides", "cyclone", "tsunami", "osf_freshness",
    }

    async def _run_async_impl(self, ctx: InvocationContext):
        plan_raw = ctx.session.state.get(key("plan"), "{}")
        try:
            plan = json.loads(plan_raw) if isinstance(plan_raw, str) else plan_raw
        except Exception:
            plan = {}

        if isinstance(plan, dict) and plan.get("intent") == "meta":
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="Meta-query — no data collection.")],
                ),
            )
            return

        domains = plan.get("domains_needed") or plan.get("domains") or []
        if not isinstance(domains, list):
            domains = []
        if not domains:
            domains = list(_SUMMARISERS.keys())

        loc = ctx.session.state.get(key("location"), {})
        marine_loc = ctx.session.state.get(key("marine_location"))
        loc_ok = loc.get("status") == "FOUND"
        marine_available = marine_loc is not None

        filtered = []
        skipped_inland = []
        for d in domains:
            if d in self.MARINE_DOMAINS and not marine_available:
                skipped_inland.append(d)
                ctx.session.state[key(d)] = {
                    "status": "BLOCKED_INLAND",
                    "reason": (
                        "Location is deep inland — no nearest coast "
                        "within the marine data range (200 km)."
                    ),
                }
            else:
                filtered.append(d)

        state_hint = loc.get("admin1") or loc.get("name")

        # ── Runners ───────────────────────────────────────────────
        async def run_marine(name: str, fn):
            if not marine_available:
                ctx.session.state[key(name)] = {
                    "status": "BLOCKED",
                    "reason": "No marine query point.",
                }
                return
            try:
                result = await asyncio.to_thread(
                    fn,
                    float(marine_loc["latitude"]),
                    float(marine_loc["longitude"]),
                )
                ctx.session.state[key(name)] = result
            except Exception as exc:
                ctx.session.state[key(name)] = {
                    "status": "ERROR",
                    "error": str(exc),
                }

        async def run_global(name: str, fn):
            if not loc_ok:
                ctx.session.state[key(name)] = {
                    "status": "BLOCKED",
                    "reason": "Location unavailable.",
                }
                return
            try:
                result = await asyncio.to_thread(
                    fn,
                    float(loc["latitude"]),
                    float(loc["longitude"]),
                )
                ctx.session.state[key(name)] = result
            except Exception as exc:
                ctx.session.state[key(name)] = {
                    "status": "ERROR",
                    "error": str(exc),
                }

        async def run_hint(name: str, fn):
            try:
                result = await asyncio.to_thread(fn)
                ctx.session.state[key(name)] = result
            except Exception as exc:
                ctx.session.state[key(name)] = {
                    "status": "ERROR",
                    "error": str(exc),
                }

        # ── Dispatchers ───────────────────────────────────────────
        # Marine tools -> marine_loc
        marine_dispatcher = {
            "ocean":       ("ocean",       get_copernicus_marine_snapshot),
            "geofence":    ("geofence",    check_geofence),
            "pfz":         ("pfz",         get_pfz_advisory),
            "coral":       ("coral",       get_coral_bleaching_alert),
            "biolum":      ("biolum",      get_bioluminescence_forecast),
            "algal_bloom": ("algal_bloom", get_algal_bloom_risk),
        }
        # Global tools -> raw loc
        global_dispatcher = {
            "weather": ("weather", lambda a, b: get_weather_conditions(a, b, 3)),
            "imd":     ("imd",     lambda a, b: get_imd_coastal_bulletin(
                                        a, b, state_hint=state_hint
                                    )),
        }
        # Hint-based tools -> no coordinates needed
        hint_dispatcher = {
            "hwassa":        ("hwassa",        lambda: get_hwassa_alerts(
                                                    0, 0, state_hint=state_hint
                                                )),
            "currents":      ("currents",      lambda: get_ocean_current_alerts(
                                                    0, 0, state_hint=state_hint
                                                )),
            "tides":         ("tides",         lambda: get_tide_prediction(
                                                    loc.get("name")
                                                )),
            "cyclone":       ("cyclone",       lambda: get_cyclone_alerts()),
            "tsunami":       ("tsunami",       lambda: get_tsunami_bulletins()),
            "osf_freshness": ("osf_freshness", lambda: get_osf_freshness()),
        }

        # ── Fire in parallel ──────────────────────────────────────
        tasks = []
        for d in filtered:
            if d in marine_dispatcher:
                _, fn = marine_dispatcher[d]
                tasks.append(run_marine(d, fn))
            elif d in global_dispatcher:
                _, fn = global_dispatcher[d]
                tasks.append(run_global(d, fn))
            elif d in hint_dispatcher:
                _, fn = hint_dispatcher[d]
                tasks.append(run_hint(d, fn))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # ── Emit per-domain summaries ─────────────────────────────
        all_dispatchers = {**marine_dispatcher, **global_dispatcher, **hint_dispatcher}
        for d in filtered:
            if d not in all_dispatchers:
                continue
            state_name = all_dispatchers[d][0]
            result = ctx.session.state.get(key(state_name), {}) or {}
            formatter = _SUMMARISERS.get(state_name)
            text = formatter(result) if formatter else _json_block(result)
            yield Event(
                invocation_id=ctx.invocation_id,
                author=f"{self.name}_{state_name}",
                content=types.Content(role="model", parts=[types.Part(text=text)]),
            )

        for d in skipped_inland:
            yield Event(
                invocation_id=ctx.invocation_id,
                author=f"{self.name}_{d}",
                content=types.Content(
                    role="model",
                    parts=[types.Part(
                        text=(
                            f"{d.upper()} DATA: SKIPPED — location is deep "
                            f"inland (no coast within 200 km). No data was "
                            f"collected. Do not provide any values for this "
                            f"domain."
                        )
                    )],
                ),
            )

        if not filtered and not skipped_inland:
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="No data collectors requested.")],
                ),
            )


# ═════════════════════════════════════════════════════════════════════
# RISK ASSESSMENT
# ═════════════════════════════════════════════════════════════════════

class RiskAssessmentAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        ocean = ctx.session.state.get(key("ocean"), {})
        weather = ctx.session.state.get(key("weather"), {})
        geofence = ctx.session.state.get(key("geofence"), {})
        hwassa = ctx.session.state.get(key("hwassa"), {})
        cyclone = ctx.session.state.get(key("cyclone"), {})
        tsunami = ctx.session.state.get(key("tsunami"), {})
        currents = ctx.session.state.get(key("currents"), {})

        result = calculate_marine_risk(
            ocean, weather, geofence,
            hwassa=hwassa,
            cyclone=cyclone,
            tsunami=tsunami,
            currents=currents,
        )
        ctx.session.state[key("risk")] = result

        blockers_txt = (
            "\n".join(f"  • {b}" for b in result.get("blockers", []))
            or "  (none)"
        )

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(
                    text=(
                        f"Deterministic marine risk assessment:\n"
                        f"- Risk level: {result['risk_level']}\n"
                        f"- Risk score: {result['risk_score']}/100\n"
                        f"- Blockers:\n{blockers_txt}\n"
                        f"- Reasons: {result['reasons']}\n"
                        f"- Evaluated parameters: "
                        f"{result['evaluated_parameters']}"
                    )
                )],
            ),
        )


# ═════════════════════════════════════════════════════════════════════
# REVIEW LOOP HELPERS
# ═════════════════════════════════════════════════════════════════════

def _review_status(review_text: str) -> str:
    first = (review_text or "").strip().split("\n", 1)[0].strip().upper()
    for word in ("PASS", "RECHECK", "BLOCKED"):
        if first.startswith(word):
            return word
    return "UNKNOWN"


class SimpleQueryGateAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        plan_raw = ctx.session.state.get(key("plan"), "{}")
        try:
            plan = json.loads(plan_raw) if isinstance(plan_raw, str) else plan_raw
        except Exception:
            plan = {}

        domains = (
            (plan.get("domains_needed") if isinstance(plan, dict) else None)
            or (plan.get("domains") if isinstance(plan, dict) else None)
            or []
        )
        if not isinstance(domains, list):
            domains = []

        if len(domains) <= 1:
            ctx.session.state[key("review")] = (
                "PASS\n"
                "Single-domain query — evidence is sufficient."
            )
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(
                        text=(
                            f"Simple query ({len(domains)} domain) — "
                            f"auto-PASS, exiting review loop."
                        )
                    )],
                ),
                actions=__import__(
                    "google.adk.events", fromlist=["EventActions"]
                ).EventActions(escalate=True),
            )
            return

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(
                    text=(
                        f"Complex query ({len(domains)} domains) — "
                        f"proceeding with review loop."
                    )
                )],
            ),
        )


class RecheckAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        review = str(ctx.session.state.get(key("review"), ""))
        status = _review_status(review)
        loc = ctx.session.state.get(key("location"), {})
        marine_loc = ctx.session.state.get(key("marine_location"))

        if status != "RECHECK" or loc.get("status") != "FOUND":
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="No recheck required.")],
                ),
            )
            return

        review_lower = review.lower()
        state_hint = loc.get("admin1") or loc.get("name")
        tasks = []

        # Global-tool rechecks use the raw loc
        if any(w in review_lower for w in ("weather", "wind", "forecast")):
            tasks.append(("weather", asyncio.to_thread(
                get_weather_conditions,
                float(loc["latitude"]), float(loc["longitude"]), 3,
            )))
        if "imd" in review_lower or "port" in review_lower or "bulletin" in review_lower:
            tasks.append(("imd", asyncio.to_thread(
                get_imd_coastal_bulletin,
                float(loc["latitude"]), float(loc["longitude"]), state_hint,
            )))

        # Marine-tool rechecks use the marine loc
        if marine_loc is not None:
            m_lat = float(marine_loc["latitude"])
            m_lon = float(marine_loc["longitude"])
            if any(w in review_lower for w in ("ocean", "wave", "current", "sst", "temperature")):
                tasks.append(("ocean", asyncio.to_thread(
                    get_copernicus_marine_snapshot, m_lat, m_lon,
                )))
            if "geofence" in review_lower or "restriction" in review_lower:
                tasks.append(("geofence", asyncio.to_thread(
                    check_geofence, m_lat, m_lon,
                )))
            if "pfz" in review_lower or "fish" in review_lower:
                tasks.append(("pfz", asyncio.to_thread(
                    get_pfz_advisory, m_lat, m_lon,
                )))
            if "coral" in review_lower or "reef" in review_lower:
                tasks.append(("coral", asyncio.to_thread(
                    get_coral_bleaching_alert, m_lat, m_lon,
                )))

        if "hwassa" in review_lower or "high wave" in review_lower or "swell" in review_lower:
            tasks.append(("hwassa", asyncio.to_thread(
                get_hwassa_alerts, 0, 0, state_hint,
            )))

        results = await asyncio.gather(*[t[1] for t in tasks]) if tasks else []
        for (name, _), value in zip(tasks, results):
            ctx.session.state[key(name)] = value

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(
                    text=f"Recheck completed for {len(results)} domain(s)."
                )],
            ),
        )


class ReviewGateAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        review = str(ctx.session.state.get(key("review"), ""))
        status = _review_status(review)
        should_stop = status in ("PASS", "BLOCKED")

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(
                    text=(
                        f"Review status={status}. "
                        + ("Loop complete." if should_stop
                           else "Requires another evidence pass.")
                    )
                )],
            ),
            actions=__import__("google.adk.events", fromlist=["EventActions"]).EventActions(
                escalate=should_stop
            ),
        )