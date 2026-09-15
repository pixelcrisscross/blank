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
from tools.coastal_check import is_coastal
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
            "could not be extracted. IMD may have changed the layout."
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
        f"- Synoptic situation: {r.get('synoptic_situation')}\n"
        f"- Wind: {r.get('wind')}\n"
        f"- Weather: {r.get('weather')}\n"
        f"- Visibility: {r.get('visibility')}\n"
        f"- Sea condition: {r.get('sea_condition')}\n"
        f"- PORT SIGNAL: {r.get('port_signal')} "
        f"(active={r.get('port_signal_active')})\n"
        f"- STORM SURGE / TIDAL WARNING: "
        f"{r.get('storm_surge_warning')} "
        f"(active={r.get('storm_surge_active')})\n"
        f"- TIDAL WAVE: {r.get('tidal_wave')} "
        f"(active={r.get('tidal_wave_active')})\n"
        f"- Attribution: {r.get('attribution')}"
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
}


# ═════════════════════════════════════════════════════════════════════
# LOCATION RESOLUTION
# ═════════════════════════════════════════════════════════════════════

class ResolveLocationAgent(BaseAgent):
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
            result = await asyncio.to_thread(
                resolve_location_query, location_hint
            )
        else:
            result = {"status": "NOT_FOUND", "error": "No location in plan."}

        ctx.session.state[key("location")] = result

        if result.get("status") == "FOUND":
            text = (
                f"Location resolved: {result.get('name')}, "
                f"{result.get('admin1')}, {result.get('country')}. "
                f"Latitude={result.get('latitude')}, "
                f"Longitude={result.get('longitude')}."
            )
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
    """Emits ORCA's capability list when the plan intent is 'meta'."""

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
- Coral reefs (NOAA Coral Reef Watch): bleaching alerts for Indian reef regions
- Tides (local harmonic prediction): high/low extremes for supported ports
- Bioluminescence (heuristic): Noctiluca forecast for tourism
- Algal blooms (Copernicus chlorophyll): bloom risk classification
- IMD Coastal Bulletin: official port signals, storm surge warnings
- Safety (deterministic): risk score from ocean + weather + geofence
- Location resolution: place names or explicit coordinates

Note: marine domains require a coastal location. Inland locations return
weather only.

Example questions:
- "What is the SST off Goa right now?"
- "Where can I fish near Kochi tomorrow?"
- "Are there coral reefs near Lakshadweep?"
- "Is tonight good for bioluminescence in Goa?"
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
    Read domains_needed from the plan and run only the required collectors
    concurrently. Skips marine domains when the location is inland.
    Emits per-domain summaries so downstream LLM agents see real numbers.
    """

    MARINE_DOMAINS: ClassVar[set[str]] = {
        "ocean", "pfz", "coral", "tides", "biolum", "algal_bloom",
    }
    GLOBAL_DOMAINS: ClassVar[set[str]] = {
        "weather", "geofence", "imd",
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

        domains = (
            plan.get("domains_needed")
            or plan.get("domains")
            or []
        )
        if not isinstance(domains, list):
            domains = []
        if not domains:
            domains = list(_SUMMARISERS.keys())

        loc = ctx.session.state.get(key("location"), {})
        loc_ok = loc.get("status") == "FOUND"

        inland = False
        coast_info = None
        if loc_ok:
            try:
                coast_info = await asyncio.to_thread(
                    is_coastal,
                    float(loc["latitude"]),
                    float(loc["longitude"]),
                )
                inland = not coast_info["coastal"]
            except Exception:
                inland = False

        filtered = []
        skipped_inland = []
        for d in domains:
            if inland and d in self.MARINE_DOMAINS:
                skipped_inland.append(d)
                ctx.session.state[key(d)] = {
                    "status": "BLOCKED_INLAND",
                    "reason": (
                        f"Location is ~{coast_info['distance_km']} km from "
                        f"the nearest coastline. Marine data is not "
                        f"meaningful here."
                    ),
                }
            else:
                filtered.append(d)

        async def run(name: str, fn):
            if not loc_ok:
                ctx.session.state[key(name)] = {
                    "status": "BLOCKED",
                    "reason": "Location unavailable.",
                }
                return
            try:
                result = await asyncio.to_thread(
                    fn, float(loc["latitude"]), float(loc["longitude"])
                )
                ctx.session.state[key(name)] = result
            except Exception as exc:
                ctx.session.state[key(name)] = {
                    "status": "ERROR",
                    "error": str(exc),
                }

        state_hint = loc.get("admin1") or loc.get("name")

        dispatcher = {
            "ocean":       ("ocean",       get_copernicus_marine_snapshot),
            "weather":     ("weather",     lambda a, b: get_weather_conditions(a, b, 3)),
            "geofence":    ("geofence",    check_geofence),
            "pfz":         ("pfz",         get_pfz_advisory),
            "coral":       ("coral",       get_coral_bleaching_alert),
            "tides":       ("tides",       lambda a, b: get_tide_prediction(
                                                loc.get("name"))),
            "biolum":      ("biolum",      get_bioluminescence_forecast),
            "algal_bloom": ("algal_bloom", get_algal_bloom_risk),
            "imd":         ("imd",         lambda a, b: get_imd_coastal_bulletin(
                                                a, b, state_hint=state_hint
                                            )),
        }

        collected = [d for d in filtered if d in dispatcher]

        if collected:
            await asyncio.gather(
                *[run(dispatcher[d][0], dispatcher[d][1]) for d in collected],
                return_exceptions=True,
            )

        for d in collected:
            state_name = dispatcher[d][0]
            result = ctx.session.state.get(key(state_name), {}) or {}
            formatter = _SUMMARISERS.get(state_name)
            text = formatter(result) if formatter else _json_block(result)
            yield Event(
                invocation_id=ctx.invocation_id,
                author=f"{self.name}_{state_name}",
                content=types.Content(role="model", parts=[types.Part(text=text)]),
            )

        for d in skipped_inland:
            state_name = d
            yield Event(
                invocation_id=ctx.invocation_id,
                author=f"{self.name}_{state_name}",
                content=types.Content(
                    role="model",
                    parts=[types.Part(
                        text=(
                            f"{state_name.upper()} DATA: SKIPPED — location is "
                            f"inland (~{coast_info['distance_km']} km from the "
                            f"nearest coast). No data was collected. Do not "
                            f"provide any values for this domain."
                        )
                    )],
                ),
            )

        if skipped_inland:
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(
                        text=(
                            f"Skipped marine domains {skipped_inland}: "
                            f"location is inland "
                            f"(~{coast_info['distance_km']} km from the "
                            f"nearest coast). Only weather and geofence apply."
                        )
                    )],
                ),
            )

        if not collected and not skipped_inland:
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
        result = calculate_marine_risk(ocean, weather, geofence)
        ctx.session.state[key("risk")] = result
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
    """
    Inside the review loop, immediately exits for single-domain queries.

    Runs first inside the LoopAgent. For simple queries, it writes
    PASS to the review state and escalates, which causes LoopAgent to
    stop before peer_review_agent, review_agent, etc. run.

    For complex queries (2+ domains), it does nothing and the review
    loop proceeds normally.
    """

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
                "Single-domain query — evidence is sufficient for a "
                "direct answer. No peer review needed."
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
        lat = float(loc["latitude"])
        lon = float(loc["longitude"])
        state_hint = loc.get("admin1") or loc.get("name")
        tasks = []

        if any(w in review_lower for w in ("weather", "wind", "forecast")):
            tasks.append(("weather", asyncio.to_thread(
                get_weather_conditions, lat, lon, 3,
            )))
        if any(w in review_lower for w in ("ocean", "wave", "current", "sst", "temperature")):
            tasks.append(("ocean", asyncio.to_thread(
                get_copernicus_marine_snapshot, lat, lon,
            )))
        if "geofence" in review_lower or "restriction" in review_lower:
            tasks.append(("geofence", asyncio.to_thread(
                check_geofence, lat, lon,
            )))
        if "pfz" in review_lower or "fish" in review_lower:
            tasks.append(("pfz", asyncio.to_thread(
                get_pfz_advisory, lat, lon,
            )))
        if "coral" in review_lower or "reef" in review_lower:
            tasks.append(("coral", asyncio.to_thread(
                get_coral_bleaching_alert, lat, lon,
            )))
        if "imd" in review_lower or "port" in review_lower or "bulletin" in review_lower:
            tasks.append(("imd", asyncio.to_thread(
                get_imd_coastal_bulletin, lat, lon, state_hint,
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