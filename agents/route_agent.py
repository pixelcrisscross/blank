"""
Route Safety Agent for ORCA.

Activates only when the planner sets intent = "route".

Pipeline:
  1. Read start_coords and end_coords from plan.
  2. Sample great-circle waypoints (every ~25 km, up to 10 points).
  3. For each waypoint: check weather + geofence in parallel.
  4. Produce a structured route risk summary.
  5. Store result in session.state["orca_route_data"].

Competition brief requirement:
  "What is the safest route for a fishing vessel considering weather
   and sea-state conditions?"
  "Assisting with route optimization, safe navigation, and operational
   planning based on prevailing and forecast marine conditions."
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from state.schemas import key
from tools.route_service import (
    sample_route_waypoints,
    summarise_route_risk,
    _weather_risk_label,
)
from tools.weather_service import get_weather_conditions
from tools.geofence import check_geofence


def _parse_coords(raw) -> tuple[float, float] | None:
    """Parse {latitude, longitude} dict or [lat, lon] list."""
    if isinstance(raw, dict):
        try:
            return (float(raw["latitude"]), float(raw["longitude"]))
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return (float(raw[0]), float(raw[1]))
        except (TypeError, ValueError):
            return None
    return None


async def _evaluate_waypoint(wp: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate weather and geofence risk at a single waypoint.
    Returns the waypoint dict augmented with risk fields.
    """
    lat, lon = wp["latitude"], wp["longitude"]
    result = dict(wp)

    # Run weather + geofence concurrently
    weather_task = asyncio.to_thread(get_weather_conditions, lat, lon, 1)
    geofence_task = asyncio.to_thread(check_geofence, lat, lon)

    weather_r, geofence_r = await asyncio.gather(
        weather_task, geofence_task, return_exceptions=True
    )

    # ── Weather ──────────────────────────────────────────────────────
    if isinstance(weather_r, Exception):
        result["weather_error"] = str(weather_r)
        result["risk_level"] = "UNKNOWN"
        result["risk_reason"] = "Weather data unavailable."
    else:
        cur = (weather_r.get("current") or {}) if isinstance(weather_r, dict) else {}
        wind_speed = cur.get("wind_speed_10m")
        wind_gust = cur.get("wind_gusts_10m")
        weather_code = cur.get("weather_code")
        precipitation = cur.get("precipitation")

        risk_label = _weather_risk_label(
            wind_speed, wind_gust, weather_code, precipitation
        )
        result["risk_level"] = risk_label
        result["wind_speed_kmh"] = wind_speed
        result["wind_gust_kmh"] = wind_gust
        result["weather_code"] = weather_code
        result["precipitation_mm"] = precipitation

        reasons = []
        if wind_speed is not None and wind_speed >= 30:
            reasons.append(f"wind {wind_speed:.0f} km/h")
        if wind_gust is not None and wind_gust >= 45:
            reasons.append(f"gusts {wind_gust:.0f} km/h")
        if weather_code is not None and weather_code >= 80:
            reasons.append(f"severe weather code {int(weather_code)}")
        if precipitation is not None and precipitation >= 10:
            reasons.append(f"heavy precipitation {precipitation:.1f} mm")
        result["risk_reason"] = (
            "; ".join(reasons) if reasons else "Conditions within normal limits."
        )

    # ── Geofence ─────────────────────────────────────────────────────
    if isinstance(geofence_r, Exception):
        result["geofence_error"] = str(geofence_r)
        result["geofence_inside"] = False
        result["geofence_proximity"] = False
    else:
        g = geofence_r if isinstance(geofence_r, dict) else {}
        result["geofence_inside"] = bool(g.get("inside_restricted_zone"))
        result["geofence_proximity"] = bool(g.get("proximity_warning"))
        result["geofence_matches"] = g.get("matches", [])
        result["geofence_proximity_warnings"] = g.get("proximity_warnings", [])

        # Upgrade risk if inside a restricted zone
        if result["geofence_inside"]:
            result["risk_level"] = "BLOCKED"
            zone_names = [m.get("name", "zone") for m in result["geofence_matches"]]
            result["risk_reason"] = (
                (result.get("risk_reason") or "")
                + f" | GEOFENCE: entering restricted zone ({', '.join(zone_names)})."
            )
        elif result["geofence_proximity"]:
            if result.get("risk_level") == "LOW":
                result["risk_level"] = "MODERATE"
            result["risk_reason"] = (
                (result.get("risk_reason") or "")
                + " | PROXIMITY: approaching maritime boundary or protected area."
            )

    return result


class RouteSafetyAgent(BaseAgent):
    """
    Evaluates marine route safety waypoint by waypoint and produces a
    structured risk summary.

    Only activates when plan.intent == "route". For all other intents
    it yields a single skip event.
    """

    async def _run_async_impl(self, ctx: InvocationContext):
        from agents.data_agents import _parse_plan_json

        plan_raw = ctx.session.state.get(key("plan"), "{}")
        plan = _parse_plan_json(plan_raw)

        if plan.get("intent") != "route":
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="Route agent: not a route query — skipping.")],
                ),
            )
            return

        # ── Extract coordinates ───────────────────────────────────────
        start = _parse_coords(plan.get("start_coords"))
        end = _parse_coords(plan.get("end_coords"))

        if start is None or end is None:
            error_msg = (
                "Route request received but start or end coordinates are "
                "missing from the plan. Please provide both start and end "
                "locations."
            )
            ctx.session.state[key("route")] = {
                "status": "ERROR",
                "error": error_msg,
            }
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=f"Route agent: {error_msg}")],
                ),
            )
            return

        start_lat, start_lon = start
        end_lat, end_lon = end

        # ── Sample waypoints ─────────────────────────────────────────
        waypoints = sample_route_waypoints(
            start_lat, start_lon, end_lat, end_lon,
            step_km=25.0,
            max_points=10,
        )

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(
                    text=(
                        f"Route agent: evaluating {len(waypoints)} waypoints "
                        f"from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})."
                    )
                )],
            ),
        )

        # ── Evaluate each waypoint in parallel ──────────────────────
        evaluated = await asyncio.gather(
            *[_evaluate_waypoint(wp) for wp in waypoints],
            return_exceptions=True,
        )

        # Replace exceptions with error stubs
        clean = []
        for i, res in enumerate(evaluated):
            if isinstance(res, Exception):
                wp = waypoints[i]
                clean.append({
                    **wp,
                    "risk_level": "UNKNOWN",
                    "risk_reason": f"Evaluation error: {res}",
                    "geofence_inside": False,
                    "geofence_proximity": False,
                })
            else:
                clean.append(res)

        # ── Summarise and store ──────────────────────────────────────
        summary = summarise_route_risk(clean)
        summary["start"] = {"latitude": start_lat, "longitude": start_lon}
        summary["end"] = {"latitude": end_lat, "longitude": end_lon}
        summary["waypoints"] = clean
        summary["status"] = "OK"

        ctx.session.state[key("route")] = summary

        # ── Emit structured report ────────────────────────────────────
        risk_order = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "BLOCKED": 3, "UNKNOWN": -1}
        hazard_count = len(summary.get("hazard_segments", []))
        overall = summary["overall_risk"]

        lines = [
            f"Route safety assessment — overall risk: {overall}",
            f"Waypoints checked: {len(clean)} | Hazard segments: {hazard_count}",
        ]
        for seg in summary.get("hazard_segments", [])[:5]:
            lines.append(
                f"  ⚠ {seg['waypoint']} ({seg['distance_km']} km): "
                f"{seg['risk_level']} — {seg.get('risk_reason', '')}"
            )
        lines.append(summary["note"])

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text="\n".join(lines))],
            ),
        )
