"""
Pipecat tool wrappers around the ORCA marine-intelligence tools.
"""

from __future__ import annotations

import asyncio

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from tools.copernicus_service import get_copernicus_marine_snapshot
from tools.weather_service import get_weather_conditions
from tools.location import resolve_location_query
from tools.geofence import check_geofence


async def orca_resolve_location(params: FunctionCallParams) -> None:
    args = params.arguments or {}
    query = str(args.get("query", "")).strip()
    if not query:
        await params.result_callback({"status": "NOT_FOUND", "error": "No query supplied."})
        return

    logger.info(f"[tool] orca_resolve_location({query!r})")
    try:
        result = await asyncio.to_thread(resolve_location_query, query)
        await params.result_callback(result)
    except Exception as exc:
        logger.exception("[tool] orca_resolve_location failed")
        await params.result_callback({"status": "ERROR", "error": str(exc)})


async def orca_marine_snapshot(params: FunctionCallParams) -> None:
    args = params.arguments or {}
    try:
        latitude = float(args["latitude"])
        longitude = float(args["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        await params.result_callback({"status": "ERROR", "error": f"Invalid coordinates: {exc}"})
        return

    logger.info(f"[tool] orca_marine_snapshot({latitude}, {longitude})")
    try:
        raw = await asyncio.to_thread(get_copernicus_marine_snapshot, latitude, longitude)
        obs = raw.get("observations", {}) or {}
        keep = {
            "parameter", "value", "unit", "status", "observation_time",
            "speed_ms", "direction_deg",
            "significant_wave_height_m", "mean_wave_period_s", "wave_direction_deg",
        }
        compact = {
            "status": raw.get("status"),
            "location": raw.get("location"),
            "retrieved_at": raw.get("retrieved_at"),
            "observations": {
                name: {k: v for k, v in payload.items() if k in keep}
                for name, payload in obs.items()
            },
            "warnings": raw.get("warnings", []),
        }
        await params.result_callback(compact)
    except Exception as exc:
        logger.exception("[tool] orca_marine_snapshot failed")
        await params.result_callback({"status": "ERROR", "error": str(exc)})


async def orca_weather(params: FunctionCallParams) -> None:
    args = params.arguments or {}
    try:
        latitude = float(args["latitude"])
        longitude = float(args["longitude"])
        days = int(args.get("days", 3))
    except (KeyError, TypeError, ValueError) as exc:
        await params.result_callback({"status": "ERROR", "error": f"Invalid arguments: {exc}"})
        return

    logger.info(f"[tool] orca_weather({latitude}, {longitude}, days={days})")
    try:
        result = await asyncio.to_thread(get_weather_conditions, latitude, longitude, days)
        hourly = result.get("hourly", {}) or {}
        compact_hourly = {k: (v[:6] if isinstance(v, list) else v) for k, v in hourly.items()}
        await params.result_callback({
            "status": result.get("status"),
            "current": result.get("current", {}),
            "current_units": result.get("current_units", {}),
            "hourly_summary": compact_hourly,
        })
    except Exception as exc:
        logger.exception("[tool] orca_weather failed")
        await params.result_callback({"status": "ERROR", "error": str(exc)})


async def orca_geofence(params: FunctionCallParams) -> None:
    args = params.arguments or {}
    try:
        latitude = float(args["latitude"])
        longitude = float(args["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        await params.result_callback({"status": "ERROR", "error": str(exc)})
        return

    logger.info(f"[tool] orca_geofence({latitude}, {longitude})")
    try:
        result = await asyncio.to_thread(check_geofence, latitude, longitude)
        await params.result_callback(result)
    except Exception as exc:
        logger.exception("[tool] orca_geofence failed")
        await params.result_callback({"status": "ERROR", "error": str(exc)})


FUNCTION_HANDLERS = {
    "orca_resolve_location": orca_resolve_location,
    "orca_marine_snapshot":  orca_marine_snapshot,
    "orca_weather":          orca_weather,
    "orca_geofence":         orca_geofence,
}


ORCA_TOOLS_SCHEMA = ToolsSchema(standard_tools=[
    FunctionSchema(
        name="orca_resolve_location",
        description=(
            "Resolve a place name, port, or coordinate string into latitude "
            "and longitude. Call this FIRST when the user mentions a "
            "location by name."
        ),
        properties={"query": {"type": "string", "description": "Place name or coordinate string."}},
        required=["query"],
    ),
    FunctionSchema(
        name="orca_marine_snapshot",
        description=(
            "Get current sea surface temperature, ocean currents, and wave "
            "conditions from Copernicus Marine at a given latitude/longitude."
        ),
        properties={"latitude": {"type": "number"}, "longitude": {"type": "number"}},
        required=["latitude", "longitude"],
    ),
    FunctionSchema(
        name="orca_weather",
        description=(
            "Get current and forecast marine weather (wind, gusts, "
            "precipitation, visibility) at a location."
        ),
        properties={
            "latitude": {"type": "number"},
            "longitude": {"type": "number"},
            "days": {"type": "integer", "description": "Forecast days (1-7). Default 3."},
        },
        required=["latitude", "longitude"],
    ),
    FunctionSchema(
        name="orca_geofence",
        description="Check whether a location lies inside a restricted or protected marine zone.",
        properties={"latitude": {"type": "number"}, "longitude": {"type": "number"}},
        required=["latitude", "longitude"],
    ),
])