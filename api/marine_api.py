from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from tools.copernicus_service import (
    get_copernicus_marine_snapshot,
)
from tools.copernicus_grid import (
    get_copernicus_grid,
)
from tools.copernicus_wmts import (
    discover_layers,
    discover_orca_layers,
    get_capabilities_xml,
    build_orca_tile_template,
    build_orca_legend_url,
    get_orca_wmts_layer,
)
from tools.pfz_service import (
    get_pfz_advisory,
    _fetch_geojson,
    _wfs_url,
    _PFZ_LINES_LAYER,
    _LC_WORKSPACE,
    _discover_latest_lc_layer,
)
from tools.coral_service import get_coral_bleaching_alert
from tools.tide_service import get_tide_prediction
from tools.bioluminescence import get_bioluminescence_forecast
from tools.algal_bloom import get_algal_bloom_risk
from tools.geofence import check_geofence
from tools.route_service import sample_route_waypoints, summarise_route_risk, _weather_risk_label
from tools.weather_service import get_weather_conditions


app = FastAPI(
    title="ORCA Marine Data Gateway",
    version="0.4.0",
    description=(
        "Local gateway for Copernicus Marine numerical data, WMTS map "
        "layers, INCOIS PFZ advisories, NOAA coral alerts, local tide "
        "predictions, bioluminescence forecasts, algal-bloom risk, "
        "maritime geofencing, and route safety assessment."
    ),
)


_DEFAULT_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:7861",
    "http://localhost:7861",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

_cors_env = os.getenv("ORCA_CORS_ORIGINS", "")
_cors_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else _DEFAULT_ORIGINS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═════════════════════════════════════════════════════════════════════════════
# ROOT / HEALTH
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "ORCA Marine Data Gateway",
        "version": "0.4.0",
        "status": "online",
        "endpoints": {
            "health": "/health",

            # Copernicus point + grid
            "point": "/marine/point",
            "grid": "/marine/grid",

            # WMTS map layers
            "capabilities": "/marine/map/capabilities",
            "layers": "/marine/map/layers",
            "orca_layers": "/marine/map/orca-layers",
            "map_config": "/marine/map/config/{parameter}",

            # Biological + environmental domains
            "pfz": "/marine/pfz",
            "pfz_geojson": "/marine/pfz/geojson",
            "coral": "/marine/coral",
            "tides": "/marine/tides",
            "bioluminescence": "/marine/bioluminescence",
            "algal_bloom": "/marine/algal-bloom",

            # Phase 4 + Phase 6: Geofencing and route safety
            "geofence": "/marine/geofence",
            "route": "/marine/route",
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "orca-marine-data-gateway"}


# ═════════════════════════════════════════════════════════════════════════════
# COPERNICUS POINT + GRID
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/marine/point")
def marine_point(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
) -> dict[str, Any]:
    try:
        return get_copernicus_marine_snapshot(
            latitude=latitude,
            longitude=longitude,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Copernicus data retrieval failed",
                "message": str(exc),
            },
        )


@app.get("/marine/grid")
def marine_grid(
    parameter: str = Query(
        ...,
        description="Supported values: sst, currents, waves",
    ),
    minimum_latitude: float = Query(..., ge=-90, le=90),
    maximum_latitude: float = Query(..., ge=-90, le=90),
    minimum_longitude: float = Query(..., ge=-180, le=180),
    maximum_longitude: float = Query(..., ge=-180, le=180),
) -> dict[str, Any]:
    if minimum_latitude >= maximum_latitude:
        raise HTTPException(
            status_code=400,
            detail="minimum_latitude must be less than maximum_latitude",
        )
    if minimum_longitude >= maximum_longitude:
        raise HTTPException(
            status_code=400,
            detail="minimum_longitude must be less than maximum_longitude",
        )
    try:
        return get_copernicus_grid(
            parameter,
            minimum_latitude=minimum_latitude,
            maximum_latitude=maximum_latitude,
            minimum_longitude=minimum_longitude,
            maximum_longitude=maximum_longitude,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Copernicus grid retrieval failed",
                "message": str(exc),
            },
        )


# ═════════════════════════════════════════════════════════════════════════════
# WMTS MAP LAYERS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/marine/map/capabilities", response_class=Response)
def wmts_capabilities() -> Response:
    try:
        xml = get_capabilities_xml()
        return Response(content=xml, media_type="application/xml")
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Copernicus WMTS capabilities failed",
                "message": str(exc),
            },
        )


@app.get("/marine/map/layers")
def wmts_layers() -> dict[str, Any]:
    try:
        layers = discover_layers()
        return {
            "service": "Copernicus Marine WMTS",
            "endpoint": "https://wmts.marine.copernicus.eu/teroWmts",
            "count": len(layers),
            "layers": layers,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "WMTS layer discovery failed",
                "message": str(exc),
            },
        )


@app.get("/marine/map/orca-layers")
def orca_map_layers() -> dict[str, Any]:
    try:
        layers = discover_layers()
        categorized = discover_orca_layers(layers)
        return {
            "service": "Copernicus Marine WMTS",
            "endpoint": "https://wmts.marine.copernicus.eu/teroWmts",
            "count": len(layers),
            "categories": categorized,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Copernicus WMTS discovery failed",
                "message": str(exc),
            },
        )


@app.get("/marine/map/config/{parameter}")
def map_config(parameter: str) -> dict[str, Any]:
    parameter = parameter.lower().strip()
    try:
        definition = get_orca_wmts_layer(parameter)
        tile_url = build_orca_tile_template(parameter)
        legend_url = build_orca_legend_url(parameter)
        return {
            "parameter": parameter,
            "title": definition["title"],
            "unit": definition["unit"],
            "layer": definition["layer"],
            "projection": "EPSG:3857",
            "tile_matrix_set": definition["matrix_set"],
            "tile_size": 256,
            "tile_url": tile_url,
            "legend_url": legend_url,
            "style": definition["style"],
            "source": "Copernicus Marine WMTS",
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Map configuration failed",
                "message": str(exc),
            },
        )


# ═════════════════════════════════════════════════════════════════════════════
# PFZ (INCOIS)
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/marine/pfz")
def pfz(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(100.0, gt=0, le=500),
) -> dict[str, Any]:
    try:
        return get_pfz_advisory(latitude, longitude, radius_km)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "PFZ advisory fetch failed",
                "message": str(exc),
            },
        )


@app.get("/marine/pfz/geojson")
def pfz_geojson() -> dict[str, Any]:
    """Proxy raw INCOIS GeoJSON for the web map (avoids browser CORS)."""
    try:
        lc_layer = _discover_latest_lc_layer()
        return {
            "pfz_lines": _fetch_geojson(
                _wfs_url("PFZ_Automation", _PFZ_LINES_LAYER)
            ),
            "landing_centres": _fetch_geojson(
                _wfs_url(_LC_WORKSPACE, lc_layer)
            ),
            "layer_used": lc_layer,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "PFZ GeoJSON proxy failed",
                "message": str(exc),
            },
        )


# ═════════════════════════════════════════════════════════════════════════════
# CORAL (NOAA CRW)
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/marine/coral")
def coral(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
) -> dict[str, Any]:
    try:
        return get_coral_bleaching_alert(latitude, longitude)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Coral bleaching alert fetch failed",
                "message": str(exc),
            },
        )


# ═════════════════════════════════════════════════════════════════════════════
# TIDES (local)
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/marine/tides")
def tides(
    port: str | None = Query(
        None,
        description=(
            "Optional port name (e.g. kochi, mumbai, chennai, "
            "visakhapatnam, goa). If omitted, returns moon-regime only."
        ),
    ),
) -> dict[str, Any]:
    try:
        return get_tide_prediction(port)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Tide prediction failed",
                "message": str(exc),
            },
        )


# ═════════════════════════════════════════════════════════════════════════════
# BIOLUMINESCENCE
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/marine/bioluminescence")
def bioluminescence(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
) -> dict[str, Any]:
    try:
        return get_bioluminescence_forecast(latitude, longitude)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Bioluminescence forecast failed",
                "message": str(exc),
            },
        )


# ═════════════════════════════════════════════════════════════════════════════
# ALGAL BLOOM
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/marine/algal-bloom")
def algal_bloom(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
) -> dict[str, Any]:
    try:
        return get_algal_bloom_risk(latitude, longitude)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Algal bloom risk fetch failed",
                "message": str(exc),
            },
        )


# ═════════════════════════════════════════════════════════════════════════════
# GEOFENCE (Phase 4)
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/marine/geofence")
def geofence(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
) -> dict[str, Any]:
    """
    Check whether a location is inside or near a maritime boundary zone.

    Zones include: IMBL (India–Sri Lanka), Marine Protected Areas,
    EEZ approach buffers, and ecologically sensitive zones.
    """
    try:
        return check_geofence(latitude, longitude)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Geofence check failed",
                "message": str(exc),
            },
        )


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE SAFETY (Phase 6)
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/marine/route")
def marine_route(
    start_latitude: float = Query(..., ge=-90, le=90),
    start_longitude: float = Query(..., ge=-180, le=180),
    end_latitude: float = Query(..., ge=-90, le=90),
    end_longitude: float = Query(..., ge=-180, le=180),
    step_km: float = Query(25.0, gt=5, le=200,
                           description="Waypoint spacing in km (default 25)."),
) -> dict[str, Any]:
    """
    Evaluate route safety for a fishing vessel or coastal operator.

    Samples waypoints every step_km along the great-circle route,
    evaluates weather and geofence risk at each, and returns an
    overall risk assessment with hazard segment details.
    """
    if start_latitude == end_latitude and start_longitude == end_longitude:
        raise HTTPException(
            status_code=400,
            detail="Start and end points must be different.",
        )
    try:
        waypoints = sample_route_waypoints(
            start_latitude, start_longitude,
            end_latitude, end_longitude,
            step_km=step_km,
            max_points=10,
        )

        # Evaluate each waypoint: weather + geofence
        import asyncio

        async def _eval_all():
            from agents.route_agent import _evaluate_waypoint
            results = await asyncio.gather(
                *[_evaluate_waypoint(wp) for wp in waypoints],
                return_exceptions=True,
            )
            clean = []
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    clean.append({
                        **waypoints[i],
                        "risk_level": "UNKNOWN",
                        "risk_reason": f"Evaluation error: {res}",
                        "geofence_inside": False,
                        "geofence_proximity": False,
                    })
                else:
                    clean.append(res)
            return clean

        evaluated = asyncio.run(_eval_all())
        summary = summarise_route_risk(evaluated)
        summary["start"] = {
            "latitude": start_latitude, "longitude": start_longitude
        }
        summary["end"] = {
            "latitude": end_latitude, "longitude": end_longitude
        }
        summary["waypoints"] = evaluated
        summary["status"] = "OK"
        return summary

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Route safety assessment failed",
                "message": str(exc),
            },
        )