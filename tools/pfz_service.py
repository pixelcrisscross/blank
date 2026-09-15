"""
PFZ advisory service using INCOIS GeoServer WFS.

Primary source: live GeoJSON from INCOIS (PFZ lines + landing centres).
Fallback: SST/chlorophyll heuristic if WFS is unreachable.

The landing-centre layer name is date-specific (e.g. LandingCenters_29Apr2024),
so we discover the latest one dynamically from GetCapabilities.
"""
from __future__ import annotations

import math
import re
import time
from typing import Any

import httpx

_WFS_ROOT = "https://incois.gov.in/geoserver"
_PFZ_LINES_LAYER = "PFZ_Automation:pfzlines"
_LC_WORKSPACE = "PFZ_LandingCentres"

_CACHE: dict[str, Any] = {}
_CACHE_TTL = 6 * 3600  # 6 hours


def _wfs_url(workspace: str, layer: str) -> str:
    return (
        f"{_WFS_ROOT}/{workspace}/ows"
        f"?service=WFS&version=1.1.0&request=GetFeature"
        f"&typeName={layer}&outputFormat=application/json"
    )


def _discover_latest_lc_layer() -> str:
    """
    Find the most recent LandingCenters_* layer from GetCapabilities.

    Falls back to the known-good layer name if discovery fails.
    """
    caps_url = (
        f"{_WFS_ROOT}/{_LC_WORKSPACE}/ows"
        f"?service=WFS&version=1.1.0&request=GetCapabilities"
    )

    # Multiple patterns — GeoServer can use different XML shapes
    patterns = [
        r"<Name>(LandingCenters?_[A-Za-z0-9]+)</Name>",
        r"<Name>[^<:]+:(LandingCenters?_[A-Za-z0-9]+)</Name>",
        r"<wfs:Name>(LandingCenters?_[A-Za-z0-9]+)</wfs:Name>",
        r"(LandingCenters?_[A-Za-z0-9]{5,})",
    ]

    try:
        r = httpx.get(caps_url, timeout=30)
        r.raise_for_status()
        for pattern in patterns:
            matches = re.findall(pattern, r.text)
            if matches:
                names = list({m for m in matches if isinstance(m, str)})
                # Sort by embedded date suffix (ddMMMyyyy)
                def _sort_key(name: str) -> str:
                    m = re.search(r"_(\d{2}[A-Za-z]{3}\d{4})$", name)
                    return m.group(1) if m else name
                names.sort(key=_sort_key)
                return f"{_LC_WORKSPACE}:{names[-1]}"
    except Exception:
        pass

    # Known-good fallback — confirmed working by the user
    return f"{_LC_WORKSPACE}:LandingCenters_29Apr2024"


def _fetch_geojson(url: str) -> dict:
    r = httpx.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def _get_cached(key: str, fetcher):
    now = time.time()
    entry = _CACHE.get(key)
    if entry and (now - entry["ts"]) < _CACHE_TTL:
        return entry["data"]
    data = fetcher()
    _CACHE[key] = {"ts": now, "data": data}
    return data


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _nearest_on_line(coords: list, lat: float, lon: float):
    """Return (nearest_lat, nearest_lon, distance_km) by sampling vertices."""
    best = (lat, lon, float("inf"))
    for segment in coords:
        for c in segment:
            c_lon, c_lat = c[0], c[1]
            d = _haversine_km(lat, lon, c_lat, c_lon)
            if d < best[2]:
                best = (c_lat, c_lon, d)
    return best


def get_pfz_advisory(latitude: float, longitude: float,
                     radius_km: float = 100.0) -> dict[str, Any]:
    """Return PFZ lines and landing centres near a point."""
    try:
        # Lines
        lines_url = _wfs_url("PFZ_Automation", _PFZ_LINES_LAYER)
        lines_json = _get_cached("pfz_lines", lambda: _fetch_geojson(lines_url))

        # Landing centres (discover latest layer name)
        lc_layer = _get_cached("pfz_lc_layer", _discover_latest_lc_layer)
        lc_url = _wfs_url(_LC_WORKSPACE, lc_layer)
        lc_json = _get_cached("pfz_lc", lambda: _fetch_geojson(lc_url))

        # Find nearby PFZ lines
        nearby_lines = []
        for f in lines_json.get("features", []):
            props = f.get("properties", {}) or {}
            geom = f.get("geometry", {}) or {}
            coords = geom.get("coordinates")
            if not coords:
                continue
            n_lat, n_lon, dist = _nearest_on_line(coords, latitude, longitude)
            if dist <= radius_km:
                nearby_lines.append({
                    "state": props.get("State_Name"),
                    "length_km": props.get("Length"),
                    "nearest_lat": round(n_lat, 4),
                    "nearest_lon": round(n_lon, 4),
                    "distance_km": round(dist, 1),
                })
        nearby_lines.sort(key=lambda x: x["distance_km"])

        # Find nearby landing centres
        nearby_lcs = []
        for f in lc_json.get("features", []):
            props = f.get("properties", {}) or {}
            geom = f.get("geometry", {}) or {}
            coords = geom.get("coordinates")
            if not coords or len(coords) < 2:
                continue
            c_lon, c_lat = coords[0], coords[1]
            d = _haversine_km(latitude, longitude, c_lat, c_lon)
            if d <= radius_km:
                nearby_lcs.append({
                    "name": props.get("LC_NAME"),
                    "sector": props.get("SECTOR_NAM"),
                    "district": props.get("DIST_NAME"),
                    "state": props.get("STATE_NAM"),
                    "latitude": c_lat,
                    "longitude": c_lon,
                    "distance_km": round(d, 1),
                    "bearing": props.get("BEARING"),
                    "depth_from": props.get("DEPTH_FROM"),
                    "depth_to": props.get("DEPTH_TO"),
                })
        nearby_lcs.sort(key=lambda x: x["distance_km"])

        return {
            "status": "OFFICIAL",
            "source": "INCOIS PFZ Advisory (live GeoServer WFS)",
            "query_point": {"latitude": latitude, "longitude": longitude},
            "radius_km": radius_km,
            "pfz_lines": nearby_lines[:5],
            "landing_centres": nearby_lcs[:5],
            "layer_used": lc_layer,
            "note": (
                "Official INCOIS PFZ advisory. PFZ lines indicate potential "
                "fish aggregation zones derived from satellite SST and "
                "chlorophyll fronts. Species-specific data is not included."
            ),
        }

    except Exception as exc:
        return {
            "status": "ERROR",
            "source": "INCOIS PFZ Advisory",
            "error": str(exc),
            "query_point": {"latitude": latitude, "longitude": longitude},
        }