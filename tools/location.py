"""
Location resolution.

Primary geocoder: Nominatim (OpenStreetMap).
Fallback: Open-Meteo geocoder.
Last resort: small table of Indian state/UT coordinates.

Results are cached in-memory so repeated queries (e.g. follow-ups about
the same place) skip the network round-trip.
"""
from __future__ import annotations

import re
import time

import httpx

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_URL = "https://geocoding-api.open-meteo.com/v1/search"

_USER_AGENT = (
    "ORCA-Marine-Intelligence/1.0 "
    "(academic project; contact: piyushannigeri.vtu@gmail.com)"
)

_COORD_RE = re.compile(
    r"(?P<lat>-?\d+(?:\.\d+)?)\s*[, ]\s*(?P<lon>-?\d+(?:\.\d+)?)"
)

_STATE_COORDS = {
    "goa":                         ("Goa",            15.2993,  73.8243),
    "kerala":                      ("Kerala",          9.9312,  76.2673),
    "karnataka":                   ("Karnataka",      12.9141,  74.8560),
    "tamil nadu":                  ("Tamil Nadu",     13.0827,  80.2707),
    "andhra pradesh":              ("Andhra Pradesh", 17.6868,  83.2185),
    "odisha":                      ("Odisha",         19.8135,  85.8312),
    "orissa":                      ("Odisha",         19.8135,  85.8312),
    "gujarat":                     ("Gujarat",        21.6417,  69.6293),
    "maharashtra":                 ("Maharashtra",    19.0760,  72.8777),
    "west bengal":                 ("West Bengal",    22.5726,  88.3639),
    "lakshadweep":                 ("Lakshadweep",    10.5667,  72.6417),
    "andaman and nicobar islands": ("Port Blair",     11.6234,  92.7265),
    "andaman":                     ("Port Blair",     11.6234,  92.7265),
    "nicobar":                     ("Car Nicobar",     9.1694,  92.7714),
    "puducherry":                  ("Puducherry",     11.9416,  79.8083),
    "pondicherry":                 ("Puducherry",     11.9416,  79.8083),
}

_last_nominatim_call = 0.0

# In-memory cache: place string -> result dict
_GEOCODE_CACHE: dict[str, dict] = {}


def _throttle_nominatim() -> None:
    global _last_nominatim_call
    elapsed = time.time() - _last_nominatim_call
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    _last_nominatim_call = time.time()


def _nominatim(place: str) -> dict | None:
    _throttle_nominatim()
    try:
        r = httpx.get(
            NOMINATIM_URL,
            params={
                "q": place,
                "format": "json",
                "limit": 5,
                "countrycodes": "in",
                "addressdetails": 1,
                "accept-language": "en",
            },
            headers={"User-Agent": _USER_AGENT},
            timeout=20,
            follow_redirects=True,
        )
        r.raise_for_status()
        results = r.json()
    except Exception:
        return None

    if not results:
        return None

    results.sort(key=lambda r: float(r.get("importance", 0) or 0), reverse=True)
    top = results[0]

    addr = top.get("address") or {}
    name = (
        top.get("name")
        or addr.get("state")
        or addr.get("city")
        or addr.get("county")
        or (top.get("display_name") or "").split(",")[0].strip()
    )

    try:
        lat = float(top["lat"])
        lon = float(top["lon"])
    except (KeyError, TypeError, ValueError):
        return None

    return {
        "status": "FOUND",
        "name": name,
        "country": addr.get("country") or "India",
        "country_code": (addr.get("country_code") or "in").upper(),
        "admin1": addr.get("state"),
        "latitude": lat,
        "longitude": lon,
        "timezone": "Asia/Kolkata",
        "source": "Nominatim (OpenStreetMap)",
    }


def _open_meteo(place: str) -> dict | None:
    try:
        r = httpx.get(
            OPEN_METEO_URL,
            params={"name": place, "count": 10, "language": "en", "format": "json"},
            timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("results") or []
    except Exception:
        return None

    if not results:
        return None

    indian = [x for x in results if x.get("country_code") == "IN"]
    pool = indian or results
    top = max(pool, key=lambda x: x.get("population", 0) or 0)

    return {
        "status": "FOUND",
        "name": top.get("name"),
        "country": top.get("country"),
        "country_code": top.get("country_code"),
        "admin1": top.get("admin1"),
        "latitude": top.get("latitude"),
        "longitude": top.get("longitude"),
        "timezone": top.get("timezone"),
        "source": "Open-Meteo Geocoding",
    }


def resolve_location_query(query: str) -> dict:
    """
    Resolve a place name or coordinate string into lat/lon.

    Order:
      1. Explicit coordinates
      2. Nominatim (OpenStreetMap)
      3. Open-Meteo geocoder
      4. State/UT coordinate table
    """
    query = (query or "").strip()
    if not query:
        return {"status": "NOT_FOUND", "error": "No location supplied."}

    # Explicit coordinates
    m = _COORD_RE.search(query)
    if m:
        lat = float(m.group("lat"))
        lon = float(m.group("lon"))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return {
                "status": "FOUND",
                "name": "Explicit coordinates",
                "latitude": lat,
                "longitude": lon,
                "source": "User-provided coordinates",
            }

    # Clean query text
    cleaned = query
    for marker in ["location:", "place:", "from", "near", "off"]:
        cleaned = cleaned.replace(marker, " ")
    place = cleaned[:120].strip()
    if not place:
        return {"status": "NOT_FOUND", "error": "No usable location text."}

    # Cache lookup
    cached = _GEOCODE_CACHE.get(place)
    if cached is not None:
        return cached

    # Geocode
    result = _nominatim(place) or _open_meteo(place)

    if result is None:
        key = place.lower()
        if key in _STATE_COORDS:
            name, lat, lon = _STATE_COORDS[key]
            result = {
                "status": "FOUND",
                "name": name,
                "country": "India",
                "country_code": "IN",
                "admin1": name,
                "latitude": lat,
                "longitude": lon,
                "timezone": "Asia/Kolkata",
                "source": "ORCA state-coordinates table",
            }

    if result is None:
        result = {"status": "NOT_FOUND", "place": place}

    _GEOCODE_CACHE[place] = result
    return result