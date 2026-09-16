"""
Location resolution with a stronger fallback chain.

Primary geocoder: Nominatim (OpenStreetMap).
Fallbacks (in order):
  1. Nominatim with India filter
  2. Nominatim without filter (catches localities missing from the
     India-only index)
  3. Nominatim with "India" appended to the query
  4. Open-Meteo geocoder

Results are cached in-memory.
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

_last_nominatim_call = 0.0
_GEOCODE_CACHE: dict[str, dict] = {}

_STRIP_PREFIXES = (
    "information regarding ", "tell me about ", "all information about ",
    "all information you can give w.r.t off ",
    "all information you can give w.r.t ",
    "give me all the data w.r.t off ",
    "give me all the data w.r.t ",
    "give me all the data about ",
    "information about ", "information on ", "what's happening at ",
    "what is happening at ", "conditions at ", "conditions off ",
    "off ", "near ", "the ", "from ",
)
_STRIP_SUFFIXES = (
    " coast", " coastline", " beach", " harbour", " harbor",
    " port", " district", " waters", " sea", " offshore",
    " harbour area", " harbor area",
)


def _normalize(place: str) -> str:
    s = (place or "").lower().strip()
    changed = True
    while changed:
        changed = False
        for prefix in _STRIP_PREFIXES:
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
                changed = True
        for suffix in _STRIP_SUFFIXES:
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
                changed = True
    return re.sub(r"\s+", " ", s)


def _throttle_nominatim() -> None:
    global _last_nominatim_call
    elapsed = time.time() - _last_nominatim_call
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    _last_nominatim_call = time.time()


def _nominatim_query(place: str, *, countrycodes: str | None) -> dict | None:
    _throttle_nominatim()
    params = {
        "q": place,
        "format": "json",
        "limit": 5,
        "addressdetails": 1,
        "accept-language": "en",
    }
    if countrycodes:
        params["countrycodes"] = countrycodes

    try:
        r = httpx.get(
            NOMINATIM_URL,
            params=params,
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

    # Prefer Indian results when the query is ambiguous.
    indian = [
        r for r in results
        if (r.get("address") or {}).get("country_code", "").lower() == "in"
    ]
    pool = indian or results
    pool.sort(key=lambda r: float(r.get("importance", 0) or 0), reverse=True)
    top = pool[0]

    addr = top.get("address") or {}
    name = (
        top.get("name")
        or addr.get("suburb")
        or addr.get("neighbourhood")
        or addr.get("city_district")
        or addr.get("city")
        or addr.get("town")
        or addr.get("county")
        or addr.get("state")
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


def _nominatim(place: str) -> dict | None:
    """Try Nominatim in three progressively broader modes."""
    # 1. India-only — the common case.
    r = _nominatim_query(place, countrycodes="in")
    if r is not None:
        return r

    # 2. No country filter — catches localities the India-only index
    #    doesn't have (some neighbourhoods, small towns).
    r = _nominatim_query(place, countrycodes=None)
    if r is not None:
        return r

    # 3. Append "India" — helps when the free-form query is too short.
    r = _nominatim_query(f"{place}, India", countrycodes=None)
    if r is not None:
        return r

    return None


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
      2. Nominatim (three modes)
      3. Open-Meteo
    """
    query = (query or "").strip()
    if not query:
        return {"status": "NOT_FOUND", "error": "No location supplied."}

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

    # Strip conversational filler, then normalize.
    cleaned = query
    for marker in ["location:", "place:"]:
        cleaned = cleaned.replace(marker, " ")
    place = _normalize(cleaned[:160])
    if not place:
        return {"status": "NOT_FOUND", "error": "No usable location text."}

    cached = _GEOCODE_CACHE.get(place)
    if cached is not None:
        return cached

    result = _nominatim(place) or _open_meteo(place)

    if result is None:
        result = {"status": "NOT_FOUND", "place": place}

    _GEOCODE_CACHE[place] = result
    return result