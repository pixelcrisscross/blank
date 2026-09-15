"""
IMD Coastal Weather Bulletin service — HTML scraper.

Fetches IMD's official Coastal Weather Bulletin for the ACWC/CWC region
covering a given coastal location. Returns port signals, storm surge
warnings, ocean current alerts, wind, visibility, and sea state.

Source: https://mausam.imd.gov.in/Forecast/coastal_bulletin_new.php?id=N

Bulletins contain multiple sub-region blocks ("North Maharashtra coast",
"South Maharashtra and Goa coast", etc.), so we split by region header and
pick the block whose name matches the caller's state hint.

Cache: 1 hour. Bulletins update every 12 hours.

Attribution: India Meteorological Department. Non-commercial use only
per RTI IMETD/R/E/25/00381.
"""
from __future__ import annotations

import re
import time
from typing import Any

import httpx

_BASE = "https://mausam.imd.gov.in/Forecast/coastal_bulletin_new.php"

_REGIONS = {
    1: {"centre": "ACWC Kolkata",           "covers": ["West Bengal", "Andaman & Nicobar"], "min_lat":  6.0, "max_lat": 23.5, "min_lon": 86.0, "max_lon": 94.5},
    2: {"centre": "CWC Thiruvananthapuram", "covers": ["Kerala", "Karnataka", "Lakshadweep"], "min_lat":  7.5, "max_lat": 15.0, "min_lon": 71.0, "max_lon": 77.5},
    3: {"centre": "CWC Ahmedabad",          "covers": ["Gujarat"],                    "min_lat": 20.0, "max_lat": 24.0, "min_lon": 68.0, "max_lon": 73.0},
    4: {"centre": "ACWC Mumbai",            "covers": ["Maharashtra", "Goa"],           "min_lat": 15.0, "max_lat": 21.0, "min_lon": 72.0, "max_lon": 74.5},
    5: {"centre": "CWC Bhubaneswar",        "covers": ["Odisha"],                      "min_lat": 17.5, "max_lat": 22.0, "min_lon": 81.0, "max_lon": 87.5},
    6: {"centre": "ACWC Chennai",           "covers": ["Tamil Nadu"],                  "min_lat":  8.0, "max_lat": 14.0, "min_lon": 76.5, "max_lon": 81.0},
    7: {"centre": "CWC Visakhapatnam",      "covers": ["Andhra Pradesh"],              "min_lat": 13.0, "max_lat": 19.5, "min_lon": 79.5, "max_lon": 85.5},
}

_LABEL_ALIASES = {
    "wind": ["wind"],
    "weather": ["weather"],
    "visibility": ["visibility"],
    "sea_condition": ["sea condition", "sea conditions", "sea state"],
    "port_signal": ["port signal", "port signals", "port warning", "signal at port"],
    "storm_surge_warning": [
        "storm surge/tidal warning", "storm surge", "tidal warning",
        "storm surge/tidal wave",
    ],
    "tidal_wave": ["tidal wave"],
}

_REGION_STOPWORDS = {
    "nil", "warning", "warnings", "tidal", "storm", "surge",
    "signal", "port", "issue", "time", "situation", "synoptic",
    "wave", "current", "currents", "alert",
}

_SENTENCE_PUNCT = ".!?:;"

_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 3600


# ═════════════════════════════════════════════════════════════════════
# REGION SELECTION (coverage box → bulletin id)
# ═════════════════════════════════════════════════════════════════════

def _region_for_point(latitude: float, longitude: float) -> int | None:
    for rid, box in _REGIONS.items():
        if (box["min_lat"] <= latitude <= box["max_lat"]
                and box["min_lon"] <= longitude <= box["max_lon"]):
            return rid
    best_rid, best_d = None, float("inf")
    for rid, box in _REGIONS.items():
        c_lat = (box["min_lat"] + box["max_lat"]) / 2
        c_lon = (box["min_lon"] + box["max_lon"]) / 2
        d = (c_lat - latitude) ** 2 + (c_lon - longitude) ** 2
        if d < best_d:
            best_d, best_rid = d, rid
    return best_rid


# ═════════════════════════════════════════════════════════════════════
# HTML STRIPPING
# ═════════════════════════════════════════════════════════════════════

def _strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    for entity, char in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
                         ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")]:
        text = text.replace(entity, char)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ═════════════════════════════════════════════════════════════════════
# REGION SPLITTING — walk back from each "coast" keyword
# ═════════════════════════════════════════════════════════════════════

def _walk_back_region_name(text: str, coast_pos: int) -> tuple[str, int]:
    """
    Return (region_name, name_start_pos) by walking back from the
    "coast" keyword. Stops at stopwords, sentence punctuation, or
    lowercase words (except "and", which joins multi-word names).
    """
    i = coast_pos
    while i > 0 and text[i - 1].isspace():
        i -= 1

    words: list[str] = []
    region_start = i

    while i > 0 and len(words) < 6:
        while i > 0 and text[i - 1].isspace():
            i -= 1
        if i == 0:
            break
        if text[i - 1] in _SENTENCE_PUNCT:
            break
        word_end = i
        while i > 0 and (text[i - 1].isalnum() or text[i - 1] in "'-"):
            i -= 1
        word_start = i
        word = text[word_start:word_end]
        if not word:
            break
        if word.lower() in _REGION_STOPWORDS:
            break
        if word.lower() == "and":
            words.insert(0, word)
            region_start = word_start
            continue
        if not word[0].isupper():
            break
        words.insert(0, word)
        region_start = word_start

    return (" ".join(words), region_start)


def _split_regions(text: str) -> list[tuple[str, str]]:
    """Return [(region_name, block_text), ...]."""
    coast_matches = list(re.finditer(r"coast\b", text))
    if not coast_matches:
        return [("", text)]

    anchors: list[tuple[str, int, int]] = []
    for cm in coast_matches:
        cp = cm.start()
        name, name_start = _walk_back_region_name(text, cp)
        if not name:
            continue
        anchors.append((name, name_start, cm.end()))

    if not anchors:
        return [("", text)]

    blocks: list[tuple[str, str]] = []
    for i, (name, _, coast_end) in enumerate(anchors):
        block_start = coast_end
        if i + 1 < len(anchors):
            block_end = anchors[i + 1][1]  # next region's name start
        else:
            block_end = len(text)
        if block_end <= block_start:
            block_end = len(text)
        blocks.append((name, text[block_start:block_end]))
    return blocks


def _pick_region(regions: list[tuple[str, str]],
                 state_hint: str | None) -> tuple[str, str]:
    """Pick the region matching state_hint; else the longest block."""
    if not regions:
        return ("", "")
    if state_hint:
        hint_nospace = state_hint.lower().replace(" ", "")
        for name, block in regions:
            if hint_nospace in name.lower().replace(" ", ""):
                return (name, block)
    return max(regions, key=lambda x: len(x[1]))


# ═════════════════════════════════════════════════════════════════════
# LABEL DETECTION — longest-alias-wins per position
# ═════════════════════════════════════════════════════════════════════

def _find_label_positions(text: str) -> list[tuple[str, int, int]]:
    """Return (key, start_pos, label_len) sorted by position."""
    hits: list[tuple[str, int, int]] = []
    lower = text.lower()

    for key, aliases in _LABEL_ALIASES.items():
        sorted_aliases = sorted(aliases, key=len, reverse=True)
        matched: set[int] = set()
        for alias in sorted_aliases:
            start = 0
            while True:
                idx = lower.find(alias, start)
                if idx < 0:
                    break
                before_ok = idx == 0 or not lower[idx - 1].isalnum()
                after_idx = idx + len(alias)
                after_ok = (
                    after_idx >= len(lower)
                    or not lower[after_idx].isalnum()
                )
                if before_ok and after_ok and idx not in matched:
                    hits.append((key, idx, len(alias)))
                    matched.add(idx)
                start = idx + 1

    hits.sort(key=lambda x: x[1])
    return hits


# ═════════════════════════════════════════════════════════════════════
# VALUE EXTRACTION — slice first, strip after
# ═════════════════════════════════════════════════════════════════════

def _extract_value_at(text: str, start_pos: int, label_len: int,
                     next_pos: int | None) -> str | None:
    # Slice raw first so positions stay anchored to the original text
    if next_pos is not None and next_pos > start_pos:
        raw = text[start_pos + label_len: next_pos]
    else:
        raw = text[start_pos + label_len:]

    # Drop "Time of Issue ..." trailing marker
    raw = re.sub(r"\s*Time\s+of\s+Issue.*$", "", raw, flags=re.IGNORECASE)

    # Strip leading separators + whitespace
    raw = re.sub(r"^\s*[\|:\-–>»\.]*\s*", "", raw, count=1)
    # Strip trailing separators + whitespace
    raw = re.sub(r"[\s\|:\-–>»]+$", "", raw)
    raw = raw.strip()

    if len(raw) > 1500:
        raw = raw[:1500] + "…"
    return raw or None


# ═════════════════════════════════════════════════════════════════════
# PARSER
# ═════════════════════════════════════════════════════════════════════

def _parse_bulletin(text: str, state_hint: str | None = None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}

    # Centre name
    m = re.search(r"\b(ACWC|CWC)\s+([A-Z][A-Za-z]+)", text)
    if m:
        parsed["centre_reported"] = f"{m.group(1)} {m.group(2)}"

    # Valid window
    m = re.search(
        r"Bulletin\s+Valid\s+for\s+12\s+hrs\s+from\s+(.*?)\s+to\s+(.*?)"
        r"(?=\s+(?:-->|Synoptic|Time\s+of\s+Issue|ACWC|CWC|$))",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        parsed["valid_from"] = m.group(1).strip(" .-")
        parsed["valid_to"] = m.group(2).strip(" .-")

    # Time of Issue — accept both YYYY-MM-DD and DD-MM-YYYY
    issues = re.findall(
        r"Time\s+of\s+Issue\s*[\|:\-–>»]*\s*"
        r"(\d{1,2}:\d{2}\s*IST\s*of\s*\d{1,4}[-/]\d{1,2}[-/]\d{1,4})",
        text, flags=re.IGNORECASE,
    )
    if issues:
        parsed["issued_at"] = issues[-1].strip()

    # Synoptic situation — extract from full text (before any region header)
    syn = re.search(
        r"Synoptic\s+Situation\s+(.*?)"
        r"(?=\s+[A-Z][A-Za-z]+\s+(?:and\s+)?[A-Z][A-Za-z]*\s+coast\b)",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    if syn:
        value = syn.group(1).strip(" .")
        if len(value) > 1500:
            value = value[:1500] + "…"
        parsed["synoptic_situation"] = value

    # Region split — pick the matching block
    regions = _split_regions(text)
    region_name, block = _pick_region(regions, state_hint)
    if region_name:
        parsed["region"] = region_name

    # Field extraction within the chosen block
    hits = _find_label_positions(block)
    for i, (key, pos, label_len) in enumerate(hits):
        if key in parsed:
            continue
        next_pos = hits[i + 1][1] if i + 1 < len(hits) else None
        value = _extract_value_at(block, pos, label_len, next_pos)
        if value:
            parsed[key] = value

    return parsed


# ═════════════════════════════════════════════════════════════════════
# FETCH
# ═════════════════════════════════════════════════════════════════════

def _fetch_bulletin(region_id: int,
                    state_hint: str | None = None) -> dict[str, Any]:
    cache_key = f"{region_id}:{state_hint or ''}"
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and (now - cached[0]) < _CACHE_TTL:
        return cached[1]

    url = f"{_BASE}?id={region_id}"
    r = httpx.get(
        url, timeout=25, follow_redirects=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; ORCA-Marine-Intelligence/1.0; "
                "academic, non-commercial)"
            ),
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    r.raise_for_status()

    text = _strip_html(r.text)
    parsed = _parse_bulletin(text, state_hint=state_hint)
    parsed["bulletin_id"] = region_id
    parsed["source_url"] = url
    parsed["raw_text_excerpt"] = text[:2000]

    parsed["parse_ok"] = any(
        k in parsed
        for k in ("wind", "sea_condition", "port_signal",
                  "storm_surge_warning", "visibility")
    )

    _CACHE[cache_key] = (now, parsed)
    return parsed


# ═════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════

def _is_active(value: str) -> bool:
    v = (value or "").strip().upper()
    if not v or v in ("/", "-", "N/A", "NA", "."):
        return False
    if "NIL" in v or "NO WARNING" in v or v == "NONE":
        return False
    if len(v) < 4:
        return False
    return True


def get_imd_coastal_bulletin(
    latitude: float,
    longitude: float,
    state_hint: str | None = None,
) -> dict[str, Any]:
    """
    Return the IMD Coastal Weather Bulletin for the region covering a point.

    Rejects inland points before fetching — IMD coastal bulletins apply
    only to coastal waters.
    """
    # Coastal guard: IMD bulletins are coastal-only.
    from tools.coastal_check import is_coastal
    coast = is_coastal(latitude, longitude)
    if not coast["coastal"]:
        dist = coast.get("distance_km")
        return {
            "status": "NO_REGION",
            "source": "IMD Coastal Weather Bulletin (HTML)",
            "note": (
                f"Location is inland"
                + (f" (~{dist} km from the nearest coast)" if dist else "")
                + ". IMD coastal bulletins apply only to coastal waters. "
                  "Use Open-Meteo for inland weather."
            ),
            "distance_to_coast_km": dist,
        }

    region_id = _region_for_point(latitude, longitude)
    if region_id is None:
        return {
            "status": "NO_REGION",
            "source": "IMD Coastal Weather Bulletin (HTML)",
            "note": "No IMD coastal region covers this point.",
        }

    try:
        parsed = _fetch_bulletin(region_id, state_hint=state_hint)
    except Exception as exc:
        return {
            "status": "ERROR",
            "source": "IMD Coastal Weather Bulletin (HTML)",
            "region_id": region_id,
            "error": str(exc),
        }

    if not parsed.get("parse_ok"):
        return {
            "status": "PARSE_FAILED",
            "source": "IMD Coastal Weather Bulletin (HTML)",
            "region_id": region_id,
            "note": "Page fetched but no fields could be parsed.",
            "raw_text_excerpt": parsed.get("raw_text_excerpt", "")[:500],
        }

    box = _REGIONS[region_id]
    port_signal = parsed.get("port_signal") or ""
    storm_surge = parsed.get("storm_surge_warning") or ""
    tidal_wave = parsed.get("tidal_wave") or ""

    def _is_active(value: str) -> bool:
        v = (value or "").strip().upper()
        if not v or v in ("/", "-", "N/A", "NA", "."):
            return False
        if "NIL" in v or "NO WARNING" in v or v == "NONE":
            return False
        if len(v) < 4:
            return False
        return True

    return {
        "status": "OK",
        "source": "IMD Coastal Weather Bulletin (HTML)",
        "bulletin_id": region_id,
        "centre": parsed.get("centre_reported") or box["centre"],
        "region": parsed.get("region"),
        "covers": box["covers"],
        "issued_at": parsed.get("issued_at"),
        "valid_from": parsed.get("valid_from"),
        "valid_to": parsed.get("valid_to"),
        "synoptic_situation": parsed.get("synoptic_situation"),
        "wind": parsed.get("wind"),
        "weather": parsed.get("weather"),
        "visibility": parsed.get("visibility"),
        "sea_condition": parsed.get("sea_condition"),
        "port_signal": port_signal or None,
        "port_signal_active": _is_active(port_signal),
        "storm_surge_warning": storm_surge or None,
        "storm_surge_active": _is_active(storm_surge),
        "tidal_wave": tidal_wave or None,
        "tidal_wave_active": _is_active(tidal_wave),
        "attribution": "India Meteorological Department (IMD)",
        "note": (
            "Official IMD Coastal Weather Bulletin. Port Signal indicates "
            "warning flags hoisted at named ports. Storm Surge / Tidal "
            "Warning flags wave-surge risk for small craft. "
            "Non-commercial use only."
        ),
    }