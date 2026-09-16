"""
ORCA map visualization tool.

Generates a Folium interactive HTML map showing:
  - Query point
  - Marine query point (if different)
  - Live INCOIS PFZ lines and landing centres
  - A floating sidebar with current marine conditions and alerts

Output: a self-contained HTML file written to ./output/maps/.
The filename is deterministic per location (overwritten on each run)
so the synthesis model cannot hallucinate a different path.

This tool is invoked ONLY when the user explicitly asks for a map,
plot, or visualization. It is never called automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import folium
    _FOLIUM_AVAILABLE = True
except ImportError:
    _FOLIUM_AVAILABLE = False


_OUTPUT_DIR = Path("output") / "maps"


def _ensure_output_dir() -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR.resolve()


def _fmt_val(v, places: int = 2) -> str:
    try:
        if v is None:
            return "—"
        return f"{float(v):.{places}f}"
    except (TypeError, ValueError):
        return "—"


def _build_conditions_html(ocean: dict, weather: dict) -> str:
    rows: list[str] = []

    obs = (ocean or {}).get("observations", {}) or {}
    sst = (obs.get("temperature") or {}).get("value")
    cur = obs.get("currents") or {}
    wav = obs.get("waves") or {}
    chl = (obs.get("chlorophyll") or {}).get("value")

    rows.append(f"<b>SST</b>: {_fmt_val(sst, 2)} °C")
    rows.append(
        f"<b>Current</b>: {_fmt_val(cur.get('speed_ms'), 2)} m/s "
        f"@ {_fmt_val(cur.get('direction_deg'), 0)}°"
    )
    rows.append(
        f"<b>Waves</b>: {_fmt_val(wav.get('significant_wave_height_m'), 2)} m "
        f"/ {_fmt_val(wav.get('mean_wave_period_s'), 1)} s"
    )
    rows.append(f"<b>Chlorophyll</b>: {_fmt_val(chl, 2)} mg/m³")

    cur_w = (weather or {}).get("current", {}) or {}
    rows.append(
        f"<b>Wind</b>: {_fmt_val(cur_w.get('wind_speed_10m'), 1)} km/h "
        f"(gust {_fmt_val(cur_w.get('wind_gusts_10m'), 1)})"
    )

    return "<br>".join(rows)


def _build_alerts_html(hwassa: dict, currents: dict) -> str:
    parts: list[str] = []

    if hwassa and hwassa.get("status") == "OK":
        sev = hwassa.get("max_severity", "NONE")
        if sev != "NONE":
            color = (hwassa.get("max_color") or "Yellow").lower()
            color_hex = "#d97706" if color == "orange" else "#ca8a04"
            parts.append(
                f"<b style='color:{color_hex};'>"
                f"⚠ {hwassa.get('alert_type')} — "
                f"{hwassa.get('district')}</b>"
            )
            period = hwassa.get("swell_period_s_range")
            if period:
                parts.append(f"Swell period {period[0]:.0f}-{period[1]:.0f} s")
            height = hwassa.get("wave_height_m_range")
            if height:
                parts.append(f"Wave height {height[0]:.1f}-{height[1]:.1f} m")
            if hwassa.get("rip_current_risk"):
                parts.append(
                    "<i style='color:#b91c1c;'>"
                    "Rip current risk — swimmers take care.</i>"
                )

    if currents and currents.get("status") == "OK":
        sev = currents.get("max_severity", "NONE")
        if sev != "NONE":
            color = (currents.get("max_color") or "Yellow").lower()
            color_hex = "#d97706" if color == "orange" else "#ca8a04"
            parts.append(
                f"<b style='color:{color_hex};'>"
                f"⚠ {currents.get('alert_type')} — "
                f"{currents.get('district')}</b>"
            )

    if not parts:
        return "<i style='color:#6b7280;'>No active INCOIS alerts for this region.</i>"

    return "<br>".join(parts)


def _safe_filename(label: str) -> str:
    """
    Convert a label into a deterministic, filesystem-safe filename.

    'ORCA — Kochi' → 'ORCA_Kochi'
    'ORCA -- Kochi' → 'ORCA_Kochi'
    Multiple underscores are collapsed. Leading/trailing stripped.
    """
    out: list[str] = []
    for c in label:
        if c.isalnum():
            out.append(c)
        elif c in "-_":
            out.append(c)
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("_-") or "orca_map"
    return s


def create_marine_map(
    *,
    query_lat: float,
    query_lon: float,
    query_name: str = "Query point",
    marine_lat: float | None = None,
    marine_lon: float | None = None,
    marine_name: str = "Marine query point",
    distance_km: float | None = None,
    ocean: dict | None = None,
    weather: dict | None = None,
    pfz: dict | None = None,
    hwassa: dict | None = None,
    currents: dict | None = None,
    imd: dict | None = None,
    label: str = "orca",
) -> dict[str, Any]:
    """Generate a Folium map HTML file. Returns a dict with the file path."""
    if not _FOLIUM_AVAILABLE:
        return {
            "status": "ERROR",
            "error": "folium is not installed. Run: pip install folium",
        }

    center_lat = marine_lat if marine_lat is not None else query_lat
    center_lon = marine_lon if marine_lon is not None else query_lon

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    # ── Query point ──────────────────────────────────────────────────
    folium.Marker(
        location=[query_lat, query_lon],
        popup=folium.Popup(f"<b>{query_name}</b><br>Query point", max_width=250),
        tooltip=query_name,
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    ).add_to(m)

    # ── Marine query point ───────────────────────────────────────────
    if marine_lat is not None and marine_lon is not None:
        popup = f"<b>{marine_name}</b><br>Marine query point"
        if distance_km:
            popup += f"<br>{distance_km:.1f} km from query point"
        folium.Marker(
            location=[marine_lat, marine_lon],
            popup=folium.Popup(popup, max_width=250),
            tooltip=marine_name,
            icon=folium.Icon(color="red", icon="anchor", prefix="fa"),
        ).add_to(m)

        if abs(query_lat - marine_lat) > 1e-5 or abs(query_lon - marine_lon) > 1e-5:
            folium.PolyLine(
                locations=[[query_lat, query_lon], [marine_lat, marine_lon]],
                color="gray",
                weight=1.5,
                opacity=0.6,
                dash_array="5,5",
            ).add_to(m)

    # ── PFZ lines ────────────────────────────────────────────────────
    pfz_lines = (pfz or {}).get("pfz_lines") or []
    for i, line in enumerate(pfz_lines):
        lat = line.get("nearest_lat")
        lon = line.get("nearest_lon")
        if lat is None or lon is None:
            continue
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color="#dc2626",
            fill=True,
            fill_opacity=0.6,
            popup=folium.Popup(
                f"<b>PFZ line #{i+1}</b><br>"
                f"State: {line.get('state')}<br>"
                f"Distance: {line.get('distance_km')} km",
                max_width=250,
            ),
            tooltip=f"PFZ line ({line.get('distance_km')} km)",
        ).add_to(m)

    # ── Landing centres ──────────────────────────────────────────────
    lcs = (pfz or {}).get("landing_centres") or []
    for lc in lcs[:10]:
        lat = lc.get("latitude")
        lon = lc.get("longitude")
        if lat is None or lon is None:
            continue
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color="#0284c7",
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{lc.get('name')}</b><br>"
                f"{lc.get('sector')}, {lc.get('district')}<br>"
                f"Distance: {lc.get('distance_km')} km",
                max_width=250,
            ),
            tooltip=lc.get("name"),
        ).add_to(m)

    # ── Sidebar ──────────────────────────────────────────────────────
    conditions_html = _build_conditions_html(ocean or {}, weather or {})
    alerts_html = _build_alerts_html(hwassa or {}, currents or {})

    imd_note = ""
    if imd and imd.get("status") == "OK":
        imd_note = (
            f"<br><b>IMD</b>: {imd.get('centre')} — "
            f"{imd.get('sea_condition', 'no bulletin details')}"
        )

    sidebar_html = f"""
    <div style="
        position: fixed;
        top: 12px;
        right: 12px;
        z-index: 9999;
        background: white;
        padding: 14px 16px;
        border-radius: 8px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 13px;
        line-height: 1.6;
        max-width: 300px;
        color: #1f2937;
    ">
        <div style="font-weight: 700; font-size: 14px; margin-bottom: 8px; color: #111827;">
            {label}
        </div>
        <div style="margin-bottom: 10px;">
            <div style="font-weight: 600; color: #374151; margin-bottom: 4px;">Conditions</div>
            {conditions_html}{imd_note}
        </div>
        <div>
            <div style="font-weight: 600; color: #374151; margin-bottom: 4px;">Alerts</div>
            {alerts_html}
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(sidebar_html))

    # ── Save ─────────────────────────────────────────────────────────
    out_dir = _ensure_output_dir()

    # Deterministic filename: one file per location, overwritten on
    # each run. Prevents the synthesis model from hallucinating a
    # prettier path and makes the output predictable for the user.
    safe_label = _safe_filename(label)
    filename = f"{safe_label}.html"
    out_path = (out_dir / filename).resolve()

    m.save(str(out_path))

    return {
        "status": "OK",
        "source": "ORCA map visualization",
        "map_path": str(out_path),
        "map_filename": filename,
        "center": {"latitude": center_lat, "longitude": center_lon},
        "note": "Interactive map saved to disk. Open the file in a browser.",
    }