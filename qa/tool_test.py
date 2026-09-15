"""
Tool-level QA.

Calls each tool directly with coordinates for known coastal and inland
locations. Prints a compact summary per tool so mismatches are visible.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.copernicus_service import get_copernicus_marine_snapshot
from tools.weather_service import get_weather_conditions
from tools.geofence import check_geofence
from tools.pfz_service import get_pfz_advisory
from tools.coral_service import get_coral_bleaching_alert
from tools.tide_service import get_tide_prediction
from tools.bioluminescence import get_bioluminescence_forecast
from tools.algal_bloom import get_algal_bloom_risk
from tools.imd_service import get_imd_coastal_bulletin


GOA = (15.2993, 73.8243)
LAKSHADWEEP = (10.5667, 72.6417)
KORAMANGALA = (12.9320, 77.6227)


def _sep(title):
    print()
    print("─" * 80)
    print(title)
    print("─" * 80)


def test_ocean():
    _sep("Copernicus ocean snapshot (Goa)")
    r = get_copernicus_marine_snapshot(*GOA)
    print(f"status={r.get('status')}")
    obs = r.get("observations", {})
    for name in ("temperature", "currents", "waves", "chlorophyll"):
        block = obs.get(name) or {}
        print(f"  {name:12s}: {block.get('value') or block.get('speed_ms') or block.get('significant_wave_height_m')}")


def test_weather():
    _sep("Open-Meteo weather (Goa)")
    r = get_weather_conditions(*GOA, 3)
    print(f"status={r.get('status')}  data_status={r.get('data_status')}")
    cur = r.get("current") or {}
    print(f"  wind={cur.get('wind_speed_10m')}  gust={cur.get('wind_gusts_10m')}  "
          f"precip={cur.get('precipitation')}  code={cur.get('weather_code')}")


def test_geofence():
    _sep("Geofence (Goa)")
    r = check_geofence(*GOA)
    print(f"status={r.get('status')}  inside={r.get('inside_restricted_zone')}")


def test_pfz():
    _sep("PFZ advisory (Goa)")
    r = get_pfz_advisory(*GOA)
    print(f"status={r.get('status')}")
    print(f"  lines={len(r.get('pfz_lines', []))}  "
          f"lcs={len(r.get('landing_centres', []))}")
    if r.get("pfz_lines"):
        n = r["pfz_lines"][0]
        print(f"  nearest line: {n.get('distance_km')} km at "
              f"({n.get('nearest_lat')}, {n.get('nearest_lon')})")


def test_coral():
    _sep("Coral bleaching alert")
    print("-- Lakshadweep (should have reef) --")
    r = get_coral_bleaching_alert(*LAKSHADWEEP)
    print(f"  status={r.get('status')}  alert={r.get('bleaching_alert')}  "
          f"reef={r.get('reef_region')}  dist={r.get('distance_to_reef_km')}")

    print("-- Goa (no reef, should say NO_REEF_AT_LOCATION) --")
    r = get_coral_bleaching_alert(*GOA)
    print(f"  status={r.get('status')}  nearest_reef={r.get('nearest_reef')}  "
          f"dist={r.get('distance_to_nearest_reef_km')}")


def test_tides():
    _sep("Tides (local harmonic)")
    for port in ("Goa", "Mumbai", "Chennai", "Kochi", "UnknownPort"):
        r = get_tide_prediction(port)
        print(f"  {port:12s}: status={r.get('status'):20s} "
              f"regime={r.get('moon_regime')} "
              f"range={r.get('range_m')}")


def test_biolum():
    _sep("Bioluminescence forecast (Goa)")
    r = get_bioluminescence_forecast(*GOA)
    print(f"  status={r.get('status')}  "
          f"label={r.get('likelihood_label')}  "
          f"score={r.get('likelihood_score')}")


def test_algal():
    _sep("Algal bloom risk (Goa)")
    r = get_algal_bloom_risk(*GOA)
    print(f"  status={r.get('status')}  "
          f"chl={r.get('chlorophyll_a_mg_m3')}  "
          f"level={r.get('risk_level')}")


def test_imd():
    _sep("IMD Coastal Bulletin")
    for name, (lat, lon), hint in [
        ("Goa",       GOA,         "Goa"),
        ("Mumbai",    (19.0760, 72.8777), "Maharashtra"),
        ("Chennai",   (13.0827, 80.2707), "Tamil Nadu"),
        ("Kochi",     (9.9312,  76.2673), "Kerala"),
    ]:
        r = get_imd_coastal_bulletin(lat, lon, state_hint=hint)
        print(f"  {name:10s}: status={r.get('status'):10s}  "
              f"id={r.get('bulletin_id')}  "
              f"region={r.get('region')}")
        print(f"     wind={r.get('wind')}")
        print(f"     port_signal={r.get('port_signal')}  "
              f"active={r.get('port_signal_active')}")
        print(f"     issued_at={r.get('issued_at')}")


def main():
    test_ocean()
    test_weather()
    test_geofence()
    test_pfz()
    test_coral()
    test_tides()
    test_biolum()
    test_algal()
    test_imd()


if __name__ == "__main__":
    main()