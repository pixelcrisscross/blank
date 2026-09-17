from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

import copernicusmarine
import numpy as np


# ── Dataset IDs ──────────────────────────────────────────────────────────────
TEMPERATURE_DATASET = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
CURRENT_DATASET = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
WAVE_DATASET = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
# Replace the existing CHLOROPHYLL_DATASET constant
CHLOROPHYLL_DATASET = (
    "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D"
)
CHLOROPHYLL_LOOKBACK_HOURS = 240   # gap-free is daily; 10 days is safe

SURFACE_DEPTH = 0.49402499198913574

TEMPERATURE_LOOKBACK_HOURS = 72
CURRENT_LOOKBACK_HOURS = 72
WAVE_LOOKBACK_HOURS = 48
# Wide enough to include ocean even when the requested point is on a coastal
# land grid cell. ~0.5° ≈ 55 km.
POINT_WINDOW_DEGREES = 0.5

CACHE_TTL_SECONDS = 300


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def open_small_dataset(
    *,
    dataset_id: str,
    variables: list[str],
    latitude: float,
    longitude: float,
    hours_back: int,
    surface_data: bool = False,
):
    """Open a Copernicus slice wide enough to include ocean even when the
    requested point lands on a coastal land cell."""
    end_time = utc_now()
    start_time = end_time - timedelta(hours=hours_back)

    request: dict[str, Any] = {
        "dataset_id": dataset_id,
        "variables": variables,
        "minimum_longitude": longitude - POINT_WINDOW_DEGREES,
        "maximum_longitude": longitude + POINT_WINDOW_DEGREES,
        "minimum_latitude": latitude - POINT_WINDOW_DEGREES,
        "maximum_latitude": latitude + POINT_WINDOW_DEGREES,
        "start_datetime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_datetime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if surface_data:
        request["minimum_depth"] = SURFACE_DEPTH
        request["maximum_depth"] = SURFACE_DEPTH

    return copernicusmarine.open_dataset(**request)


def latest_value(
    dataset,
    variable: str,
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    """
    Extract the latest value nearest to (lat, lon), skipping land cells.

    Strategy:
      1. Reduce to surface + latest time step.
      2. Try the exact requested point.
      3. If NaN (land), search the entire loaded window for the nearest
         ocean grid cell using the coordinate arrays.
    """
    if variable not in dataset:
        raise KeyError(
            f"Variable '{variable}' not found. Available: {list(dataset.data_vars)}"
        )

    data = dataset[variable]

    if "depth" in data.dims:
        data = data.sel(depth=SURFACE_DEPTH, method="nearest")
    if "time" in data.dims:
        data = data.sortby("time").isel(time=-1)

    observation_time = None
    if "time" in data.coords:
        try:
            observation_time = str(data.coords["time"].values)
        except Exception:
            pass

    data = data.squeeze()

    # --- Try the exact point first ---
    try:
        exact = data.sel(latitude=latitude, longitude=longitude, method="nearest")
        value = json_value(exact.values)
    except Exception:
        value = None

    if value is not None and not (
        isinstance(value, float) and np.isnan(value)
    ):
        return {"value": value, "observation_time": observation_time}

    # --- Land cell: search the whole window for the nearest valid value ---
    try:
        values = np.asarray(data.values, dtype=np.float64)
        lats = np.asarray(data.coords["latitude"].values)
        lons = np.asarray(data.coords["longitude"].values)

        if values.ndim == 2:
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        else:
            return {"value": None, "observation_time": observation_time}

        dist = (lat_grid - latitude) ** 2 + (lon_grid - longitude) ** 2
        dist = np.where(np.isfinite(values), dist, np.inf)

        if not np.isfinite(dist).any():
            return {"value": None, "observation_time": observation_time}

        idx = np.unravel_index(np.argmin(dist), dist.shape)
        nearest_lat = float(lat_grid[idx])
        nearest_lon = float(lon_grid[idx])
        nearest_val = float(values[idx])
        offset_km = float(np.sqrt(dist[idx]) * 111.0)

        return {
            "value": nearest_val,
            "observation_time": observation_time,
            "note": (
                f"Requested point ({latitude:.4f}, {longitude:.4f}) is a land "
                f"grid cell. Using nearest ocean cell ({nearest_lat:.4f}, "
                f"{nearest_lon:.4f}), ~{offset_km:.1f} km away."
            ),
        }
    except Exception as exc:
        return {
            "value": None,
            "observation_time": observation_time,
            "note": f"Land search failed: {exc}",
        }


def get_temperature(latitude: float, longitude: float) -> dict[str, Any]:
    dataset = open_small_dataset(
        dataset_id=TEMPERATURE_DATASET,
        variables=["thetao"],
        latitude=latitude,
        longitude=longitude,
        hours_back=TEMPERATURE_LOOKBACK_HOURS,
        surface_data=True,
    )
    result = latest_value(dataset, "thetao", latitude, longitude)
    return {
        "parameter": "sea_surface_temperature",
        "value": result["value"],
        "unit": "degC",
        "variable": "thetao",
        "status": "OBSERVED",
        "observation_time": result["observation_time"],
        "dataset_id": TEMPERATURE_DATASET,
        **({"note": result["note"]} if "note" in result else {}),
    }


def get_currents(latitude: float, longitude: float) -> dict[str, Any]:
    dataset = open_small_dataset(
        dataset_id=CURRENT_DATASET,
        variables=["uo", "vo"],
        latitude=latitude,
        longitude=longitude,
        hours_back=CURRENT_LOOKBACK_HOURS,
        surface_data=True,
    )
    u_result = latest_value(dataset, "uo", latitude, longitude)
    v_result = latest_value(dataset, "vo", latitude, longitude)

    u = u_result["value"]
    v = v_result["value"]
    speed = None
    direction = None

    if u is not None and v is not None:
        u = float(u)
        v = float(v)
        speed = float(np.sqrt((u * u) + (v * v)))
        direction = float((np.degrees(np.arctan2(u, v)) + 360.0) % 360.0)

    return {
        "parameter": "ocean_current",
        "u_ms": u,
        "v_ms": v,
        "speed_ms": speed,
        "direction_deg": direction,
        "speed_unit": "m/s",
        "direction_unit": "degree",
        "variables": ["uo", "vo"],
        "status": "OBSERVED",
        "observation_time": u_result["observation_time"],
        "dataset_id": CURRENT_DATASET,
    }


def get_waves(latitude: float, longitude: float) -> dict[str, Any]:
    dataset = open_small_dataset(
        dataset_id=WAVE_DATASET,
        variables=["VHM0", "VTM02", "VMDR"],
        latitude=latitude,
        longitude=longitude,
        hours_back=WAVE_LOOKBACK_HOURS,
        surface_data=False,
    )
    height_result = latest_value(dataset, "VHM0", latitude, longitude)
    period_result = latest_value(dataset, "VTM02", latitude, longitude)
    direction_result = latest_value(dataset, "VMDR", latitude, longitude)

    return {
        "parameter": "waves",
        "significant_wave_height_m": height_result["value"],
        "mean_wave_period_s": period_result["value"],
        "wave_direction_deg": direction_result["value"],
        "variables": ["VHM0", "VTM02", "VMDR"],
        "status": "OBSERVED",
        "observation_time": height_result["observation_time"],
        "dataset_id": WAVE_DATASET,
    }



def get_chlorophyll(latitude: float, longitude: float) -> dict[str, Any]:
    dataset = open_small_dataset(
        dataset_id=CHLOROPHYLL_DATASET,
        variables=["CHL"],           # <-- uppercase in this product
        latitude=latitude,
        longitude=longitude,
        hours_back=CHLOROPHYLL_LOOKBACK_HOURS,
        surface_data=True,
    )
    result = latest_value(dataset, "CHL", latitude, longitude)
    return {
        "parameter": "chlorophyll_a",
        "value": result["value"],
        "unit": "mg/m³",
        "variable": "CHL",
        "status": "OBSERVED",
        "observation_time": result["observation_time"],
        "dataset_id": CHLOROPHYLL_DATASET,
        **({"note": result["note"]} if "note" in result else {}),
    }


# ── In-memory cache ─────────────────────────────────────────────────────────
_cache: dict[tuple[float, float], tuple[float, dict[str, Any]]] = {}


def _cache_key(latitude: float, longitude: float) -> tuple[float, float]:
    return (round(latitude, 3), round(longitude, 3))


def _get_cached(key: tuple[float, float]) -> dict[str, Any] | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    cached_at, value = entry
    age = datetime.now(timezone.utc).timestamp() - cached_at
    if age > CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return value


def _set_cache(key: tuple[float, float], value: dict[str, Any]) -> None:
    _cache[key] = (datetime.now(timezone.utc).timestamp(), value)


def get_copernicus_marine_snapshot(latitude: float, longitude: float) -> dict[str, Any]:
    latitude = float(latitude)
    longitude = float(longitude)

    cache_key = _cache_key(latitude, longitude)
    cached = _get_cached(cache_key)

    if cached is not None:
        cached_copy = dict(cached)
        cached_copy["cache"] = {"hit": True, "ttl_seconds": CACHE_TTL_SECONDS}
        return cached_copy

    retrieved_at = utc_now().isoformat()

    result: dict[str, Any] = {
        "source": "Copernicus Marine",
        "location": {"latitude": latitude, "longitude": longitude},
        "retrieved_at": retrieved_at,
        "observations": {},
        "provenance": [],
        "warnings": [],
        "status": "UNKNOWN",
        "cache": {"hit": False, "ttl_seconds": CACHE_TTL_SECONDS},
    }

    tasks = {
        "temperature": get_temperature,
        "currents": get_currents,
        "waves": get_waves,
        "chlorophyll": get_chlorophyll,
    }

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            name: executor.submit(function, latitude, longitude)
            for name, function in tasks.items()
        }

        for name, future in futures.items():
            try:
                value = future.result()
                result["observations"][name] = value
            except Exception as exc:
                result["warnings"].append({"parameter": name, "error": str(exc)})

    if "temperature" in result["observations"]:
        result["provenance"].append({
            "parameter": "sea_surface_temperature",
            "dataset_id": TEMPERATURE_DATASET,
            "variables": ["thetao"],
        })
    if "currents" in result["observations"]:
        result["provenance"].append({
            "parameter": "ocean_current",
            "dataset_id": CURRENT_DATASET,
            "variables": ["uo", "vo"],
        })
    if "waves" in result["observations"]:
        result["provenance"].append({
            "parameter": "waves",
            "dataset_id": WAVE_DATASET,
            "variables": ["VHM0", "VTM02", "VMDR"],
        })
    if "chlorophyll" in result["observations"]:
        result["provenance"].append({
            "parameter": "chlorophyll_a",
            "dataset_id": CHLOROPHYLL_DATASET,
            "variables": ["CHL"],
        })

    available = len(result["observations"])
    failed = len(result["warnings"])

    if available == 4:
        result["status"] = "COMPLETE"
    elif available > 0:
        result["status"] = "PARTIAL"
    else:
        result["status"] = "NO_DATA"

    result["summary"] = {
        "parameters_available": available,
        "parameters_failed": failed,
        "data_status": result["status"],
    }

    if available > 0:
        _set_cache(cache_key, result)

    return result