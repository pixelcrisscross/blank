from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import copernicusmarine
import numpy as np


TEMPERATURE_DATASET = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
CURRENT_DATASET = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
WAVE_DATASET = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"
SURFACE_DEPTH = 0.49402499198913574


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def safe_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def serialize_array(values: np.ndarray) -> list:
    return np.asarray(values).tolist()


def open_grid_dataset(
    *,
    dataset_id: str,
    variables: list[str],
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_longitude: float,
    maximum_longitude: float,
    hours_back: int,
    surface_data: bool = False,
):
    end_time = utc_now()
    start_time = end_time - timedelta(hours=hours_back)

    request: dict[str, Any] = {
        "dataset_id": dataset_id,
        "variables": variables,
        "minimum_latitude": minimum_latitude,
        "maximum_latitude": maximum_latitude,
        "minimum_longitude": minimum_longitude,
        "maximum_longitude": maximum_longitude,
        "start_datetime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_datetime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if surface_data:
        request["minimum_depth"] = SURFACE_DEPTH
        request["maximum_depth"] = SURFACE_DEPTH

    return copernicusmarine.open_dataset(**request)


def select_latest_surface(data_array):
    data = data_array

    if "depth" in data.dims:
        data = data.sel(depth=SURFACE_DEPTH, method="nearest")

    observation_time = None
    if "time" in data.dims:
        data = data.sortby("time")
        data = data.isel(time=-1)
        if "time" in data.coords:
            observation_time = str(data.coords["time"].values)

    return data.squeeze(), observation_time


def get_sst_grid(*, minimum_latitude, maximum_latitude, minimum_longitude, maximum_longitude):
    dataset = open_grid_dataset(
        dataset_id=TEMPERATURE_DATASET,
        variables=["thetao"],
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
        hours_back=72,
        surface_data=True,
    )

    data, observation_time = select_latest_surface(dataset["thetao"])
    values = np.asarray(data.values, dtype=np.float32)
    latitudes = data.coords["latitude"].values
    longitudes = data.coords["longitude"].values

    return {
        "parameter": "sea_surface_temperature",
        "short_name": "sst",
        "unit": "degC",
        "status": "OBSERVED",
        "observation_time": observation_time,
        "dataset_id": TEMPERATURE_DATASET,
        "latitudes": serialize_array(latitudes),
        "longitudes": serialize_array(longitudes),
        "values": serialize_array(values),
        "shape": list(values.shape),
    }


def get_current_grid(*, minimum_latitude, maximum_latitude, minimum_longitude, maximum_longitude):
    dataset = open_grid_dataset(
        dataset_id=CURRENT_DATASET,
        variables=["uo", "vo"],
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
        hours_back=72,
        surface_data=True,
    )

    u, observation_time = select_latest_surface(dataset["uo"])
    v, _ = select_latest_surface(dataset["vo"])

    u_values = np.asarray(u.values, dtype=np.float32)
    v_values = np.asarray(v.values, dtype=np.float32)
    speed = np.sqrt(u_values ** 2 + v_values ** 2)
    direction = (np.degrees(np.arctan2(u_values, v_values)) + 360.0) % 360.0

    latitudes = u.coords["latitude"].values
    longitudes = u.coords["longitude"].values

    return {
        "parameter": "ocean_current",
        "short_name": "current",
        "status": "OBSERVED",
        "observation_time": observation_time,
        "dataset_id": CURRENT_DATASET,
        "components": {
            "u": {"unit": "m/s", "values": serialize_array(u_values)},
            "v": {"unit": "m/s", "values": serialize_array(v_values)},
        },
        "speed": {"unit": "m/s", "values": serialize_array(speed)},
        "direction": {"unit": "degree", "values": serialize_array(direction)},
        "latitudes": serialize_array(latitudes),
        "longitudes": serialize_array(longitudes),
        "shape": list(speed.shape),
    }


def get_wave_grid(*, minimum_latitude, maximum_latitude, minimum_longitude, maximum_longitude):
    dataset = open_grid_dataset(
        dataset_id=WAVE_DATASET,
        variables=["VHM0", "VTM02", "VMDR"],
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
        hours_back=48,
        surface_data=False,
    )

    wave_height, observation_time = select_latest_surface(dataset["VHM0"])
    wave_period, _ = select_latest_surface(dataset["VTM02"])
    wave_direction, _ = select_latest_surface(dataset["VMDR"])

    height_values = np.asarray(wave_height.values, dtype=np.float32)
    period_values = np.asarray(wave_period.values, dtype=np.float32)
    direction_values = np.asarray(wave_direction.values, dtype=np.float32)

    latitudes = wave_height.coords["latitude"].values
    longitudes = wave_height.coords["longitude"].values

    return {
        "parameter": "waves",
        "short_name": "waves",
        "status": "OBSERVED",
        "observation_time": observation_time,
        "dataset_id": WAVE_DATASET,
        "significant_wave_height": {"unit": "m", "values": serialize_array(height_values)},
        "mean_wave_period": {"unit": "s", "values": serialize_array(period_values)},
        "wave_direction": {"unit": "degree", "values": serialize_array(direction_values)},
        "latitudes": serialize_array(latitudes),
        "longitudes": serialize_array(longitudes),
        "shape": list(height_values.shape),
    }


def get_copernicus_grid(parameter: str, *, minimum_latitude, maximum_latitude, minimum_longitude, maximum_longitude):
    parameter = parameter.lower().strip()

    if parameter in {"sst", "temperature", "sea_surface_temperature"}:
        return get_sst_grid(
            minimum_latitude=minimum_latitude,
            maximum_latitude=maximum_latitude,
            minimum_longitude=minimum_longitude,
            maximum_longitude=maximum_longitude,
        )

    if parameter in {"current", "currents", "ocean_current"}:
        return get_current_grid(
            minimum_latitude=minimum_latitude,
            maximum_latitude=maximum_latitude,
            minimum_longitude=minimum_longitude,
            maximum_longitude=maximum_longitude,
        )

    if parameter in {"wave", "waves"}:
        return get_wave_grid(
            minimum_latitude=minimum_latitude,
            maximum_latitude=maximum_latitude,
            minimum_longitude=minimum_longitude,
            maximum_longitude=maximum_longitude,
        )

    raise ValueError(
        f"Unsupported parameter: {parameter}. Supported parameters: sst, currents, waves."
    )