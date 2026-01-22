"""Data analysis, statistics, quality checks, and compliance functions."""

from pathlib import Path

import numpy as np
import xarray as xr

from netcdf_mcp.dataset import (
    TIME_COORDINATE_NAMES,
    find_time_coordinate,
    open_dataset,
)
from netcdf_mcp.response import json_response


def extract_valid_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Extract valid (non-NaN) values from an array.

    Returns: (flattened_values, valid_values, nan_count)
    """
    if hasattr(values, "mask"):
        values = np.ma.filled(values, np.nan)

    values_flat = values.flatten()
    nan_count = int(np.sum(np.isnan(values_flat)))
    valid_values = values_flat[~np.isnan(values_flat)]

    return values_flat, valid_values, nan_count


def compute_correlation(v1: np.ndarray, v2: np.ndarray) -> dict:
    """Compute correlation between two arrays, handling NaN values."""
    valid_mask = ~(np.isnan(v1) | np.isnan(v2))
    v1_valid = v1[valid_mask]
    v2_valid = v2[valid_mask]

    result = {}
    if len(v1_valid) > 1:
        correlation = float(np.corrcoef(v1_valid, v2_valid)[0, 1])
        result["correlation"] = correlation if not np.isnan(correlation) else None
        result["valid_pairs"] = int(len(v1_valid))

    return result


def compute_quality_stats(
    *, valid_values: np.ndarray, issues: list, warnings: list
) -> dict | None:
    """Compute quality statistics and detect outliers/anomalies."""
    if len(valid_values) == 0:
        issues.append("No valid data values found")
        return None

    min_val = float(np.min(valid_values))
    max_val = float(np.max(valid_values))
    mean_val = float(np.mean(valid_values))
    std_val = float(np.std(valid_values))

    inf_count = int(np.sum(np.isinf(valid_values)))
    if inf_count > 0:
        issues.append(f"Contains {inf_count} infinite values")

    if std_val > 0:
        outlier_mask = np.abs(valid_values - mean_val) > 5 * std_val
        outlier_count = int(np.sum(outlier_mask))
        if outlier_count > 0:
            warnings.append(f"{outlier_count} potential outliers (>5 std from mean)")

    if min_val == max_val:
        warnings.append("All values are identical (constant field)")

    return {"min": min_val, "max": max_val, "mean": mean_val, "std": std_val}


def check_physical_constraints(
    *, var: xr.DataArray, min_val: float, max_val: float
) -> list[str]:
    """
    Check for physical constraint violations based on variable units.

    Returns list of issues/warnings found.
    """
    warnings = []
    units = var.attrs.get("units", "").lower()

    if "kelvin" in units or units == "k":
        if min_val < 0:
            warnings.append(f"Negative Kelvin temperature: {min_val}")
        if max_val > 400:
            warnings.append(f"Very high temperature: {max_val} K")
    elif "percent" in units or units == "%":
        if min_val < 0 or max_val > 100:
            warnings.append(f"Percentage out of range: [{min_val}, {max_val}]")

    return warnings


def check_variable_attributes(*, dataset: xr.Dataset) -> tuple[list[str], list[str]]:
    """Check variable attributes for CF compliance."""
    warnings = []

    for name, var in dataset.data_vars.items():
        if "units" not in var.attrs:
            warnings.append(f"Variable '{name}' missing 'units' attribute")

        if "long_name" not in var.attrs and "standard_name" not in var.attrs:
            warnings.append(
                f"Variable '{name}' missing both 'long_name' and 'standard_name'"
            )

    for name, coord in dataset.coords.items():
        if "units" not in coord.attrs and name.lower() not in ["time", "t"]:
            warnings.append(f"Coordinate '{name}' missing 'units' attribute")

    return [], warnings


def check_time_compliance(
    *, dataset: xr.Dataset
) -> tuple[list[str], list[str], list[str]]:
    """Check time coordinate for CF compliance."""
    passed = []
    issues = []
    warnings = []

    time_result = find_time_coordinate(dataset)
    if not time_result:
        return passed, issues, warnings

    _, time_coord = time_result

    if "units" in time_coord.attrs:
        passed.append(f"Time coordinate has units: {time_coord.attrs['units']}")
    else:
        issues.append("Time coordinate missing 'units' attribute")

    if "calendar" in time_coord.attrs:
        passed.append(f"Time coordinate has calendar: {time_coord.attrs['calendar']}")
    else:
        warnings.append(
            "Time coordinate missing 'calendar' attribute (standard assumed)"
        )

    return passed, issues, warnings


def get_time_range_fallback(*, path: Path, original_error: Exception) -> str:
    """Fallback for time range when time decoding fails."""
    try:
        with open_dataset(path=path, decode_times=False) as ds:
            for name in TIME_COORDINATE_NAMES:
                if name not in ds.coords:
                    continue

                time_var = ds.coords[name]
                result = {
                    "time_coordinate": name,
                    "num_timesteps": len(time_var.values),
                    "raw_start": float(time_var.values[0]),
                    "raw_end": float(time_var.values[-1]),
                    "units": time_var.attrs.get("units", "unknown"),
                    "calendar": time_var.attrs.get("calendar", "unknown"),
                    "note": "Time decoding failed, showing raw values",
                }
                return json_response(data=result)

    except Exception:
        pass

    from netcdf_mcp.response import error_response

    return error_response(message=f"Failed to get time range: {original_error}")
