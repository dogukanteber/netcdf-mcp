"""JSON formatting and response building functions."""

import json
from typing import Any

import numpy as np
import xarray as xr


def json_response(*, data: Any) -> str:
    """Convert data to JSON string, handling numpy types."""
    return json.dumps(data, indent=2, default=str)


def error_response(*, message: str) -> str:
    """Return a JSON error response."""
    return json_response(data={"error": message})


def build_coordinate_info(*, coord: xr.DataArray, name: str) -> dict:
    """Build a dictionary with coordinate information."""
    return {
        "dtype": str(coord.dtype),
        "shape": coord.shape,
        "attrs": dict(coord.attrs),
    }


def build_variable_info(*, var: xr.DataArray, name: str) -> dict:
    """Build a dictionary with variable information."""
    return {
        "dtype": str(var.dtype),
        "shape": var.shape,
        "dims": var.dims,
        "attrs": dict(var.attrs),
    }


def build_time_range_result(*, time_name: str, time_var: xr.DataArray) -> dict:
    """Build the time range result dictionary."""
    time_values = time_var.values
    result = {
        "time_coordinate": time_name,
        "num_timesteps": len(time_values),
        "dtype": str(time_var.dtype),
        "attrs": dict(time_var.attrs),
        "start": str(time_values[0]),
        "end": str(time_values[-1]),
    }

    if len(time_values) >= 2:
        result["first_few"] = [str(t) for t in time_values[:5]]
        result["last_few"] = [str(t) for t in time_values[-5:]]

    return result


def build_coordinate_bounds(
    *, coord: xr.DataArray, name: str, default_units: str
) -> dict:
    """Build bounds information for a coordinate."""
    values = coord.values
    return {
        "name": name,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "num_values": len(values),
        "units": coord.attrs.get("units", default_units),
    }


def build_variable_explanation(
    *, var: xr.DataArray, variable_name: str, is_coordinate: bool
) -> dict:
    """Build a human-readable explanation for a variable."""
    attrs = dict(var.attrs)
    explanation_parts = []

    long_name = attrs.get("long_name", "")
    standard_name = attrs.get("standard_name", "")

    if long_name:
        explanation_parts.append(f"**{long_name}**")
    elif standard_name:
        explanation_parts.append(f"**{standard_name.replace('_', ' ').title()}**")
    else:
        explanation_parts.append(f"**{variable_name}**")

    if standard_name:
        explanation_parts.append(f"CF Standard Name: `{standard_name}`")

    units = attrs.get("units", "")
    if units:
        explanation_parts.append(f"Units: {units}")

    explanation_parts.append(f"Shape: {var.shape} with dimensions {var.dims}")

    valid_range = attrs.get("valid_range", [])
    valid_min = attrs.get("valid_min", valid_range[0] if valid_range else None)
    valid_max = attrs.get("valid_max", valid_range[1] if len(valid_range) > 1 else None)
    if valid_min is not None or valid_max is not None:
        explanation_parts.append(f"Valid range: [{valid_min}, {valid_max}]")

    cell_methods = attrs.get("cell_methods", "")
    if cell_methods:
        explanation_parts.append(f"Cell methods: {cell_methods}")

    comment = attrs.get("comment", "")
    if comment:
        explanation_parts.append(f"Note: {comment}")

    return {
        "variable": variable_name,
        "explanation": "\n".join(explanation_parts),
        "attributes": attrs,
        "dtype": str(var.dtype),
        "is_coordinate": is_coordinate,
    }
