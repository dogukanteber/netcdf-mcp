"""Climate NetCDF MCP Server - reads and understands NetCDF files for debugging."""

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from mcp.server.fastmcp import FastMCP

from climate_mcp_server.exceptions import (
    FileNotFoundError,
    InvalidFileError,
    NetCDFError,
    VariableNotFoundError,
)

mcp = FastMCP("climate-netcdf")


def _validate_netcdf_path(*, file_path: str) -> Path:
    """Validate that the path exists and is a NetCDF file."""
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in (".nc", ".nc4", ".netcdf"):
        raise InvalidFileError(f"Not a NetCDF file: {path}")

    return path


def _json_response(*, data: Any) -> str:
    """Convert data to JSON string, handling numpy types."""
    return json.dumps(data, indent=2, default=str)


def _error_response(*, message: str) -> str:
    """Return a JSON error response."""
    return _json_response(data={"error": message})


@contextmanager
def _open_dataset(*, path: Path, decode_times: bool = True):
    """Context manager for safely opening and closing xarray datasets."""
    dataset = xr.open_dataset(path, decode_times=decode_times)
    try:
        yield dataset
    finally:
        dataset.close()


def _get_variable(
    *, dataset: xr.Dataset, variable_name: str, include_coords: bool = True
) -> xr.DataArray:
    """Get a variable from the dataset, raising VariableNotFoundError if not found."""
    is_data_var = variable_name in dataset.data_vars
    is_coord = variable_name in dataset.coords

    if not is_data_var and not (include_coords and is_coord):
        available = list(dataset.data_vars.keys())
        if include_coords:
            available.extend(dataset.coords.keys())
        raise VariableNotFoundError(
            f"Variable '{variable_name}' not found. Available: {available}"
        )

    return dataset[variable_name]


def _find_coordinate(
    *, dataset: xr.Dataset, candidate_names: list[str]
) -> tuple[str, xr.DataArray] | None:
    """Find a coordinate by checking a list of candidate names."""
    for name in candidate_names:
        if name in dataset.coords:
            return name, dataset.coords[name]
    return None


def _extract_valid_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
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


def _build_coordinate_info(*, coord: xr.DataArray, name: str) -> dict:
    """Build a dictionary with coordinate information."""
    return {
        "dtype": str(coord.dtype),
        "shape": coord.shape,
        "attrs": dict(coord.attrs),
    }


def _build_variable_info(*, var: xr.DataArray, name: str) -> dict:
    """Build a dictionary with variable information."""
    return {
        "dtype": str(var.dtype),
        "shape": var.shape,
        "dims": var.dims,
        "attrs": dict(var.attrs),
    }


@mcp.tool()
def get_file_structure(file_path: str) -> str:
    """Get the complete structure of a NetCDF file.

    Returns dimensions, coordinates, data variables, and global attributes.

    Args:
        file_path: Path to the NetCDF file.

    Returns:
        JSON string containing the file structure.
    """
    try:
        path = _validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return _error_response(message=e.message)

    try:
        with _open_dataset(path=path) as ds:
            structure = {
                "file": str(path),
                "dimensions": dict(ds.dims.items()),
                "coordinates": {
                    name: _build_coordinate_info(coord=coord, name=name)
                    for name, coord in ds.coords.items()
                },
                "data_variables": {
                    name: _build_variable_info(var=var, name=name)
                    for name, var in ds.data_vars.items()
                },
                "global_attributes": dict(ds.attrs),
            }
            return _json_response(data=structure)

    except Exception as e:
        return _error_response(message=f"Failed to read NetCDF file: {e}")


@mcp.tool()
def list_variables(file_path: str) -> str:
    """List all data variables in a NetCDF file with their shapes, dtypes, and units.

    Args:
        file_path: Path to the NetCDF file.

    Returns:
        JSON string with variable names, shapes, dtypes, and units.
    """
    try:
        path = _validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return _error_response(message=e.message)

    try:
        with _open_dataset(path=path) as ds:
            variables = [
                {
                    "name": name,
                    "dtype": str(var.dtype),
                    "shape": var.shape,
                    "dims": var.dims,
                    "units": var.attrs.get("units", "unknown"),
                    "long_name": var.attrs.get("long_name", ""),
                }
                for name, var in ds.data_vars.items()
            ]
            return _json_response(data={"file": str(path), "variables": variables})

    except Exception as e:
        return _error_response(message=f"Failed to read NetCDF file: {e}")


@mcp.tool()
def get_global_attributes(file_path: str) -> str:
    """Get all global attributes (metadata) from a NetCDF file.

    Returns title, institution, history, conventions, and other file-level metadata.

    Args:
        file_path: Path to the NetCDF file.

    Returns:
        JSON string with all global attributes.
    """
    try:
        path = _validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return _error_response(message=e.message)

    try:
        with _open_dataset(path=path) as ds:
            return _json_response(
                data={"file": str(path), "global_attributes": dict(ds.attrs)}
            )

    except Exception as e:
        return _error_response(message=f"Failed to read NetCDF file: {e}")


@mcp.tool()
def get_variable_info(file_path: str, variable_name: str) -> str:
    """Get detailed information about a specific variable in a NetCDF file.

    Returns dtype, shape, dimensions, all attributes, and fill value.

    Args:
        file_path: Path to the NetCDF file.
        variable_name: Name of the variable to inspect.

    Returns:
        JSON string with detailed variable information.
    """
    try:
        path = _validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return _error_response(message=e.message)

    try:
        with _open_dataset(path=path) as ds:
            try:
                var = _get_variable(dataset=ds, variable_name=variable_name)
            except VariableNotFoundError as e:
                return _error_response(message=e.message)

            info = {
                "name": variable_name,
                "dtype": str(var.dtype),
                "shape": var.shape,
                "dims": var.dims,
                "size": var.size,
                "nbytes": var.nbytes,
                "attrs": dict(var.attrs),
                "fill_value": var.attrs.get("_FillValue", var.attrs.get("missing_value")),
                "is_coordinate": variable_name in ds.coords,
            }
            return _json_response(data=info)

    except Exception as e:
        return _error_response(message=f"Failed to read variable: {e}")


@mcp.tool()
def get_variable_stats(file_path: str, variable_name: str) -> str:
    """Get statistical summary of a variable: min, max, mean, std, NaN count, percentiles.

    Args:
        file_path: Path to the NetCDF file.
        variable_name: Name of the variable to analyze.

    Returns:
        JSON string with statistical summary.
    """
    try:
        path = _validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return _error_response(message=e.message)

    try:
        with _open_dataset(path=path) as ds:
            try:
                var = _get_variable(dataset=ds, variable_name=variable_name)
            except VariableNotFoundError as e:
                return _error_response(message=e.message)

            _, valid_values, nan_count = _extract_valid_values(var.values)

            stats = {
                "name": variable_name,
                "shape": var.shape,
                "total_elements": int(var.size),
                "nan_count": nan_count,
                "valid_count": int(len(valid_values)),
            }

            if len(valid_values) == 0:
                stats["warning"] = "No valid (non-NaN) values found"
                return _json_response(data=stats)

            stats.update({
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
                "mean": float(np.mean(valid_values)),
                "std": float(np.std(valid_values)),
                "percentiles": {
                    "1%": float(np.percentile(valid_values, 1)),
                    "25%": float(np.percentile(valid_values, 25)),
                    "50%": float(np.percentile(valid_values, 50)),
                    "75%": float(np.percentile(valid_values, 75)),
                    "99%": float(np.percentile(valid_values, 99)),
                },
            })
            return _json_response(data=stats)

    except Exception as e:
        return _error_response(message=f"Failed to compute statistics: {e}")


@mcp.tool()
def get_variable_sample(
    file_path: str, variable_name: str, max_samples: int = 100
) -> str:
    """Extract a sample of actual data values from a variable for inspection.

    Args:
        file_path: Path to the NetCDF file.
        variable_name: Name of the variable to sample.
        max_samples: Maximum number of samples to return (default: 100).

    Returns:
        JSON string with sample values and their indices.
    """
    try:
        path = _validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return _error_response(message=e.message)

    try:
        with _open_dataset(path=path) as ds:
            try:
                var = _get_variable(dataset=ds, variable_name=variable_name)
            except VariableNotFoundError as e:
                return _error_response(message=e.message)

            values = var.values.flatten()
            total = len(values)

            if total <= max_samples:
                indices = list(range(total))
                sampled = values.tolist()
            else:
                indices = np.linspace(0, total - 1, max_samples, dtype=int).tolist()
                sampled = values[indices].tolist()

            result = {
                "name": variable_name,
                "shape": var.shape,
                "total_elements": total,
                "samples_returned": len(sampled),
                "sample_indices": indices,
                "sample_values": sampled,
            }
            return _json_response(data=result)

    except Exception as e:
        return _error_response(message=f"Failed to sample variable: {e}")


@mcp.tool()
def get_dimension_values(
    file_path: str, dimension_name: str, max_values: int = 100
) -> str:
    """Get values along a dimension (e.g., time steps, lat/lon coordinates).

    Args:
        file_path: Path to the NetCDF file.
        dimension_name: Name of the dimension/coordinate to get values for.
        max_values: Maximum number of values to return (default: 100).

    Returns:
        JSON string with dimension values.
    """
    try:
        path = _validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return _error_response(message=e.message)

    try:
        with _open_dataset(path=path) as ds:
            if dimension_name not in ds.coords:
                available = list(ds.coords.keys())
                return _error_response(
                    message=f"Dimension '{dimension_name}' not found. Available: {available}"
                )

            coord = ds.coords[dimension_name]
            values = coord.values
            total = len(values)

            if total <= max_values:
                output_values = values.tolist()
            else:
                indices = np.linspace(0, total - 1, max_values, dtype=int)
                output_values = values[indices].tolist()

            result = {
                "name": dimension_name,
                "dtype": str(coord.dtype),
                "total_values": total,
                "values_returned": len(output_values),
                "values": output_values,
                "attrs": dict(coord.attrs),
            }
            return _json_response(data=result)

    except Exception as e:
        return _error_response(message=f"Failed to get dimension values: {e}")


def _check_physical_constraints(
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


def _compute_quality_stats(
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


@mcp.tool()
def check_data_quality(file_path: str, variable_name: str) -> str:
    """Check a variable for data quality issues: NaN/missing values, outliers, suspicious patterns.

    Args:
        file_path: Path to the NetCDF file.
        variable_name: Name of the variable to check.

    Returns:
        JSON string with data quality report.
    """
    try:
        path = _validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return _error_response(message=e.message)

    try:
        with _open_dataset(path=path) as ds:
            try:
                var = _get_variable(dataset=ds, variable_name=variable_name)
            except VariableNotFoundError as e:
                return _error_response(message=e.message)

            values_flat, valid_values, nan_count = _extract_valid_values(var.values)
            total = len(values_flat)
            nan_percent = (nan_count / total) * 100 if total > 0 else 0

            issues: list[str] = []
            warnings: list[str] = []

            if nan_count > 0:
                msg = f"Contains {nan_count} NaN values ({nan_percent:.1f}%)"
                if nan_percent > 50:
                    issues.append(f"High missing data: {nan_percent:.1f}% NaN values")
                else:
                    warnings.append(msg)

            quality_stats = _compute_quality_stats(
                valid_values=valid_values, issues=issues, warnings=warnings
            )

            if quality_stats:
                physical_warnings = _check_physical_constraints(
                    var=var,
                    min_val=quality_stats["min"],
                    max_val=quality_stats["max"],
                )
                warnings.extend(physical_warnings)

            report = {
                "name": variable_name,
                "shape": var.shape,
                "total_elements": total,
                "nan_count": nan_count,
                "nan_percent": round(nan_percent, 2),
                "issues": issues,
                "warnings": warnings,
                "status": "OK" if not issues else "ISSUES_FOUND",
                "stats": quality_stats,
            }
            return _json_response(data=report)

    except Exception as e:
        return _error_response(message=f"Failed to check data quality: {e}")


def _compute_correlation(v1: np.ndarray, v2: np.ndarray) -> dict:
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


@mcp.tool()
def compare_variables(file_path: str, variable1: str, variable2: str) -> str:
    """Compare two variables for shape compatibility, range differences, and correlation.

    Args:
        file_path: Path to the NetCDF file.
        variable1: Name of the first variable.
        variable2: Name of the second variable.

    Returns:
        JSON string with comparison results.
    """
    try:
        path = _validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return _error_response(message=e.message)

    try:
        with _open_dataset(path=path) as ds:
            try:
                var1 = _get_variable(dataset=ds, variable_name=variable1)
            except VariableNotFoundError as e:
                return _error_response(message=e.message)

            try:
                var2 = _get_variable(dataset=ds, variable_name=variable2)
            except VariableNotFoundError as e:
                return _error_response(message=e.message)

            comparison = {
                "variable1": {
                    "name": variable1,
                    "shape": var1.shape,
                    "dims": var1.dims,
                    "dtype": str(var1.dtype),
                    "units": var1.attrs.get("units", "unknown"),
                },
                "variable2": {
                    "name": variable2,
                    "shape": var2.shape,
                    "dims": var2.dims,
                    "dtype": str(var2.dtype),
                    "units": var2.attrs.get("units", "unknown"),
                },
                "compatible": var1.shape == var2.shape,
                "same_dimensions": var1.dims == var2.dims,
                "shared_dims": list(set(var1.dims) & set(var2.dims)),
            }

            if var1.shape == var2.shape:
                correlation_result = _compute_correlation(
                    var1.values.flatten(), var2.values.flatten()
                )
                comparison.update(correlation_result)

            return _json_response(data=comparison)

    except Exception as e:
        return _error_response(message=f"Failed to compare variables: {e}")


TIME_COORDINATE_NAMES = ["time", "Time", "TIME", "t"]
LAT_COORDINATE_NAMES = ["lat", "latitude", "Lat", "Latitude", "LAT", "LATITUDE", "y"]
LON_COORDINATE_NAMES = ["lon", "longitude", "Lon", "Longitude", "LON", "LONGITUDE", "x"]


def _find_time_coordinate(dataset: xr.Dataset) -> tuple[str, xr.DataArray] | None:
    """Find time coordinate by checking standard names and attributes."""
    result = _find_coordinate(dataset=dataset, candidate_names=TIME_COORDINATE_NAMES)
    if result:
        return result

    for name, coord in dataset.coords.items():
        if "time" in coord.attrs.get("long_name", "").lower():
            return name, coord

    return None


def _build_time_range_result(
    *, time_name: str, time_var: xr.DataArray
) -> dict:
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


@mcp.tool()
def get_time_range(file_path: str) -> str:
    """Get the time range of a NetCDF file in human-readable format.

    Handles CF-compliant time coordinates with various calendars.

    Args:
        file_path: Path to the NetCDF file.

    Returns:
        JSON string with time range information.
    """
    try:
        path = _validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return _error_response(message=e.message)

    try:
        with _open_dataset(path=path, decode_times=True) as ds:
            time_result = _find_time_coordinate(ds)
            if not time_result:
                return _error_response(
                    message=f"No time coordinate found. Available coords: {list(ds.coords.keys())}"
                )

            time_name, time_var = time_result
            result = _build_time_range_result(time_name=time_name, time_var=time_var)
            return _json_response(data=result)

    except Exception as e:
        return _get_time_range_fallback(path=path, original_error=e)


def _get_time_range_fallback(*, path: Path, original_error: Exception) -> str:
    """Fallback for time range when time decoding fails."""
    try:
        with _open_dataset(path=path, decode_times=False) as ds:
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
                return _json_response(data=result)

    except Exception:
        pass

    return _error_response(message=f"Failed to get time range: {original_error}")


def _build_coordinate_bounds(
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


@mcp.tool()
def get_spatial_bounds(file_path: str) -> str:
    """Get the geographic extent (lat/lon bounds) of a NetCDF file.

    Args:
        file_path: Path to the NetCDF file.

    Returns:
        JSON string with spatial bounds.
    """
    try:
        path = _validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return _error_response(message=e.message)

    try:
        with _open_dataset(path=path) as ds:
            result: dict[str, Any] = {"file": str(path)}

            lat_result = _find_coordinate(
                dataset=ds, candidate_names=LAT_COORDINATE_NAMES
            )
            lon_result = _find_coordinate(
                dataset=ds, candidate_names=LON_COORDINATE_NAMES
            )

            if lat_result:
                lat_name, lat_var = lat_result
                result["latitude"] = _build_coordinate_bounds(
                    coord=lat_var, name=lat_name, default_units="degrees_north"
                )
            else:
                result["latitude"] = None

            if lon_result:
                lon_name, lon_var = lon_result
                result["longitude"] = _build_coordinate_bounds(
                    coord=lon_var, name=lon_name, default_units="degrees_east"
                )
            else:
                result["longitude"] = None

            if lat_result and lon_result:
                lat_range = result["latitude"]["max"] - result["latitude"]["min"]
                lon_range = result["longitude"]["max"] - result["longitude"]["min"]
                result["coverage"] = {
                    "lat_range": lat_range,
                    "lon_range": lon_range,
                    "is_global": lat_range >= 170 and lon_range >= 350,
                }

            return _json_response(data=result)

    except Exception as e:
        return _error_response(message=f"Failed to get spatial bounds: {e}")


def _check_variable_attributes(
    *, dataset: xr.Dataset
) -> tuple[list[str], list[str]]:
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


def _check_time_compliance(
    *, dataset: xr.Dataset
) -> tuple[list[str], list[str], list[str]]:
    """Check time coordinate for CF compliance."""
    passed = []
    issues = []
    warnings = []

    time_result = _find_time_coordinate(dataset)
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


@mcp.tool()
def check_cf_compliance(file_path: str) -> str:
    """Check if a NetCDF file follows CF (Climate and Forecast) conventions.

    Performs basic checks for common CF requirements.

    Args:
        file_path: Path to the NetCDF file.

    Returns:
        JSON string with compliance report.
    """
    try:
        path = _validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return _error_response(message=e.message)

    try:
        with _open_dataset(path=path) as ds:
            issues: list[str] = []
            warnings: list[str] = []
            passed: list[str] = []

            conventions = ds.attrs.get("Conventions", ds.attrs.get("conventions", ""))
            if not conventions:
                issues.append("Missing 'Conventions' global attribute")
            elif "CF" in conventions:
                passed.append(f"Conventions attribute present: {conventions}")
            else:
                warnings.append(
                    f"Conventions attribute exists but doesn't mention CF: {conventions}"
                )

            _, attr_warnings = _check_variable_attributes(dataset=ds)
            warnings.extend(attr_warnings)

            time_passed, time_issues, time_warnings = _check_time_compliance(dataset=ds)
            passed.extend(time_passed)
            issues.extend(time_issues)
            warnings.extend(time_warnings)

            report = {
                "file": str(path),
                "conventions": conventions or "Not specified",
                "is_cf_compliant": len(issues) == 0,
                "passed_checks": passed,
                "issues": issues,
                "warnings": warnings,
                "num_variables": len(ds.data_vars),
                "num_coordinates": len(ds.coords),
            }
            return _json_response(data=report)

    except Exception as e:
        return _error_response(message=f"Failed to check CF compliance: {e}")


def _build_variable_explanation(
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
    valid_max = attrs.get(
        "valid_max", valid_range[1] if len(valid_range) > 1 else None
    )
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


@mcp.tool()
def explain_variable(file_path: str, variable_name: str) -> str:
    """Generate a human-readable explanation of what a variable represents.

    Uses standard_name, long_name, units, and other attributes.

    Args:
        file_path: Path to the NetCDF file.
        variable_name: Name of the variable to explain.

    Returns:
        JSON string with human-readable explanation.
    """
    try:
        path = _validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return _error_response(message=e.message)

    try:
        with _open_dataset(path=path) as ds:
            try:
                var = _get_variable(dataset=ds, variable_name=variable_name)
            except VariableNotFoundError as e:
                return _error_response(message=e.message)

            result = _build_variable_explanation(
                var=var,
                variable_name=variable_name,
                is_coordinate=variable_name in ds.coords,
            )
            return _json_response(data=result)

    except Exception as e:
        return _error_response(message=f"Failed to explain variable: {e}")


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
