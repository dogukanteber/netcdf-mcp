"""Climate NetCDF MCP Server.

An MCP server that reads and understands NetCDF files for debugging purposes.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("climate-netcdf")


# =============================================================================
# Helper Functions
# =============================================================================


def _validate_netcdf_path(file_path: str) -> tuple[Path | None, str | None]:
    """Validate that the path exists and is a NetCDF file.

    Returns:
        (path, None) if valid, (None, error_message) if invalid.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return None, f"File not found: {path}"

    if path.suffix.lower() not in (".nc", ".nc4", ".netcdf"):
        return None, f"Not a NetCDF file: {path}"

    return path, None


def _json_response(data: Any) -> str:
    """Convert data to JSON string, handling numpy types."""
    return json.dumps(data, indent=2, default=str)


def _error_response(message: str) -> str:
    """Return a JSON error response."""
    return _json_response({"error": message})


# =============================================================================
# File Structure & Metadata Tools
# =============================================================================


@mcp.tool()
def get_file_structure(file_path: str) -> str:
    """Get the complete structure of a NetCDF file.

    Returns dimensions, coordinates, data variables, and global attributes.

    Args:
        file_path: Path to the NetCDF file.

    Returns:
        JSON string containing the file structure.
    """
    path, error = _validate_netcdf_path(file_path)
    if error:
        return _error_response(error)

    try:
        ds = xr.open_dataset(path)

        structure = {
            "file": str(path),
            "dimensions": {name: size for name, size in ds.dims.items()},
            "coordinates": {},
            "data_variables": {},
            "global_attributes": dict(ds.attrs),
        }

        for name, coord in ds.coords.items():
            structure["coordinates"][name] = {
                "dtype": str(coord.dtype),
                "shape": coord.shape,
                "attrs": dict(coord.attrs),
            }

        for name, var in ds.data_vars.items():
            structure["data_variables"][name] = {
                "dtype": str(var.dtype),
                "shape": var.shape,
                "dims": var.dims,
                "attrs": dict(var.attrs),
            }

        ds.close()
        return _json_response(structure)

    except Exception as e:
        return _error_response(f"Failed to read NetCDF file: {e}")


@mcp.tool()
def list_variables(file_path: str) -> str:
    """List all data variables in a NetCDF file with their shapes, dtypes, and units.

    Args:
        file_path: Path to the NetCDF file.

    Returns:
        JSON string with variable names, shapes, dtypes, and units.
    """
    path, error = _validate_netcdf_path(file_path)
    if error:
        return _error_response(error)

    try:
        ds = xr.open_dataset(path)

        variables = []
        for name, var in ds.data_vars.items():
            variables.append({
                "name": name,
                "dtype": str(var.dtype),
                "shape": var.shape,
                "dims": var.dims,
                "units": var.attrs.get("units", "unknown"),
                "long_name": var.attrs.get("long_name", ""),
            })

        ds.close()
        return _json_response({"file": str(path), "variables": variables})

    except Exception as e:
        return _error_response(f"Failed to read NetCDF file: {e}")


@mcp.tool()
def get_global_attributes(file_path: str) -> str:
    """Get all global attributes (metadata) from a NetCDF file.

    Returns title, institution, history, conventions, and other file-level metadata.

    Args:
        file_path: Path to the NetCDF file.

    Returns:
        JSON string with all global attributes.
    """
    path, error = _validate_netcdf_path(file_path)
    if error:
        return _error_response(error)

    try:
        ds = xr.open_dataset(path)
        attrs = dict(ds.attrs)
        ds.close()
        return _json_response({"file": str(path), "global_attributes": attrs})

    except Exception as e:
        return _error_response(f"Failed to read NetCDF file: {e}")


# =============================================================================
# Variable Inspection Tools
# =============================================================================


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
    path, error = _validate_netcdf_path(file_path)
    if error:
        return _error_response(error)

    try:
        ds = xr.open_dataset(path)

        if variable_name not in ds.data_vars and variable_name not in ds.coords:
            ds.close()
            available = list(ds.data_vars.keys()) + list(ds.coords.keys())
            return _error_response(
                f"Variable '{variable_name}' not found. Available: {available}"
            )

        var = ds[variable_name]
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

        ds.close()
        return _json_response(info)

    except Exception as e:
        return _error_response(f"Failed to read variable: {e}")


@mcp.tool()
def get_variable_stats(file_path: str, variable_name: str) -> str:
    """Get statistical summary of a variable: min, max, mean, std, NaN count, percentiles.

    Args:
        file_path: Path to the NetCDF file.
        variable_name: Name of the variable to analyze.

    Returns:
        JSON string with statistical summary.
    """
    path, error = _validate_netcdf_path(file_path)
    if error:
        return _error_response(error)

    try:
        ds = xr.open_dataset(path)

        if variable_name not in ds.data_vars and variable_name not in ds.coords:
            ds.close()
            return _error_response(f"Variable '{variable_name}' not found.")

        var = ds[variable_name]
        values = var.values

        # Handle masked arrays
        if hasattr(values, "mask"):
            values = np.ma.filled(values, np.nan)

        values_flat = values.flatten()
        valid_values = values_flat[~np.isnan(values_flat)]

        stats = {
            "name": variable_name,
            "shape": var.shape,
            "total_elements": int(var.size),
            "nan_count": int(np.sum(np.isnan(values_flat))),
            "valid_count": int(len(valid_values)),
        }

        if len(valid_values) > 0:
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
        else:
            stats["warning"] = "No valid (non-NaN) values found"

        ds.close()
        return _json_response(stats)

    except Exception as e:
        return _error_response(f"Failed to compute statistics: {e}")


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
    path, error = _validate_netcdf_path(file_path)
    if error:
        return _error_response(error)

    try:
        ds = xr.open_dataset(path)

        if variable_name not in ds.data_vars and variable_name not in ds.coords:
            ds.close()
            return _error_response(f"Variable '{variable_name}' not found.")

        var = ds[variable_name]
        values = var.values.flatten()

        # Sample evenly across the data
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

        ds.close()
        return _json_response(result)

    except Exception as e:
        return _error_response(f"Failed to sample variable: {e}")


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
    path, error = _validate_netcdf_path(file_path)
    if error:
        return _error_response(error)

    try:
        ds = xr.open_dataset(path)

        if dimension_name not in ds.coords:
            ds.close()
            available = list(ds.coords.keys())
            return _error_response(
                f"Dimension '{dimension_name}' not found. Available: {available}"
            )

        coord = ds.coords[dimension_name]
        values = coord.values

        total = len(values)
        if total <= max_values:
            output_values = values.tolist()
        else:
            # Return first, last, and evenly spaced values
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

        ds.close()
        return _json_response(result)

    except Exception as e:
        return _error_response(f"Failed to get dimension values: {e}")


# =============================================================================
# Data Analysis & Debugging Tools
# =============================================================================


@mcp.tool()
def check_data_quality(file_path: str, variable_name: str) -> str:
    """Check a variable for data quality issues: NaN/missing values, outliers, suspicious patterns.

    Args:
        file_path: Path to the NetCDF file.
        variable_name: Name of the variable to check.

    Returns:
        JSON string with data quality report.
    """
    path, error = _validate_netcdf_path(file_path)
    if error:
        return _error_response(error)

    try:
        ds = xr.open_dataset(path)

        if variable_name not in ds.data_vars and variable_name not in ds.coords:
            ds.close()
            return _error_response(f"Variable '{variable_name}' not found.")

        var = ds[variable_name]
        values = var.values

        if hasattr(values, "mask"):
            values = np.ma.filled(values, np.nan)

        values_flat = values.flatten()
        total = len(values_flat)
        nan_count = int(np.sum(np.isnan(values_flat)))
        valid_values = values_flat[~np.isnan(values_flat)]

        issues = []
        warnings = []

        # Check for NaN/missing values
        nan_percent = (nan_count / total) * 100 if total > 0 else 0
        if nan_count > 0:
            if nan_percent > 50:
                issues.append(f"High missing data: {nan_percent:.1f}% NaN values")
            else:
                warnings.append(f"Contains {nan_count} NaN values ({nan_percent:.1f}%)")

        if len(valid_values) > 0:
            min_val = float(np.min(valid_values))
            max_val = float(np.max(valid_values))
            mean_val = float(np.mean(valid_values))
            std_val = float(np.std(valid_values))

            # Check for infinite values
            inf_count = int(np.sum(np.isinf(valid_values)))
            if inf_count > 0:
                issues.append(f"Contains {inf_count} infinite values")

            # Check for outliers (values beyond 5 std from mean)
            if std_val > 0:
                outlier_mask = np.abs(valid_values - mean_val) > 5 * std_val
                outlier_count = int(np.sum(outlier_mask))
                if outlier_count > 0:
                    warnings.append(
                        f"{outlier_count} potential outliers (>5 std from mean)"
                    )

            # Check for constant values
            if min_val == max_val:
                warnings.append("All values are identical (constant field)")

            # Check for suspicious values based on common variable types
            units = var.attrs.get("units", "").lower()
            if "kelvin" in units or units == "k":
                if min_val < 0:
                    issues.append(f"Negative Kelvin temperature: {min_val}")
                if max_val > 400:
                    warnings.append(f"Very high temperature: {max_val} K")
            elif "percent" in units or units == "%":
                if min_val < 0 or max_val > 100:
                    warnings.append(f"Percentage out of range: [{min_val}, {max_val}]")

            quality_stats = {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val,
            }
        else:
            quality_stats = None
            issues.append("No valid data values found")

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

        ds.close()
        return _json_response(report)

    except Exception as e:
        return _error_response(f"Failed to check data quality: {e}")


@mcp.tool()
def compare_variables(
    file_path: str, variable1: str, variable2: str
) -> str:
    """Compare two variables for shape compatibility, range differences, and correlation.

    Args:
        file_path: Path to the NetCDF file.
        variable1: Name of the first variable.
        variable2: Name of the second variable.

    Returns:
        JSON string with comparison results.
    """
    path, error = _validate_netcdf_path(file_path)
    if error:
        return _error_response(error)

    try:
        ds = xr.open_dataset(path)

        all_vars = list(ds.data_vars.keys()) + list(ds.coords.keys())
        if variable1 not in all_vars:
            ds.close()
            return _error_response(f"Variable '{variable1}' not found.")
        if variable2 not in all_vars:
            ds.close()
            return _error_response(f"Variable '{variable2}' not found.")

        var1 = ds[variable1]
        var2 = ds[variable2]

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

        # If shapes match, compute correlation
        if var1.shape == var2.shape:
            v1 = var1.values.flatten()
            v2 = var2.values.flatten()

            # Remove NaN pairs
            valid_mask = ~(np.isnan(v1) | np.isnan(v2))
            v1_valid = v1[valid_mask]
            v2_valid = v2[valid_mask]

            if len(v1_valid) > 1:
                correlation = float(np.corrcoef(v1_valid, v2_valid)[0, 1])
                comparison["correlation"] = (
                    correlation if not np.isnan(correlation) else None
                )
                comparison["valid_pairs"] = int(len(v1_valid))

        ds.close()
        return _json_response(comparison)

    except Exception as e:
        return _error_response(f"Failed to compare variables: {e}")


@mcp.tool()
def get_time_range(file_path: str) -> str:
    """Get the time range of a NetCDF file in human-readable format.

    Handles CF-compliant time coordinates with various calendars.

    Args:
        file_path: Path to the NetCDF file.

    Returns:
        JSON string with time range information.
    """
    path, error = _validate_netcdf_path(file_path)
    if error:
        return _error_response(error)

    try:
        ds = xr.open_dataset(path, decode_times=True)

        # Look for time coordinate
        time_var = None
        time_name = None
        for name in ["time", "Time", "TIME", "t"]:
            if name in ds.coords:
                time_var = ds.coords[name]
                time_name = name
                break

        if time_var is None:
            # Try to find any coordinate with time-like attributes
            for name, coord in ds.coords.items():
                if "time" in coord.attrs.get("long_name", "").lower():
                    time_var = coord
                    time_name = name
                    break

        if time_var is None:
            ds.close()
            return _error_response(
                "No time coordinate found. Available coords: "
                + str(list(ds.coords.keys()))
            )

        time_values = time_var.values
        result = {
            "time_coordinate": time_name,
            "num_timesteps": len(time_values),
            "dtype": str(time_var.dtype),
            "attrs": dict(time_var.attrs),
        }

        # Try to get human-readable times
        try:
            result["start"] = str(time_values[0])
            result["end"] = str(time_values[-1])

            # Calculate duration if possible
            if len(time_values) >= 2:
                result["first_few"] = [str(t) for t in time_values[:5]]
                result["last_few"] = [str(t) for t in time_values[-5:]]
        except Exception:
            result["start"] = str(time_values[0])
            result["end"] = str(time_values[-1])

        ds.close()
        return _json_response(result)

    except Exception as e:
        # Try again without decoding times
        try:
            ds = xr.open_dataset(path, decode_times=False)
            for name in ["time", "Time", "TIME", "t"]:
                if name in ds.coords:
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
                    ds.close()
                    return _json_response(result)
            ds.close()
        except Exception:
            pass

        return _error_response(f"Failed to get time range: {e}")


@mcp.tool()
def get_spatial_bounds(file_path: str) -> str:
    """Get the geographic extent (lat/lon bounds) of a NetCDF file.

    Args:
        file_path: Path to the NetCDF file.

    Returns:
        JSON string with spatial bounds.
    """
    path, error = _validate_netcdf_path(file_path)
    if error:
        return _error_response(error)

    try:
        ds = xr.open_dataset(path)

        result = {"file": str(path)}

        # Look for latitude
        lat_var = None
        lat_name = None
        for name in ["lat", "latitude", "Lat", "Latitude", "LAT", "LATITUDE", "y"]:
            if name in ds.coords:
                lat_var = ds.coords[name]
                lat_name = name
                break

        # Look for longitude
        lon_var = None
        lon_name = None
        for name in ["lon", "longitude", "Lon", "Longitude", "LON", "LONGITUDE", "x"]:
            if name in ds.coords:
                lon_var = ds.coords[name]
                lon_name = name
                break

        if lat_var is not None:
            lat_values = lat_var.values
            result["latitude"] = {
                "name": lat_name,
                "min": float(np.min(lat_values)),
                "max": float(np.max(lat_values)),
                "num_values": len(lat_values),
                "units": lat_var.attrs.get("units", "degrees_north"),
            }
        else:
            result["latitude"] = None

        if lon_var is not None:
            lon_values = lon_var.values
            result["longitude"] = {
                "name": lon_name,
                "min": float(np.min(lon_values)),
                "max": float(np.max(lon_values)),
                "num_values": len(lon_values),
                "units": lon_var.attrs.get("units", "degrees_east"),
            }
        else:
            result["longitude"] = None

        # Check for global coverage
        if lat_var is not None and lon_var is not None:
            lat_range = float(np.max(lat_values) - np.min(lat_values))
            lon_range = float(np.max(lon_values) - np.min(lon_values))
            result["coverage"] = {
                "lat_range": lat_range,
                "lon_range": lon_range,
                "is_global": lat_range >= 170 and lon_range >= 350,
            }

        ds.close()
        return _json_response(result)

    except Exception as e:
        return _error_response(f"Failed to get spatial bounds: {e}")


# =============================================================================
# Schema & Conventions Tools
# =============================================================================


@mcp.tool()
def check_cf_compliance(file_path: str) -> str:
    """Check if a NetCDF file follows CF (Climate and Forecast) conventions.

    Performs basic checks for common CF requirements.

    Args:
        file_path: Path to the NetCDF file.

    Returns:
        JSON string with compliance report.
    """
    path, error = _validate_netcdf_path(file_path)
    if error:
        return _error_response(error)

    try:
        ds = xr.open_dataset(path)

        issues = []
        warnings = []
        passed = []

        # Check for Conventions attribute
        conventions = ds.attrs.get("Conventions", ds.attrs.get("conventions", ""))
        if conventions:
            if "CF" in conventions:
                passed.append(f"Conventions attribute present: {conventions}")
            else:
                warnings.append(
                    f"Conventions attribute exists but doesn't mention CF: {conventions}"
                )
        else:
            issues.append("Missing 'Conventions' global attribute")

        # Check for required variable attributes
        for name, var in ds.data_vars.items():
            # Check for units
            if "units" not in var.attrs:
                warnings.append(f"Variable '{name}' missing 'units' attribute")

            # Check for long_name or standard_name
            if "long_name" not in var.attrs and "standard_name" not in var.attrs:
                warnings.append(
                    f"Variable '{name}' missing both 'long_name' and 'standard_name'"
                )

        # Check coordinate variables
        for name, coord in ds.coords.items():
            if "units" not in coord.attrs:
                if name.lower() not in ["time", "t"]:  # Time can have calendar instead
                    warnings.append(f"Coordinate '{name}' missing 'units' attribute")

        # Check time coordinate
        time_coord = None
        for name in ["time", "Time", "TIME", "t"]:
            if name in ds.coords:
                time_coord = ds.coords[name]
                break

        if time_coord is not None:
            if "units" in time_coord.attrs:
                passed.append(f"Time coordinate has units: {time_coord.attrs['units']}")
            else:
                issues.append("Time coordinate missing 'units' attribute")

            if "calendar" in time_coord.attrs:
                passed.append(
                    f"Time coordinate has calendar: {time_coord.attrs['calendar']}"
                )
            else:
                warnings.append(
                    "Time coordinate missing 'calendar' attribute (standard assumed)"
                )

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

        ds.close()
        return _json_response(report)

    except Exception as e:
        return _error_response(f"Failed to check CF compliance: {e}")


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
    path, error = _validate_netcdf_path(file_path)
    if error:
        return _error_response(error)

    try:
        ds = xr.open_dataset(path)

        if variable_name not in ds.data_vars and variable_name not in ds.coords:
            ds.close()
            available = list(ds.data_vars.keys()) + list(ds.coords.keys())
            return _error_response(
                f"Variable '{variable_name}' not found. Available: {available}"
            )

        var = ds[variable_name]
        attrs = dict(var.attrs)

        # Build explanation
        explanation_parts = []

        # Name info
        long_name = attrs.get("long_name", "")
        standard_name = attrs.get("standard_name", "")

        if long_name:
            explanation_parts.append(f"**{long_name}**")
        elif standard_name:
            # Convert standard_name underscores to spaces
            explanation_parts.append(f"**{standard_name.replace('_', ' ').title()}**")
        else:
            explanation_parts.append(f"**{variable_name}**")

        # Standard name reference
        if standard_name:
            explanation_parts.append(f"CF Standard Name: `{standard_name}`")

        # Units
        units = attrs.get("units", "")
        if units:
            explanation_parts.append(f"Units: {units}")

        # Shape and dimensions
        explanation_parts.append(f"Shape: {var.shape} with dimensions {var.dims}")

        # Valid range
        valid_min = attrs.get("valid_min", attrs.get("valid_range", [None])[0])
        valid_max = attrs.get(
            "valid_max",
            attrs.get("valid_range", [None, None])[1]
            if len(attrs.get("valid_range", [])) > 1
            else None,
        )
        if valid_min is not None or valid_max is not None:
            explanation_parts.append(f"Valid range: [{valid_min}, {valid_max}]")

        # Cell methods (for aggregated data)
        cell_methods = attrs.get("cell_methods", "")
        if cell_methods:
            explanation_parts.append(f"Cell methods: {cell_methods}")

        # Comment
        comment = attrs.get("comment", "")
        if comment:
            explanation_parts.append(f"Note: {comment}")

        result = {
            "variable": variable_name,
            "explanation": "\n".join(explanation_parts),
            "attributes": attrs,
            "dtype": str(var.dtype),
            "is_coordinate": variable_name in ds.coords,
        }

        ds.close()
        return _json_response(result)

    except Exception as e:
        return _error_response(f"Failed to explain variable: {e}")


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
