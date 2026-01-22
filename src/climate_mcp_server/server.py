"""Climate NetCDF MCP Server - reads and understands NetCDF files for debugging."""

import numpy as np
from mcp.server.fastmcp import FastMCP

from climate_mcp_server.analysis import (
    check_physical_constraints,
    check_time_compliance,
    check_variable_attributes,
    compute_correlation,
    compute_quality_stats,
    extract_valid_values,
    get_time_range_fallback,
)
from climate_mcp_server.dataset import (
    LAT_COORDINATE_NAMES,
    LON_COORDINATE_NAMES,
    find_coordinate,
    find_time_coordinate,
    get_variable,
    open_dataset,
    validate_netcdf_path,
)
from climate_mcp_server.exceptions import NetCDFError, VariableNotFoundError
from climate_mcp_server.response import (
    build_coordinate_bounds,
    build_coordinate_info,
    build_time_range_result,
    build_variable_explanation,
    build_variable_info,
    error_response,
    json_response,
)

mcp = FastMCP("climate-netcdf")


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
        path = validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return error_response(message=e.message)

    try:
        with open_dataset(path=path) as ds:
            structure = {
                "file": str(path),
                "dimensions": dict(ds.dims.items()),
                "coordinates": {
                    name: build_coordinate_info(coord=coord, name=name)
                    for name, coord in ds.coords.items()
                },
                "data_variables": {
                    name: build_variable_info(var=var, name=name)
                    for name, var in ds.data_vars.items()
                },
                "global_attributes": dict(ds.attrs),
            }
            return json_response(data=structure)

    except Exception as e:
        return error_response(message=f"Failed to read NetCDF file: {e}")


@mcp.tool()
def list_variables(file_path: str) -> str:
    """List all data variables in a NetCDF file with their shapes, dtypes, and units.

    Args:
        file_path: Path to the NetCDF file.

    Returns:
        JSON string with variable names, shapes, dtypes, and units.
    """
    try:
        path = validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return error_response(message=e.message)

    try:
        with open_dataset(path=path) as ds:
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
            return json_response(data={"file": str(path), "variables": variables})

    except Exception as e:
        return error_response(message=f"Failed to read NetCDF file: {e}")


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
        path = validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return error_response(message=e.message)

    try:
        with open_dataset(path=path) as ds:
            return json_response(
                data={"file": str(path), "global_attributes": dict(ds.attrs)}
            )

    except Exception as e:
        return error_response(message=f"Failed to read NetCDF file: {e}")


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
        path = validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return error_response(message=e.message)

    try:
        with open_dataset(path=path) as ds:
            try:
                var = get_variable(dataset=ds, variable_name=variable_name)
            except VariableNotFoundError as e:
                return error_response(message=e.message)

            info = {
                "name": variable_name,
                "dtype": str(var.dtype),
                "shape": var.shape,
                "dims": var.dims,
                "size": var.size,
                "nbytes": var.nbytes,
                "attrs": dict(var.attrs),
                "fill_value": var.attrs.get(
                    "_FillValue", var.attrs.get("missing_value")
                ),
                "is_coordinate": variable_name in ds.coords,
            }
            return json_response(data=info)

    except Exception as e:
        return error_response(message=f"Failed to read variable: {e}")


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
        path = validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return error_response(message=e.message)

    try:
        with open_dataset(path=path) as ds:
            try:
                var = get_variable(dataset=ds, variable_name=variable_name)
            except VariableNotFoundError as e:
                return error_response(message=e.message)

            _, valid_values, nan_count = extract_valid_values(var.values)

            stats = {
                "name": variable_name,
                "shape": var.shape,
                "total_elements": int(var.size),
                "nan_count": nan_count,
                "valid_count": int(len(valid_values)),
            }

            if len(valid_values) == 0:
                stats["warning"] = "No valid (non-NaN) values found"
                return json_response(data=stats)

            stats.update(
                {
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
                }
            )
            return json_response(data=stats)

    except Exception as e:
        return error_response(message=f"Failed to compute statistics: {e}")


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
        path = validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return error_response(message=e.message)

    try:
        with open_dataset(path=path) as ds:
            try:
                var = get_variable(dataset=ds, variable_name=variable_name)
            except VariableNotFoundError as e:
                return error_response(message=e.message)

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
            return json_response(data=result)

    except Exception as e:
        return error_response(message=f"Failed to sample variable: {e}")


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
        path = validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return error_response(message=e.message)

    try:
        with open_dataset(path=path) as ds:
            if dimension_name not in ds.coords:
                available = list(ds.coords.keys())
                return error_response(
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
            return json_response(data=result)

    except Exception as e:
        return error_response(message=f"Failed to get dimension values: {e}")


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
        path = validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return error_response(message=e.message)

    try:
        with open_dataset(path=path) as ds:
            try:
                var = get_variable(dataset=ds, variable_name=variable_name)
            except VariableNotFoundError as e:
                return error_response(message=e.message)

            values_flat, valid_values, nan_count = extract_valid_values(var.values)
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

            quality_stats = compute_quality_stats(
                valid_values=valid_values, issues=issues, warnings=warnings
            )

            if quality_stats:
                physical_warnings = check_physical_constraints(
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
            return json_response(data=report)

    except Exception as e:
        return error_response(message=f"Failed to check data quality: {e}")


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
        path = validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return error_response(message=e.message)

    try:
        with open_dataset(path=path) as ds:
            try:
                var1 = get_variable(dataset=ds, variable_name=variable1)
            except VariableNotFoundError as e:
                return error_response(message=e.message)

            try:
                var2 = get_variable(dataset=ds, variable_name=variable2)
            except VariableNotFoundError as e:
                return error_response(message=e.message)

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
                correlation_result = compute_correlation(
                    var1.values.flatten(), var2.values.flatten()
                )
                comparison.update(correlation_result)

            return json_response(data=comparison)

    except Exception as e:
        return error_response(message=f"Failed to compare variables: {e}")


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
        path = validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return error_response(message=e.message)

    try:
        with open_dataset(path=path, decode_times=True) as ds:
            time_result = find_time_coordinate(ds)
            if not time_result:
                return error_response(
                    message=f"No time coordinate found. Available coords: {list(ds.coords.keys())}"
                )

            time_name, time_var = time_result
            result = build_time_range_result(time_name=time_name, time_var=time_var)
            return json_response(data=result)

    except Exception as e:
        return get_time_range_fallback(path=path, original_error=e)


@mcp.tool()
def get_spatial_bounds(file_path: str) -> str:
    """Get the geographic extent (lat/lon bounds) of a NetCDF file.

    Args:
        file_path: Path to the NetCDF file.

    Returns:
        JSON string with spatial bounds.
    """
    try:
        path = validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return error_response(message=e.message)

    try:
        with open_dataset(path=path) as ds:
            result: dict = {"file": str(path)}

            lat_result = find_coordinate(
                dataset=ds, candidate_names=LAT_COORDINATE_NAMES
            )
            lon_result = find_coordinate(
                dataset=ds, candidate_names=LON_COORDINATE_NAMES
            )

            if lat_result:
                lat_name, lat_var = lat_result
                result["latitude"] = build_coordinate_bounds(
                    coord=lat_var, name=lat_name, default_units="degrees_north"
                )
            else:
                result["latitude"] = None

            if lon_result:
                lon_name, lon_var = lon_result
                result["longitude"] = build_coordinate_bounds(
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

            return json_response(data=result)

    except Exception as e:
        return error_response(message=f"Failed to get spatial bounds: {e}")


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
        path = validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return error_response(message=e.message)

    try:
        with open_dataset(path=path) as ds:
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

            _, attr_warnings = check_variable_attributes(dataset=ds)
            warnings.extend(attr_warnings)

            time_passed, time_issues, time_warnings = check_time_compliance(dataset=ds)
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
            return json_response(data=report)

    except Exception as e:
        return error_response(message=f"Failed to check CF compliance: {e}")


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
        path = validate_netcdf_path(file_path=file_path)
    except NetCDFError as e:
        return error_response(message=e.message)

    try:
        with open_dataset(path=path) as ds:
            try:
                var = get_variable(dataset=ds, variable_name=variable_name)
            except VariableNotFoundError as e:
                return error_response(message=e.message)

            result = build_variable_explanation(
                var=var,
                variable_name=variable_name,
                is_coordinate=variable_name in ds.coords,
            )
            return json_response(data=result)

    except Exception as e:
        return error_response(message=f"Failed to explain variable: {e}")


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
