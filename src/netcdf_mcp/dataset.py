"""Dataset operations for NetCDF files."""

from contextlib import contextmanager
from pathlib import Path

import xarray as xr

from netcdf_mcp.exceptions import (
    FileNotFoundError,
    InvalidFileError,
    VariableNotFoundError,
)

TIME_COORDINATE_NAMES = ["time", "Time", "TIME", "t"]
LAT_COORDINATE_NAMES = ["lat", "latitude", "Lat", "Latitude", "LAT", "LATITUDE", "y"]
LON_COORDINATE_NAMES = ["lon", "longitude", "Lon", "Longitude", "LON", "LONGITUDE", "x"]


def validate_netcdf_path(*, file_path: str) -> Path:
    """Validate that the path exists and is a NetCDF file."""
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in (".nc", ".nc4", ".netcdf"):
        raise InvalidFileError(f"Not a NetCDF file: {path}")

    return path


@contextmanager
def open_dataset(*, path: Path, decode_times: bool = True):
    """Context manager for safely opening and closing xarray datasets."""
    dataset = xr.open_dataset(path, decode_times=decode_times)
    try:
        yield dataset
    finally:
        dataset.close()


def get_variable(
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


def find_coordinate(
    *, dataset: xr.Dataset, candidate_names: list[str]
) -> tuple[str, xr.DataArray] | None:
    """Find a coordinate by checking a list of candidate names."""
    for name in candidate_names:
        if name in dataset.coords:
            return name, dataset.coords[name]
    return None


def find_time_coordinate(dataset: xr.Dataset) -> tuple[str, xr.DataArray] | None:
    """Find time coordinate by checking standard names and attributes."""
    result = find_coordinate(dataset=dataset, candidate_names=TIME_COORDINATE_NAMES)
    if result:
        return result

    for name, coord in dataset.coords.items():
        if "time" in coord.attrs.get("long_name", "").lower():
            return name, coord

    return None
