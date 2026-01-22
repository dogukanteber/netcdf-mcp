"""Climate NetCDF MCP Server.

An MCP server that reads and understands NetCDF files for debugging purposes.
"""

import json
from pathlib import Path

import xarray as xr
from mcp.server.fastmcp import FastMCP

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
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return json.dumps({"error": f"File not found: {path}"})

    if not path.suffix.lower() in (".nc", ".nc4", ".netcdf"):
        return json.dumps({"error": f"Not a NetCDF file: {path}"})

    try:
        ds = xr.open_dataset(path)

        structure = {
            "file": str(path),
            "dimensions": {name: size for name, size in ds.dims.items()},
            "coordinates": {},
            "data_variables": {},
            "global_attributes": dict(ds.attrs),
        }

        # Coordinates
        for name, coord in ds.coords.items():
            structure["coordinates"][name] = {
                "dtype": str(coord.dtype),
                "shape": coord.shape,
                "attrs": dict(coord.attrs),
            }

        # Data variables
        for name, var in ds.data_vars.items():
            structure["data_variables"][name] = {
                "dtype": str(var.dtype),
                "shape": var.shape,
                "dims": var.dims,
                "attrs": dict(var.attrs),
            }

        ds.close()
        return json.dumps(structure, indent=2, default=str)

    except Exception as e:
        return json.dumps({"error": f"Failed to read NetCDF file: {e}"})


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
