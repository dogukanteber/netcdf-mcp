# NetCDF MCP Server

An MCP server that reads and understands NetCDF files, designed to give coding agents context about scientific data files.

## Installation

```bash
uv pip install -e .
```

## Usage

### With Cursor/Claude

Add to your MCP configuration file (`~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "netcdf-mcp": {
      "command": "/path/to/uv",
      "args": ["run", "--directory", "/path/to/netcdf-mcp", "netcdf-mcp"]
    }
  }
}
```

Replace `/path/to/uv` with your `uv` binary location (e.g., `/opt/homebrew/bin/uv`) and `/path/to/netcdf-mcp` with where you cloned this repository.

### Standalone

```bash
netcdf-mcp
```

## Tools

| Tool | Description |
|------|-------------|
| `get_file_structure` | Get dimensions, coordinates, variables, and attributes |
| `list_variables` | List all data variables with shapes and units |
| `get_global_attributes` | Get file-level metadata |
| `get_variable_info` | Detailed info about a specific variable |
| `get_variable_stats` | Statistical summary (min, max, mean, percentiles) |
| `get_variable_sample` | Extract sample values for inspection |
| `get_dimension_values` | Get coordinate values (time, lat, lon) |
| `check_data_quality` | Check for NaN, outliers, and suspicious patterns |
| `compare_variables` | Compare two variables for compatibility and correlation |
| `get_time_range` | Get temporal extent in human-readable format |
| `get_spatial_bounds` | Get geographic extent (lat/lon bounds) |
| `check_cf_compliance` | Verify CF conventions compliance |
| `explain_variable` | Human-readable explanation of a variable |
