"""Custom exceptions for NetCDF operations."""


class NetCDFError(Exception):
    """Base exception for NetCDF operations."""

    def __init__(self, message: str, extra: dict | None = None):
        self.message = message
        self.extra = extra or {}
        super().__init__(message)


class FileNotFoundError(NetCDFError):
    """Raised when the specified file does not exist."""


class InvalidFileError(NetCDFError):
    """Raised when the file is not a valid NetCDF file."""


class VariableNotFoundError(NetCDFError):
    """Raised when a requested variable is not found in the dataset."""


class DimensionNotFoundError(NetCDFError):
    """Raised when a requested dimension is not found in the dataset."""


class TimeCoordinateNotFoundError(NetCDFError):
    """Raised when no time coordinate is found in the dataset."""
