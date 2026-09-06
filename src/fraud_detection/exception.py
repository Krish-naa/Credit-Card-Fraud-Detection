"""Custom exception that captures file name and line number for clear debugging."""
import sys


def _error_detail(error: Exception) -> str:
    _, _, exc_tb = sys.exc_info()
    if exc_tb is None:
        return str(error)
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error in [{file_name}] line [{line_number}]: {error}"


class FraudException(Exception):
    """Raised for any error inside the fraud detection pipeline."""

    def __init__(self, error: Exception):
        message = _error_detail(error)
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message
