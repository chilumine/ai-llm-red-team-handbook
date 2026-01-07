"""
Centralized logging configuration for LLM red team scripts.

Provides consistent logging setup across all scripts with support
for file and console output, different log levels, and formatted output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# Color codes for console output
class LogColors:
    """ANSI color codes for log levels."""
    RESET = '\033[0m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    GRAY = '\033[90m'


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to console output.
    """
    
    COLORS = {
        logging.DEBUG: LogColors.GRAY,
        logging.INFO: LogColors.BLUE,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.RED,
    }
    
    def format(self, record):
        """Format log record with colors."""
        # Add color for log level
        if sys.stderr.isatty():
            levelname = record.levelname
            color = self.COLORS.get(record.levelno, LogColors.RESET)
            record.levelname = f"{color}{levelname}{LogColors.RESET}"
        
        return super().format(record)


def setup_logging(
    name: str = __name__,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    verbose: bool = False,
    quiet: bool = False
) -> logging.Logger:
    """
    Set up logging configuration for a script.
    
    Args:
        name: Logger name (typically __name__)
        level: Default logging level
        log_file: Optional file path for log output
        verbose: If True, set level to DEBUG
        quiet: If True, set level to WARNING
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logging(__name__, verbose=True)
        >>> logger.info("This is an info message")
        >>> logger.debug("This debug message will show (verbose=True)")
    """
    # Determine log level
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    
    console_format = ColoredFormatter(
        '%(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_log_file_path(script_name: str, log_dir: Optional[Path] = None) -> Path:
    """
    Generate a log file path for a script.
    
    Args:
        script_name: Name of the script
        log_dir: Optional directory for logs (defaults to ./logs)
        
    Returns:
        Path object for log file
        
    Example:
        >>> path = get_log_file_path("test_script")
        >>> print(path)
        logs/test_script_20260107_112345.log
    """
    if log_dir is None:
        log_dir = Path.cwd() / 'logs'
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{script_name}_{timestamp}.log"
    
    return log_dir / filename


# Example usage patterns
if __name__ == '__main__':
    # Basic usage
    logger = setup_logging(__name__, verbose=True)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # With file output
    log_file = get_log_file_path("example_script")
    logger_with_file = setup_logging(__name__, log_file=log_file, verbose=True)
    logger_with_file.info(f"Logging to {log_file}")
