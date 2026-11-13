"""
Logger Utility Module for VisionPort.
Sets up standardized console and file logging across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from .config import get_project_root


def setup_logger(
    name: str = "VisionPort",
    log_file: Optional[str] = "visionport.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configures and returns a standardized logger instance.
    
    Args:
        name (str): Name of the logger, typically __name__ of calling module.
        log_file (Optional[str]): Log filename. If provided, log messages will be saved
            to this file inside the project root directory.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger was already created
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        fmt="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optional File Handler
    if log_file:
        root_dir = get_project_root()
        log_path = root_dir / log_file
        try:
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as err:
            logger.warning("Could not initialize file handler at %s: %s", log_path, err)
            
    return logger
