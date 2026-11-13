"""
Configuration Loader Module for VisionPort.
Handles loading and validating the YAML configuration file using pathlib for Windows compatibility.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def get_project_root() -> Path:
    """
    Returns the absolute Path of the VisionPort project root directory.
    
    Returns:
        Path: Absolute path to the project root.
    """
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """
    Loads and parses the project YAML configuration file.
    
    Args:
        config_path (Optional[Path | str]): Custom path to config.yaml.
            If None, defaults to config.yaml in the project root.
            
    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
        
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file is empty or corrupted.
    """
    root_dir = get_project_root()
    
    if config_path is None:
        target_path = root_dir / "config.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = root_dir / target_path
            
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")
        
    try:
        with open(target_path, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file)
            
        if not isinstance(config_data, dict):
            raise ValueError(f"Invalid format in configuration file: {target_path}")
            
        return config_data
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML configuration: {exc}") from exc


def get_path(key_or_relative_path: str, config: Optional[Dict[str, Any]] = None) -> Path:
    """
    Resolves a configured path key or relative path string into an absolute Path on Windows.
    
    Args:
        key_or_relative_path (str): Key inside `config['paths']` or a relative path string.
        config (Optional[Dict[str, Any]]): Preloaded config dictionary.
        
    Returns:
        Path: Resolved absolute Path.
    """
    root = get_project_root()
    if config is not None and "paths" in config:
        rel_path = config["paths"].get(key_or_relative_path, key_or_relative_path)
    else:
        rel_path = key_or_relative_path
        
    resolved_path = (root / Path(rel_path)).resolve()
    return resolved_path
