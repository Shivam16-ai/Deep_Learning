"""
Module: image_display.py
Task 3: Image reading and displaying using OpenCV.

Provides robust functions to load, inspect, resize, and display traffic images
with comprehensive error handling and window lifecycle management on Windows.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from utils.logger import setup_logger

logger = setup_logger("Task3_ImageDisplay")


def inspect_image_metadata(img: np.ndarray, source_name: str = "frame") -> Dict[str, Any]:
    """
    Extracts geometric and photometric properties from an image.
    
    Args:
        img (np.ndarray): Image array.
        source_name (str): Label or filename for identification.
        
    Returns:
        Dict[str, Any]: Image properties dictionary.
    """
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    aspect_ratio = round(width / height, 3) if height > 0 else 0.0
    
    metadata = {
        "source": source_name,
        "width": width,
        "height": height,
        "channels": channels,
        "aspect_ratio": aspect_ratio,
        "dtype": str(img.dtype),
        "mean_intensity": round(float(np.mean(img)), 2),
        "min_intensity": int(np.min(img)),
        "max_intensity": int(np.max(img)),
    }
    return metadata


def load_and_inspect_image(
    image_path: Path | str,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Reads an image from disk using OpenCV and logs its dimensional properties.
    
    Args:
        image_path (Path | str): Path to the image file.
        
    Returns:
        Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
            Loaded image in BGR format and its metadata dictionary, or (None, None).
    """
    path = Path(image_path).resolve()
    if not path.exists():
        logger.error("Image file does not exist at: %s", path)
        return None, None
        
    img = cv2.imread(str(path))
    if img is None:
        logger.error("Failed to decode image from path: %s", path)
        return None, None
        
    meta = inspect_image_metadata(img, source_name=path.name)
    logger.info(
        "Image loaded from %s -> %dx%d | Channels: %d | Mean intensity: %.1f",
        meta["source"], meta["width"], meta["height"], meta["channels"], meta["mean_intensity"]
    )
    return img, meta


def convert_color_space(img: np.ndarray, target: str = "RGB") -> np.ndarray:
    """
    Converts image between standard computer vision color spaces.
    
    Args:
        img (np.ndarray): Input BGR image.
        target (str): Target format: 'RGB', 'GRAY', or 'HSV'.
        
    Returns:
        np.ndarray: Converted image.
    """
    target = target.upper()
    if target == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif target in ("GRAY", "GRAYSCALE"):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif target == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        logger.warning("Unsupported color space '%s'. Returning original.", target)
        return img


def display_image(
    image_or_path: np.ndarray | Path | str,
    window_title: str = "VisionPort - Traffic Image Display",
    window_size: Optional[Tuple[int, int]] = None,
    wait_time_ms: int = 0,
    headless: bool = False,
    save_copy_to: Optional[Path | str] = None,
) -> bool:
    """
    Displays an image in an OpenCV GUI window on Windows, with headless fallback support.
    
    Args:
        image_or_path (np.ndarray | Path | str): Numpy array or path to the image.
        window_title (str): Title of the display window.
        window_size (Optional[Tuple[int, int]]): Optional (width, height) to resize window.
        wait_time_ms (int): Milliseconds to wait for key press (0 = indefinitely).
        headless (bool): If True, suppresses GUI display and validates/saves without opening a window.
        save_copy_to (Optional[Path | str]): Optional destination to write a verification copy.
        
    Returns:
        bool: True if display/processing was successful, False otherwise.
    """
    if isinstance(image_or_path, (str, Path)):
        img, _ = load_and_inspect_image(image_or_path)
    elif isinstance(image_or_path, np.ndarray):
        img = image_or_path
    else:
        logger.error("Invalid input type for display_image: %s", type(image_or_path))
        return False
        
    if img is None:
        return False
        
    if save_copy_to:
        out_path = Path(save_copy_to).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        logger.info("Saved inspection image copy to: %s", out_path)
        
    if headless:
        logger.info("Headless display mode active. Image processed successfully: %s", window_title)
        return True
        
    try:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        if window_size:
            cv2.resizeWindow(window_title, window_size[0], window_size[1])
            
        cv2.imshow(window_title, img)
        logger.info("Displaying window '%s'. Press any key to close (or wait %d ms).", window_title, wait_time_ms)
        cv2.waitKey(wait_time_ms)
        cv2.destroyWindow(window_title)
        return True
    except Exception as exc:
        logger.exception("Error occurred while displaying image: %s", exc)
        return False
