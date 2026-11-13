"""
Module: image_capture.py
Task 2: Traffic image capture.

Captures frames from a local camera/webcam or synthetic traffic source,
validates the frame acquisition, and saves images to the captures directory on Windows.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from utils.logger import setup_logger
from utils.config import get_path, load_config
from utils.sample_data import create_synthetic_traffic_frame

logger = setup_logger("Task2_ImageCapture")


def capture_traffic_image(
    device_index: int = 0,
    output_dir: Optional[Path | str] = None,
    filename_prefix: str = "traffic_capture",
    simulate_if_no_camera: bool = True,
    resolution: Tuple[int, int] = (1280, 720),
    skip_hardware_probe: bool = False,
) -> Optional[Path]:
    """
    Captures a traffic image frame from the specified camera device and saves it.
    If no camera is detected and `simulate_if_no_camera` is True, generates a synthetic
    traffic frame so automated testing and offline development remain robust.
    
    Args:
        device_index (int): Camera device index (typically 0 on Windows).
        output_dir (Optional[Path | str]): Directory where the capture will be saved.
            Defaults to the configured 'captures' directory.
        filename_prefix (str): Prefix for the saved image filename.
        simulate_if_no_camera (bool): Whether to simulate a capture if hardware is unavailable.
        resolution (Tuple[int, int]): Desired capture (width, height).
        skip_hardware_probe (bool): If True, bypasses hardware probe (useful in automated tests).
        
    Returns:
        Optional[Path]: Absolute Path to the saved image file, or None if capture failed.
    """
    logger.info("Initiating traffic image capture (Device index: %d)...", device_index)
    
    # Resolve output directory using pathlib
    if output_dir is None:
        try:
            cfg = load_config()
            target_dir = get_path("captures_dir", cfg)
        except Exception:
            target_dir = Path("captures").resolve()
    else:
        target_dir = Path(output_dir).resolve()
        
    target_dir.mkdir(parents=True, exist_ok=True)
    
    frame: Optional[np.ndarray] = None
    
    if not skip_hardware_probe:
        try:
            # Try capturing from hardware device with Windows DirectShow backend
            cap = cv2.VideoCapture(device_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                ret, read_frame = cap.read()
                cap.release()
                if ret and read_frame is not None:
                    frame = read_frame
                    logger.info("Acquired live frame from physical camera device %d.", device_index)
            else:
                cap.release()
        except Exception as cam_exc:
            logger.warning("Hardware camera probe encountered non-fatal error: %s", cam_exc)
            
    if frame is None:
        if simulate_if_no_camera:
            logger.warning(
                "Physical camera index %d not accessible. Using simulated traffic sensor frame.",
                device_index,
            )
            frame = create_synthetic_traffic_frame(frame_idx=0, width=resolution[0], height=resolution[1])
        else:
            logger.error("Unable to capture frame from camera device index: %d", device_index)
            return None
            
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.jpg"
    save_path = target_dir / filename
    
    try:
        success = cv2.imwrite(str(save_path), frame)
        if success:
            logger.info("Successfully saved captured image to: %s", save_path)
            return save_path
        else:
            logger.error("Failed to write image file at: %s", save_path)
            return None
    except Exception as exc:
        logger.exception("Unexpected error during image save: %s", exc)
        return None
