"""
Module: sample_data.py
Utility to generate synthetic traffic images and video sequences for testing
Tasks 1 through 5 locally without downloading external datasets.
"""

from pathlib import Path
from typing import Tuple
import cv2
import numpy as np

from utils.logger import setup_logger
from utils.config import get_path, load_config

logger = setup_logger("SampleDataGenerator")


def create_synthetic_traffic_frame(frame_idx: int = 0, width: int = 1280, height: int = 720) -> np.ndarray:
    """
    Renders a synthetic traffic scene with asphalt, lane dividers, traffic lights, and vehicles.
    
    Args:
        frame_idx (int): Current frame index (used to animate vehicle positions).
        width (int): Frame width in pixels.
        height (int): Frame height in pixels.
        
    Returns:
        np.ndarray: Generated image frame in BGR format.
    """
    # Background: Dark gray asphalt
    frame = np.full((height, width, 3), (45, 45, 45), dtype=np.uint8)
    
    # Sidewalks / Road shoulders
    frame[0:120, :] = (90, 95, 90)       # Top shoulder
    frame[height - 80:height, :] = (90, 95, 90) # Bottom shoulder
    
    # Solid white boundary lines
    cv2.line(frame, (0, 120), (width, 120), (240, 240, 240), 4)
    cv2.line(frame, (0, height - 80), (width, height - 80), (240, 240, 240), 4)
    
    # Lane dividers (Dashed yellow and white lines)
    lane_height = (height - 200) // 3
    for lane_i in range(1, 3):
        y_lane = 120 + lane_i * lane_height
        for x in range(0, width, 60):
            cv2.line(frame, (x, y_lane), (x + 35, y_lane), (0, 220, 220), 2)
            
    # Traffic signal fixture (Top-right corner)
    cv2.rectangle(frame, (width - 120, 20), (width - 40, 100), (20, 20, 20), -1)
    cv2.circle(frame, (width - 80, 40), 12, (0, 0, 255), -1)   # Red
    cv2.circle(frame, (width - 80, 60), 12, (50, 50, 50), -1)  # Yellow (off)
    cv2.circle(frame, (width - 80, 80), 12, (50, 50, 50), -1)  # Green (off)
    
    # Vehicles (Animated across lanes based on frame_idx)
    # Vehicle 1: Blue Car in Lane 1 (moving right)
    v1_x = int((frame_idx * 12 + 100) % (width + 200)) - 100
    v1_y = 120 + (lane_height // 2) - 30
    cv2.rectangle(frame, (v1_x, v1_y), (v1_x + 130, v1_y + 60), (220, 80, 30), -1)  # Body
    cv2.rectangle(frame, (v1_x + 25, v1_y + 10), (v1_x + 95, v1_y + 50), (180, 220, 240), -1) # Roof/Glass
    
    # Vehicle 2: Green Truck in Lane 2 (moving right, slower)
    v2_x = int((frame_idx * 7 + 450) % (width + 300)) - 150
    v2_y = 120 + lane_height + (lane_height // 2) - 40
    cv2.rectangle(frame, (v2_x, v2_y), (v2_x + 210, v2_y + 80), (40, 160, 40), -1) # Cabin + Cargo
    cv2.rectangle(frame, (v2_x + 140, v2_y + 5), (v2_x + 200, v2_y + 75), (20, 100, 20), -1)
    
    # Vehicle 3: Orange Motorcycle in Lane 3 (moving right, faster)
    v3_x = int((frame_idx * 18 + 50) % (width + 100)) - 50
    v3_y = 120 + 2 * lane_height + (lane_height // 2) - 15
    cv2.rectangle(frame, (v3_x, v3_y), (v3_x + 70, v3_y + 30), (0, 140, 255), -1)
    
    # Frame information text overlay
    cv2.putText(
        frame,
        f"VisionPort Synthetic Traffic Feed | Frame: {frame_idx:04d}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    
    return frame


def generate_sample_assets(force: bool = False) -> Tuple[Path, Path]:
    """
    Generates a sample traffic image and video in dataset/images and dataset/videos.
    
    Args:
        force (bool): Whether to overwrite if files already exist.
        
    Returns:
        Tuple[Path, Path]: Paths to (sample_image_path, sample_video_path).
    """
    cfg = load_config()
    images_dir = get_path("dataset_images_dir", cfg)
    videos_dir = get_path("dataset_videos_dir", cfg)
    
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    sample_img_path = images_dir / "sample_traffic.jpg"
    sample_vid_path = videos_dir / "sample_traffic.mp4"
    
    # 1. Generate Image
    if not sample_img_path.exists() or force:
        frame = create_synthetic_traffic_frame(frame_idx=15)
        cv2.imwrite(str(sample_img_path), frame)
        logger.info("Generated sample traffic image: %s", sample_img_path)
    else:
        logger.info("Sample image already exists: %s", sample_img_path)
        
    # 2. Generate Video (60 frames at 30 FPS = 2 seconds)
    if not sample_vid_path.exists() or force:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(sample_vid_path), fourcc, 30.0, (1280, 720))
        for i in range(60):
            f = create_synthetic_traffic_frame(frame_idx=i)
            writer.write(f)
        writer.release()
        logger.info("Generated sample traffic video (60 frames): %s", sample_vid_path)
    else:
        logger.info("Sample video already exists: %s", sample_vid_path)
        
    return sample_img_path, sample_vid_path
