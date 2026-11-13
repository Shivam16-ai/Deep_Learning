"""
Module: video_processing.py
Task 4: Traffic video playback.

Handles video stream loading, interactive playback controls, video stream verification,
and frame-by-frame iteration on Windows.
"""

from pathlib import Path
from typing import Generator, Optional, Tuple
import cv2
import numpy as np

from utils.logger import setup_logger

logger = setup_logger("Task4_VideoProcessing")


class VideoProcessor:
    """
    Manages reading, metadata extraction, playback, and writing of traffic video streams.
    """
    
    def __init__(self, video_path: Path | str):
        """
        Initializes VideoProcessor with target video file path.
        
        Args:
            video_path (Path | str): Path to the video file.
        """
        self.video_path = Path(video_path).resolve()
        self.fps: float = 0.0
        self.frame_count: int = 0
        self.width: int = 0
        self.height: int = 0
        self.duration_seconds: float = 0.0
        self._load_metadata()

    def _load_metadata(self) -> bool:
        """Loads metadata properties from the video stream."""
        if not self.video_path.exists():
            logger.error("Video file does not exist: %s", self.video_path)
            return False
            
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            logger.error("Failed to open video file: %s", self.video_path)
            return False
            
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_seconds = round(self.frame_count / self.fps, 2) if self.fps > 0 else 0.0
        cap.release()
        
        logger.info(
            "Video Metadata -> File: %s | Resolution: %dx%d | FPS: %.2f | Frames: %d | Duration: %.2fs",
            self.video_path.name, self.width, self.height, self.fps, self.frame_count, self.duration_seconds
        )
        return True

    def iter_frames(self, max_frames: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yields (frame_index, frame_image) from the video stream sequentially.
        
        Args:
            max_frames (Optional[int]): Maximum number of frames to yield.
            
        Yields:
            Tuple[int, np.ndarray]: Sequential (frame_idx, frame_bgr).
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            logger.error("Could not open video stream for iteration: %s", self.video_path)
            return
            
        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                yield frame_idx, frame
                frame_idx += 1
                if max_frames and frame_idx >= max_frames:
                    break
        finally:
            cap.release()


def play_traffic_video(
    video_path: Path | str,
    window_title: str = "VisionPort - Traffic Video Playback",
    window_size: Optional[Tuple[int, int]] = None,
    max_frames: Optional[int] = None,
    headless: bool = False,
) -> bool:
    """
    Plays back a traffic video stream with interactive controls:
    - 'q' or 'ESC' to exit playback.
    - 'spacebar' to pause/resume playback.
    
    Args:
        video_path (Path | str): Path to the traffic video file.
        window_title (str): Window title for video display.
        window_size (Optional[Tuple[int, int]]): Custom window size (width, height).
        max_frames (Optional[int]): Limit frames to process/play.
        headless (bool): If True, processes frames sequentially without GUI popup (for CI/automated test).
        
    Returns:
        bool: True if video completed or was stopped cleanly, False on failure.
    """
    path = Path(video_path).resolve()
    if not path.exists():
        logger.error("Video path not found: %s", path)
        return False
        
    processor = VideoProcessor(path)
    if processor.frame_count == 0:
        logger.error("Unable to read valid frames from video: %s", path)
        return False
        
    if headless:
        processed_count = 0
        for idx, _ in processor.iter_frames(max_frames=max_frames):
            processed_count += 1
        logger.info("Headless video playback test passed. Processed %d frames.", processed_count)
        return True
        
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.error("Could not open video stream for playback: %s", path)
        return False
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = int(1000 / fps) if fps > 0 else 33
    
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    if window_size:
        cv2.resizeWindow(window_title, window_size[0], window_size[1])
        
    logger.info("Starting video playback for: %s. Controls: [SPACE] Pause | [Q/ESC] Quit", path.name)
    
    paused = False
    frame_idx = 0
    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.info("Video playback reached end of stream: %s", path.name)
                    break
                cv2.imshow(window_title, frame)
                frame_idx += 1
                if max_frames and frame_idx >= max_frames:
                    break
                
            key = cv2.waitKey(delay_ms if not paused else 100) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                logger.info("Playback terminated by user at frame %d.", frame_idx)
                break
            elif key == ord(" "):  # Spacebar toggles pause
                paused = not paused
                logger.debug("Playback %s", "paused" if paused else "resumed")
                
        return True
    except Exception as exc:
        logger.exception("Playback error occurred: %s", exc)
        return False
    finally:
        cap.release()
        cv2.destroyWindow(window_title)
