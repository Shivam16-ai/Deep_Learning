"""
Module: frame_extraction.py
Task 5: Video frame extraction.

Extracts individual image frames from traffic video files at specified intervals
and writes them to the designated extracted frames directory on Windows.
"""

from pathlib import Path
from typing import Optional, List
import cv2

from utils.logger import setup_logger
from utils.config import get_path, load_config

logger = setup_logger("Task5_FrameExtraction")


def extract_frames_from_video(
    video_path: Path | str,
    output_dir: Optional[Path | str] = None,
    frame_interval: int = 10,
    max_frames: Optional[int] = None,
) -> List[Path]:
    """
    Extracts frames from a traffic video at regular intervals and saves them.
    
    Args:
        video_path (Path | str): Path to input video file.
        output_dir (Optional[Path | str]): Directory to store extracted frames.
            Defaults to 'output/extracted_frames'.
        frame_interval (int): Step interval (extracts every Nth frame). Default 10.
        max_frames (Optional[int]): Upper limit of frames to extract. None extracts all matching.
        
    Returns:
        List[Path]: List of Paths to the successfully saved frame images.
    """
    src_path = Path(video_path).resolve()
    if not src_path.exists():
        logger.error("Source video for extraction does not exist: %s", src_path)
        return []
        
    # Resolve output directory
    if output_dir is None:
        try:
            cfg = load_config()
            target_dir = get_path("extracted_frames_dir", cfg)
        except Exception:
            target_dir = Path("output/extracted_frames").resolve()
    else:
        target_dir = Path(output_dir).resolve()
        
    target_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        logger.error("Could not open video file: %s", src_path)
        return []
        
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(
        "Starting frame extraction from '%s' (Total frames: %d, FPS: %.1f, Step: %d)",
        src_path.name, total_video_frames, fps, frame_interval
    )
    
    extracted_paths: List[Path] = []
    frame_idx = 0
    saved_count = 0
    video_stem = src_path.stem
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            if frame_idx % frame_interval == 0:
                frame_filename = f"{video_stem}_frame_{saved_count:05d}.jpg"
                save_path = target_dir / frame_filename
                
                success = cv2.imwrite(str(save_path), frame)
                if success:
                    extracted_paths.append(save_path)
                    saved_count += 1
                else:
                    logger.warning("Failed to write frame: %s", save_path)
                    
                if max_frames and saved_count >= max_frames:
                    logger.info("Reached maximum extraction limit of %d frames.", max_frames)
                    break
                    
            frame_idx += 1
            
        logger.info("Successfully extracted %d frames to %s", len(extracted_paths), target_dir)
        return extracted_paths
    except Exception as exc:
        logger.exception("Error during frame extraction: %s", exc)
        return extracted_paths
    finally:
        cap.release()
