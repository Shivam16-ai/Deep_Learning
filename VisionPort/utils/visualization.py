"""
Visualization Utilities for VisionPort.
Provides helper functions to overlay bounding boxes, labels, tracking IDs,
and traffic analytics metrics onto OpenCV image frames.
"""

from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


# Standard BGR color palette for vehicle classes
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car": (0, 255, 0),         # Green
    "motorcycle": (255, 165, 0), # Orange
    "bus": (255, 0, 0),          # Blue
    "truck": (0, 255, 255),      # Yellow
    "default": (0, 200, 200),
}


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    show_labels: bool = True,
    show_conf: bool = True,
) -> np.ndarray:
    """
    Draws vehicle bounding boxes, class labels, and confidence scores on the frame.
    (Supports Task 8: Vehicle labels and confidence scores).
    
    Args:
        frame (np.ndarray): Input image/frame in BGR format.
        detections (List[Dict[str, Any]]): List of detection dictionaries containing:
            'bbox': [x1, y1, x2, y2],
            'class_name': str,
            'confidence': float.
        show_labels (bool): Whether to render the class label text.
        show_conf (bool): Whether to display the confidence score.
        
    Returns:
        np.ndarray: Annotated frame with drawn bounding boxes and text.
    """
    annotated = frame.copy()
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cls_name = det.get("class_name", "vehicle")
        conf = det.get("confidence", 0.0)
        
        color = CLASS_COLORS.get(cls_name.lower(), CLASS_COLORS["default"])
        
        # Draw bounding rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label string
        if show_labels and show_conf:
            label_text = f"{cls_name} {conf:.2f}"
        elif show_labels:
            label_text = f"{cls_name}"
        elif show_conf:
            label_text = f"{conf:.2f}"
        else:
            label_text = ""
            
        if label_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Label background box
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - text_h - baseline - 4)),
                (x1 + text_w + 4, max(text_h + baseline + 4, y1)),
                color,
                -1,
            )
            # Label text in black or white
            cv2.putText(
                annotated,
                label_text,
                (x1 + 2, max(text_h + 2, y1 - 4)),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_AA,
            )
            
    return annotated


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Draws tracking bounding boxes, track IDs, and past trajectory paths.
    (Supports Task 12: Individual vehicle tracking).
    
    Args:
        frame (np.ndarray): Input frame.
        tracks (List[Dict[str, Any]]): List of tracked objects containing:
            'track_id': int,
            'bbox': [x1, y1, x2, y2],
            'history': List[Tuple[int, int]] (optional).
            
    Returns:
        np.ndarray: Annotated frame with tracking markers.
    """
    annotated = frame.copy()
    for track in tracks:
        track_id = track.get("track_id")
        bbox = track.get("bbox", [])
        if len(bbox) != 4 or track_id is None:
            continue
            
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = (255, 0, 255)  # Magenta for tracking
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"ID: {track_id}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
        
        # Draw motion trail if history exists
        history = track.get("history", [])
        for i in range(1, len(history)):
            pt1 = tuple(int(c) for c in history[i - 1])
            pt2 = tuple(int(c) for c in history[i])
            cv2.line(annotated, pt1, pt2, color, 2)
            
    return annotated


def overlay_traffic_metrics(
    frame: np.ndarray,
    metrics: Dict[str, Any],
    origin: Tuple[int, int] = (20, 30),
) -> np.ndarray:
    """
    Overlays traffic analytical metrics (counts, density, speed, status) on the top corner.
    
    Args:
        frame (np.ndarray): Input image frame.
        metrics (Dict[str, Any]): Dictionary of key-value metrics to render.
        origin (Tuple[int, int]): Starting (x, y) coordinates for dashboard overlay.
        
    Returns:
        np.ndarray: Frame with transparent dashboard overlay.
    """
    annotated = frame.copy()
    x0, y0 = origin
    line_height = 24
    
    # Semi-transparent background banner
    num_lines = len(metrics)
    if num_lines > 0:
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (x0 - 10, y0 - 20),
            (x0 + 320, y0 + (num_lines * line_height) + 10),
            (30, 30, 30),
            -1,
        )
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
    curr_y = y0
    for key, value in metrics.items():
        text = f"{key}: {value}"
        cv2.putText(
            annotated,
            text,
            (x0, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
        curr_y += line_height
        
    return annotated
