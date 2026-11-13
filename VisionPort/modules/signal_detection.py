"""
Module: signal_detection.py
Task 14: Traffic signal status detection.

Identifies traffic signal light states (RED, YELLOW, GREEN) from designated
traffic light Regions of Interest (ROI) using computer vision and color segmentation.
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from utils.logger import setup_logger

logger = setup_logger("Task14_SignalDetection")


class TrafficSignalDetector:
    """
    Detects the active operational state of a traffic light.
    """

    # Default HSV threshold bounds for traffic light signals
    COLOR_RANGES = {
        "RED": [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([180, 255, 255])),
        ],
        "YELLOW": [
            (np.array([15, 100, 100]), np.array([35, 255, 255])),
        ],
        "GREEN": [
            (np.array([40, 100, 100]), np.array([85, 255, 255])),
        ],
    }

    def __init__(self, signal_roi: Optional[Tuple[int, int, int, int]] = None):
        """
        Initializes the signal detector.
        
        Args:
            signal_roi (Optional[Tuple[int, int, int, int]]):
                Bounding box of traffic light fixture in frame: (x, y, w, h).
        """
        self.signal_roi = signal_roi
        self.current_state: str = "UNKNOWN"
        logger.info("TrafficSignalDetector initialized | ROI: %s", signal_roi)

    def detect_signal_state(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes the traffic signal ROI and classifies current light state.
        
        Args:
            frame (np.ndarray): Video frame in BGR format.
            
        Returns:
            Dict[str, Any]: Signal state assessment containing:
                - 'state': str ('RED', 'YELLOW', 'GREEN', 'UNKNOWN')
                - 'confidence': float
                - 'roi': Optional[Tuple[int, int, int, int]]
        """
        if self.signal_roi is None:
            logger.debug("Signal ROI not specified. Signal state is UNKNOWN.")
            return {"state": "UNKNOWN", "confidence": 0.0, "roi": None}

        x, y, w, h = self.signal_roi
        h_frame, w_frame = frame.shape[:2]
        
        # Clip coordinates to frame bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_frame, x + w), min(h_frame, y + h)

        if x2 <= x1 or y2 <= y1:
            return {"state": "UNKNOWN", "confidence": 0.0, "roi": self.signal_roi}

        roi_img = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        color_scores: Dict[str, int] = {}

        for color, ranges in self.COLOR_RANGES.items():
            total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                total_mask = cv2.bitwise_or(total_mask, mask)
            color_scores[color] = cv2.countNonZero(total_mask)

        detected_state = "UNKNOWN"
        max_score = 0
        for color, score in color_scores.items():
            if score > max_score and score > 20:  # Noise threshold
                max_score = score
                detected_state = color

        self.current_state = detected_state
        logger.debug("Signal State detected: %s (Score: %d)", detected_state, max_score)
        
        return {
            "state": detected_state,
            "confidence": round(float(max_score) / (w * h + 1e-5), 2),
            "roi": self.signal_roi,
        }
