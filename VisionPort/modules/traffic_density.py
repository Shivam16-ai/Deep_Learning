"""
Module: traffic_density.py
Task 10: Traffic density calculation.

Calculates instantaneous and zone-based traffic density indices, classifies roadway conditions
into LOW, MEDIUM, and HIGH density tiers, tracks density transition events with timestamps,
and renders real-time HUD density badges onto video streams.

================================================================================
LIMITATIONS OF COUNT-BASED TRAFFIC DENSITY ESTIMATION (DOCUMENTED ARCHITECTURE):
================================================================================
1. Vehicle Heterogeneity & Spatial Footprint:
   A raw count treats all vehicles equally, whereas a standard bus or freight truck
   occupies 3x to 5x the road footprint of a compact sedan and over 8x that of a motorcycle.
   Relying exclusively on count without occupancy ratio or passenger car equivalent (PCE)
   distorts true road saturation.

2. Monocular Perspective Distortion & Non-Uniform Scaling:
   In uncalibrated monocular camera perspectives, the road geometry recedes toward the horizon.
   Ten vehicles detected in the background represent a vastly denser physical cluster
   than ten vehicles spread across the foreground.

3. Flow Velocity Decoupling (k = q / v):
   According to macroscopic traffic flow theory, density (k) is related to flow (q) and space-mean
   speed (v). Ten vehicles travelling at 70 km/h represent a healthy free-flow state, whereas
   the exact same count of ten vehicles stopped bumper-to-bumper represents severe gridlock.

4. Inter-Vehicle Occlusion:
   At steep camera angles, larger commercial vehicles frequently occlude motorcycles and small
   cars in adjacent lanes, causing systematic count depression precisely when traffic is densest.
================================================================================
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import csv
import json
import cv2
import numpy as np

from utils.logger import setup_logger
from utils.config import load_config, get_path

logger = setup_logger("TrafficDensity")


class TrafficDensityCalculator:
    """
    Computes traffic density, classifies roadway volume tiers,
    tracks temporal state transitions, and renders real-time density indicators.
    """

    # Standard color palette for density tiers (BGR)
    DENSITY_COLORS: Dict[str, Tuple[int, int, int]] = {
        "LOW": (0, 200, 0),        # Forest Green
        "MEDIUM": (0, 165, 255),    # Amber / Orange
        "HIGH": (0, 0, 230),       # Bright Red
    }

    def __init__(
        self,
        low_threshold: Optional[int] = None,
        medium_threshold: Optional[int] = None,
        high_threshold: Optional[int] = None,
        roi_polygon: Optional[List[Tuple[int, int]]] = None,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
        road_area_sq_meters: float = 1000.0,
    ):
        """
        Initializes the TrafficDensityCalculator.
        
        Args:
            low_threshold: Vehicle count threshold below which density is 'LOW'.
            medium_threshold: Vehicle count threshold below which density is 'MEDIUM'.
            high_threshold: Vehicle count threshold above which density is firmly 'HIGH'.
            roi_polygon: Optional polygonal road boundary [(x, y), ...].
            roi_bbox: Optional rectangular road boundary (x1, y1, x2, y2).
            road_area_sq_meters: Monitored roadway surface area in square meters.
        """
        # Load from config if not explicitly provided
        cfg_d: Dict[str, Any] = {}
        try:
            cfg = load_config()
            cfg_d = cfg.get("traffic_analysis", {}).get("density", {})
        except Exception:
            pass

        self.low_threshold = int(low_threshold if low_threshold is not None else cfg_d.get("low_threshold", 5))
        self.medium_threshold = int(medium_threshold if medium_threshold is not None else cfg_d.get("medium_threshold", 15))
        self.high_threshold = int(high_threshold if high_threshold is not None else cfg_d.get("high_threshold", 25))

        self.roi_polygon = np.array(roi_polygon, dtype=np.int32) if roi_polygon else None
        self.roi_bbox = roi_bbox
        self.road_area_sq_meters = road_area_sq_meters

        self.current_level: str = "LOW"
        self.previous_level: str = "LOW"

        # Event log recording every state change (e.g. LOW -> MEDIUM)
        self.transitions: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []

        logger.info(
            "TrafficDensityCalculator initialized | Tiers: LOW <= %d | MEDIUM <= %d | HIGH > %d",
            self.low_threshold, self.medium_threshold, self.medium_threshold
        )

    def is_in_road_roi(self, bbox: List[float]) -> bool:
        """Determines if a vehicle centroid lies inside the monitored road ROI."""
        if self.roi_polygon is None and self.roi_bbox is None:
            return True

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        if self.roi_bbox is not None:
            rx1, ry1, rx2, ry2 = self.roi_bbox
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                return False

        if self.roi_polygon is not None:
            inside = cv2.pointPolygonTest(self.roi_polygon, (float(cx), float(cy)), measureDist=False)
            if inside < 0:
                return False

        return True

    def calculate_density(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: Optional[Tuple[int, int]] = None,
        frame_idx: int = 0,
        timestamp_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculates instantaneous traffic density from detected vehicles.
        
        Args:
            detections: List of detection dictionaries from VehicleDetector.
            frame_shape: (height, width) of the frame.
            frame_idx: Sequential frame index.
            timestamp_sec: Video timestamp in seconds.
            
        Returns:
            Dict[str, Any]: Density metrics including level, count, occupancy ratio, and density index.
        """
        # Filter detections in road ROI
        road_vehicles: List[Dict[str, Any]] = []
        total_vehicle_area_px = 0.0

        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) == 4 and self.is_in_road_roi(bbox):
                road_vehicles.append(det)
                w = max(0.0, bbox[2] - bbox[0])
                h = max(0.0, bbox[3] - bbox[1])
                total_vehicle_area_px += (w * h)

        count = len(road_vehicles)

        # Classify volume tier
        if count <= self.low_threshold:
            level = "LOW"
        elif count <= self.medium_threshold:
            level = "MEDIUM"
        else:
            level = "HIGH"

        # Calculate estimated occupancy ratio
        if self.roi_bbox is not None:
            roi_area_px = max(1.0, (self.roi_bbox[2] - self.roi_bbox[0]) * (self.roi_bbox[3] - self.roi_bbox[1]))
        elif self.roi_polygon is not None:
            roi_area_px = max(1.0, float(cv2.contourArea(self.roi_polygon)))
        elif frame_shape is not None:
            roi_area_px = max(1.0, float(frame_shape[0] * frame_shape[1]))
        else:
            roi_area_px = 1280.0 * 720.0

        occupancy_ratio = min(1.0, round(total_vehicle_area_px / roi_area_px, 4))
        occupancy_percent = round(occupancy_ratio * 100.0, 2)

        # Vehicles per square kilometer approximation
        density_per_sq_km = round((count / self.road_area_sq_meters) * 1_000_000.0, 2) if self.road_area_sq_meters > 0 else 0.0

        # Normalized density score (0.0 to 1.0)
        density_score = min(1.0, round(count / float(self.high_threshold), 3))

        ts = round(float(timestamp_sec), 3) if timestamp_sec is not None else round(frame_idx / 30.0, 3)

        # Check for state transition
        if level != self.current_level:
            transition_event = {
                "frame_index": int(frame_idx),
                "timestamp_sec": ts,
                "from_level": self.current_level,
                "to_level": level,
                "vehicle_count": count,
                "occupancy_percent": occupancy_percent,
                "description": f"Traffic transitioned from {self.current_level} to {level}",
            }
            self.transitions.append(transition_event)
            logger.info("Density state change: %s at frame %d (t=%.2fs, count=%d)", transition_event["description"], frame_idx, ts, count)
            self.previous_level = self.current_level
            self.current_level = level

        record = {
            "frame_index": int(frame_idx),
            "timestamp_sec": ts,
            "vehicle_count": count,
            "density_level": level,
            "density_score": density_score,
            "occupancy_percent": occupancy_percent,
            "density_per_sq_km": density_per_sq_km,
        }
        self.history.append(record)
        return record

    def annotate_frame(
        self,
        frame: np.ndarray,
        density_metrics: Dict[str, Any],
        show_roi: bool = True,
    ) -> np.ndarray:
        """
        Draws dynamic traffic density badge, occupancy bar, and ROI onto the frame.
        
        Args:
            frame: Input image array in BGR format.
            density_metrics: Output from calculate_density().
            show_roi: Whether to render road ROI outline.
            
        Returns:
            np.ndarray: Annotated frame with density indicator.
        """
        annotated = frame.copy()
        level = density_metrics.get("density_level", "LOW")
        count = density_metrics.get("vehicle_count", 0)
        occ = density_metrics.get("occupancy_percent", 0.0)
        color = self.DENSITY_COLORS.get(level, (0, 255, 0))

        # Draw road ROI outline
        if show_roi:
            if self.roi_polygon is not None:
                cv2.polylines(annotated, [self.roi_polygon], isClosed=True, color=(255, 200, 0), thickness=2)
            elif self.roi_bbox is not None:
                rx1, ry1, rx2, ry2 = self.roi_bbox
                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)

        # Density Status Badge in upper-right area
        badge_w = 260
        badge_h = 75
        x2 = frame.shape[1] - 15
        x1 = x2 - badge_w
        y1 = 10
        y2 = y1 + badge_h

        # Semi-transparent dark background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Status Pill Bar
        cv2.rectangle(annotated, (x1, y1), (x1 + 8, y2), color, -1)

        # Text Header
        cv2.putText(
            annotated,
            f"DENSITY: {level}",
            (x1 + 18, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

        # Subtext details
        cv2.putText(
            annotated,
            f"Monitored: {count} vehicles | Occ: {occ}%",
            (x1 + 18, y1 + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        return annotated

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns statistical summary of density performance and transitions.
        
        Returns:
            Dict[str, Any]: Density summary.
        """
        if not self.history:
            return {"total_frames": 0, "distribution": {}, "transitions_count": 0}

        counts_by_level = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for h in self.history:
            lvl = h["density_level"]
            counts_by_level[lvl] = counts_by_level.get(lvl, 0) + 1

        total = len(self.history)
        pct_by_level = {k: round((v / total) * 100.0, 2) for k, v in counts_by_level.items()}

        return {
            "total_frames_analyzed": total,
            "level_distribution_frames": counts_by_level,
            "level_distribution_percent": pct_by_level,
            "total_state_transitions": len(self.transitions),
            "state_transitions": self.transitions,
            "thresholds": {
                "low": self.low_threshold,
                "medium": self.medium_threshold,
                "high": self.high_threshold,
            },
        }

    @staticmethod
    def get_limitations() -> List[Dict[str, str]]:
        """Returns documented engineering limitations of count-based density estimation."""
        return [
            {
                "limitation": "Vehicle Heterogeneity",
                "impact": "Buses and trucks occupy significantly more road space than cars or motorcycles.",
            },
            {
                "limitation": "Perspective Distortion",
                "impact": "Monocular 2D perspectives compress background vehicles, causing spatial distortion.",
            },
            {
                "limitation": "Flow Decoupling",
                "impact": "Count does not capture vehicle velocity (moving traffic vs. stationary gridlock).",
            },
            {
                "limitation": "Occlusion in Dense Queues",
                "impact": "Large vehicles occlude smaller vehicles in adjacent lanes at oblique angles.",
            },
        ]

    def export_json(self, output_path: Union[Path, str]) -> Path:
        """Exports density time-series, transitions, and limitations to JSON."""
        dest = Path(output_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "summary": self.get_summary(),
            "limitations": self.get_limitations(),
            "transitions": self.transitions,
            "frame_history": self.history,
        }

        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Exported traffic density JSON to: %s", dest)
        return dest

    def export_csv(self, output_path: Union[Path, str]) -> Path:
        """Exports density time-series to CSV."""
        dest = Path(output_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "frame_index",
            "timestamp_sec",
            "vehicle_count",
            "density_level",
            "density_score",
            "occupancy_percent",
            "density_per_sq_km",
        ]

        with open(dest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for h in self.history:
                writer.writerow(h)

        logger.info("Exported traffic density CSV to: %s", dest)
        return dest

    def reset(self) -> None:
        """Resets calculator state."""
        self.current_level = "LOW"
        self.previous_level = "LOW"
        self.transitions.clear()
        self.history.clear()
