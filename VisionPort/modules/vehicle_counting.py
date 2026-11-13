"""
Module: vehicle_counting.py
Task 9: Vehicle counting.

Provides robust vehicle counting across full frames and configurable Regions of Interest (ROI).
Maintains per-frame and cumulative category-wise counts (cars, motorcycles, buses, trucks),
integrates with real YOLO detections, renders HUD overlays, and exports analytics to JSON and CSV.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import csv
import json
import cv2
import numpy as np

from utils.logger import setup_logger
from utils.config import load_config, get_path

logger = setup_logger("VehicleCounting")


class VehicleCounter:
    """
    Manages vehicle tallying, category tracking, and ROI spatial filtering
    based on actual YOLO detections.
    """

    SUPPORTED_CLASSES = ["car", "motorcycle", "bus", "truck"]

    def __init__(
        self,
        roi_polygon: Optional[List[Tuple[int, int]]] = None,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    ):
        """
        Initializes the VehicleCounter.
        
        Args:
            roi_polygon: Optional list of (x, y) coordinates defining a polygonal counting zone.
            roi_bbox: Optional tuple (x1, y1, x2, y2) defining a rectangular counting zone.
        """
        self.roi_polygon = np.array(roi_polygon, dtype=np.int32) if roi_polygon else None
        self.roi_bbox = roi_bbox

        # Cumulative category metrics
        self.cumulative_counts: Dict[str, int] = {k: 0 for k in self.SUPPORTED_CLASSES}
        self.peak_frame_count: int = 0
        self.total_frames_processed: int = 0

        # Frame-by-frame history for analytics export
        self.history: List[Dict[str, Any]] = []

        logger.info(
            "VehicleCounter initialized | Polygonal ROI: %s | BBox ROI: %s",
            self.roi_polygon is not None, self.roi_bbox is not None
        )

    def is_inside_roi(self, bbox: List[float]) -> bool:
        """
        Checks whether the centroid of a detected bounding box falls inside the active ROI.
        If no ROI is configured, returns True (monitors entire frame).
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box coordinates.
            
        Returns:
            bool: True if centroid is within ROI, False otherwise.
        """
        if self.roi_polygon is None and self.roi_bbox is None:
            return True

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        # Check rectangular ROI
        if self.roi_bbox is not None:
            rx1, ry1, rx2, ry2 = self.roi_bbox
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                return False

        # Check polygonal ROI
        if self.roi_polygon is not None:
            point = (float(cx), float(cy))
            # measureDist=False returns +1 inside, -1 outside, 0 on edge
            inside = cv2.pointPolygonTest(self.roi_polygon, point, measureDist=False)
            if inside < 0:
                return False

        return True

    def count_frame(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int = 0,
        timestamp_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Counts vehicles in the current frame from actual YOLO detection results.
        
        Args:
            detections: List of detection dictionaries from VehicleDetector.
            frame_idx: Sequential frame index.
            timestamp_sec: Video timestamp in seconds (optional).
            
        Returns:
            Dict[str, Any]: Frame counting metrics.
        """
        frame_counts: Dict[str, int] = {k: 0 for k in self.SUPPORTED_CLASSES}
        counted_detections: List[Dict[str, Any]] = []

        for det in detections:
            cls_name = det.get("class_name", "").lower()
            bbox = det.get("bbox", [])
            if len(bbox) != 4 or cls_name not in self.SUPPORTED_CLASSES:
                continue

            if self.is_inside_roi(bbox):
                frame_counts[cls_name] += 1
                counted_detections.append(det)

        total_frame = sum(frame_counts.values())
        self.total_frames_processed += 1
        if total_frame > self.peak_frame_count:
            self.peak_frame_count = total_frame

        # Track cumulative counts
        for k, v in frame_counts.items():
            self.cumulative_counts[k] += v

        record = {
            "frame_index": int(frame_idx),
            "timestamp_sec": round(float(timestamp_sec), 3) if timestamp_sec is not None else round(frame_idx / 30.0, 3),
            "total_vehicles": total_frame,
            "counts": frame_counts,
            "counted_detections": counted_detections,
        }
        self.history.append(record)

        return record

    def annotate_frame(
        self,
        frame: np.ndarray,
        frame_metrics: Dict[str, Any],
        show_roi: bool = True,
    ) -> np.ndarray:
        """
        Overlays vehicle counts, category distribution, and ROI boundaries onto the frame.
        
        Args:
            frame: Input image array in BGR format.
            frame_metrics: Output from count_frame().
            show_roi: Whether to draw ROI boundary lines.
            
        Returns:
            np.ndarray: Annotated frame with counting HUD.
        """
        annotated = frame.copy()
        counts = frame_metrics.get("counts", {})
        total = frame_metrics.get("total_vehicles", 0)

        # Draw ROI polygon or bounding box if enabled
        if show_roi:
            if self.roi_polygon is not None:
                cv2.polylines(annotated, [self.roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
                cv2.putText(
                    annotated, "COUNTING ROI",
                    (int(self.roi_polygon[0][0]), int(self.roi_polygon[0][1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                )
            elif self.roi_bbox is not None:
                rx1, ry1, rx2, ry2 = self.roi_bbox
                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
                cv2.putText(
                    annotated, "COUNTING ROI",
                    (rx1, max(15, ry1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                )

        # Draw HUD overlay panel in top-left corner
        panel_w = 340
        panel_h = 105
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Header: Total Vehicle Count
        cv2.putText(
            annotated,
            f"ACTIVE VEHICLES: {total}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Category Breakdown
        cv2.putText(
            annotated,
            f"Cars: {counts.get('car', 0)}   | Motorcycles: {counts.get('motorcycle', 0)}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Buses: {counts.get('bus', 0)}  | Trucks: {counts.get('truck', 0)}",
            (20, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )

        return annotated

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns a comprehensive summary of vehicle counts across all processed frames.
        
        Returns:
            Dict[str, Any]: Overall statistics.
        """
        avg_vehicles = (
            round(sum(h["total_vehicles"] for h in self.history) / len(self.history), 2)
            if self.history else 0.0
        )
        category_sums: Dict[str, int] = {k: 0 for k in self.SUPPORTED_CLASSES}
        for h in self.history:
            for k in self.SUPPORTED_CLASSES:
                category_sums[k] += h["counts"].get(k, 0)

        return {
            "total_frames_analyzed": len(self.history),
            "peak_vehicles_in_single_frame": self.peak_frame_count,
            "average_vehicles_per_frame": avg_vehicles,
            "cumulative_category_counts": category_sums,
        }

    def export_json(self, output_path: Union[Path, str]) -> Path:
        """
        Exports counting time-series and summary data to a JSON file on Windows.
        
        Args:
            output_path: Path to destination JSON file.
            
        Returns:
            Path: Resolved path of exported file.
        """
        dest = Path(output_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "summary": self.get_summary(),
            "frame_history": [
                {
                    "frame_index": h["frame_index"],
                    "timestamp_sec": h["timestamp_sec"],
                    "total_vehicles": h["total_vehicles"],
                    "counts": h["counts"],
                }
                for h in self.history
            ],
        }

        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Exported vehicle counting JSON to: %s", dest)
        return dest

    def export_csv(self, output_path: Union[Path, str]) -> Path:
        """
        Exports counting time-series to a standard CSV file on Windows.
        
        Args:
            output_path: Path to destination CSV file.
            
        Returns:
            Path: Resolved path of exported file.
        """
        dest = Path(output_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "frame_index",
            "timestamp_sec",
            "car",
            "motorcycle",
            "bus",
            "truck",
            "total_vehicles",
        ]

        with open(dest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for h in self.history:
                row = {
                    "frame_index": h["frame_index"],
                    "timestamp_sec": h["timestamp_sec"],
                    "car": h["counts"].get("car", 0),
                    "motorcycle": h["counts"].get("motorcycle", 0),
                    "bus": h["counts"].get("bus", 0),
                    "truck": h["counts"].get("truck", 0),
                    "total_vehicles": h["total_vehicles"],
                }
                writer.writerow(row)

        logger.info("Exported vehicle counting CSV to: %s", dest)
        return dest

    def reset(self) -> None:
        """Resets all metrics and history."""
        self.cumulative_counts = {k: 0 for k in self.SUPPORTED_CLASSES}
        self.peak_frame_count = 0
        self.total_frames_processed = 0
        self.history.clear()
