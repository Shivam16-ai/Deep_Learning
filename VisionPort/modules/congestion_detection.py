"""
Module: congestion_detection.py
Task 11: Traffic congestion detection.

Monitors traffic density persistence over consecutive frames to detect sustained high-traffic
congestion events, suppresses spurious single-frame spikes via temporal hysteresis,
records congestion start and end timestamps, tracks event durations, and renders real-time
visual congestion alerts onto video streams.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import csv
import json
import cv2
import numpy as np

from utils.logger import setup_logger

logger = setup_logger("CongestionDetection")


class CongestionDetector:
    """
    Evaluates temporal persistence of high traffic density, generates real-time congestion
    alerts, and records historical incident logs with start, end, and duration metrics.
    """

    def __init__(
        self,
        consecutive_frames_threshold: int = 5,
        recovery_frames_threshold: int = 3,
        fps: float = 30.0,
    ):
        """
        Initializes the CongestionDetector.
        
        Args:
            consecutive_frames_threshold: Minimum consecutive HIGH density frames required
                to trigger a congestion alert (avoids false alarms from momentary spikes).
            recovery_frames_threshold: Consecutive non-HIGH frames required to mark
                congestion as resolved.
            fps: Video frames per second (used for duration calculations).
        """
        self.consecutive_frames_threshold = max(1, int(consecutive_frames_threshold))
        self.recovery_frames_threshold = max(1, int(recovery_frames_threshold))
        self.fps = float(fps) if fps > 0 else 30.0

        # State tracking
        self.is_congested: bool = False
        self.consecutive_high_frames: int = 0
        self.consecutive_recovery_frames: int = 0
        self.total_frames_processed: int = 0

        # Current active event
        self.active_event: Optional[Dict[str, Any]] = None

        # Completed historical events
        self.events: List[Dict[str, Any]] = []
        self.event_counter: int = 0

        # Per-frame status history
        self.history: List[Dict[str, Any]] = []

        logger.info(
            "CongestionDetector initialized | Persistence Threshold: %d frames | Recovery Threshold: %d frames",
            self.consecutive_frames_threshold, self.recovery_frames_threshold
        )

    def evaluate_frame(
        self,
        density_metrics: Dict[str, Any],
        frame_idx: int = 0,
        timestamp_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates congestion state given the current frame's density metrics.
        
        Args:
            density_metrics: Output from TrafficDensityCalculator.calculate_density().
            frame_idx: Sequential frame index.
            timestamp_sec: Video timestamp in seconds.
            
        Returns:
            Dict[str, Any]: Current congestion evaluation including alert status and duration.
        """
        density_level = density_metrics.get("density_level", "LOW")
        vehicle_count = density_metrics.get("vehicle_count", 0)
        ts = round(float(timestamp_sec), 3) if timestamp_sec is not None else round(frame_idx / self.fps, 3)
        self.total_frames_processed += 1

        alert_generated = False
        alert_message = ""

        if density_level == "HIGH":
            self.consecutive_high_frames += 1
            self.consecutive_recovery_frames = 0

            # Check if threshold reached to trigger congestion
            if self.consecutive_high_frames >= self.consecutive_frames_threshold:
                if not self.is_congested:
                    # New congestion event triggered
                    self.is_congested = True
                    self.event_counter += 1
                    start_frame = frame_idx - self.consecutive_high_frames + 1
                    start_ts = round(start_frame / self.fps, 3)

                    self.active_event = {
                        "event_id": self.event_counter,
                        "start_frame": int(start_frame),
                        "start_time_sec": start_ts,
                        "end_frame": None,
                        "end_time_sec": None,
                        "duration_frames": self.consecutive_high_frames,
                        "duration_sec": round(self.consecutive_high_frames / self.fps, 2),
                        "peak_vehicle_count": vehicle_count,
                        "status": "ACTIVE",
                    }
                    alert_generated = True
                    alert_message = (
                        f"CONGESTION_ALERT: Sustained High Traffic Detected "
                        f"({self.consecutive_high_frames} consecutive frames, {vehicle_count} vehicles)"
                    )
                    logger.warning("Event #%d: %s at t=%.2fs", self.event_counter, alert_message, ts)
                else:
                    # Update active event
                    if self.active_event is not None:
                        self.active_event["duration_frames"] += 1
                        self.active_event["duration_sec"] = round(self.active_event["duration_frames"] / self.fps, 2)
                        if vehicle_count > self.active_event["peak_vehicle_count"]:
                            self.active_event["peak_vehicle_count"] = vehicle_count

        else:
            self.consecutive_high_frames = 0
            if self.is_congested:
                self.consecutive_recovery_frames += 1
                if self.consecutive_recovery_frames >= self.recovery_frames_threshold:
                    # Congestion resolved
                    self.is_congested = False
                    if self.active_event is not None:
                        self.active_event["end_frame"] = int(frame_idx)
                        self.active_event["end_time_sec"] = ts
                        self.active_event["status"] = "RESOLVED"
                        self.events.append(dict(self.active_event))
                        logger.info(
                            "Event #%d: Congestion RESOLVED after %.2fs (%d frames) at t=%.2fs",
                            self.active_event["event_id"], self.active_event["duration_sec"],
                            self.active_event["duration_frames"], ts
                        )
                        self.active_event = None

        # Build current frame evaluation
        current_duration = self.active_event["duration_sec"] if self.active_event else 0.0
        status_label = "CONGESTED" if self.is_congested else ("MODERATE" if density_level == "MEDIUM" else "FREE_FLOW")

        record = {
            "frame_index": int(frame_idx),
            "timestamp_sec": ts,
            "status": status_label,
            "is_congested": self.is_congested,
            "consecutive_high_frames": self.consecutive_high_frames,
            "active_duration_sec": current_duration,
            "alert_generated": alert_generated,
            "alert_message": alert_message,
        }
        self.history.append(record)
        return record

    def finalize(self) -> None:
        """Closes any currently active event at end of video/stream processing."""
        if self.is_congested and self.active_event is not None:
            last_frame = self.total_frames_processed - 1
            last_ts = round(last_frame / self.fps, 3)
            self.active_event["end_frame"] = last_frame
            self.active_event["end_time_sec"] = last_ts
            self.active_event["status"] = "RESOLVED_AT_STREAM_END"
            self.events.append(dict(self.active_event))
            self.active_event = None
            self.is_congested = False

    def annotate_frame(
        self,
        frame: np.ndarray,
        congestion_metrics: Dict[str, Any],
    ) -> np.ndarray:
        """
        Renders prominent visual congestion alert banners onto the video frame when active.
        
        Args:
            frame: Input image array in BGR format.
            congestion_metrics: Output from evaluate_frame().
            
        Returns:
            np.ndarray: Annotated frame with alert graphics.
        """
        annotated = frame.copy()
        is_congested = congestion_metrics.get("is_congested", False)
        duration = congestion_metrics.get("active_duration_sec", 0.0)

        h, w = frame.shape[:2]

        if is_congested:
            # Render vibrant red top alert banner
            banner_h = 42
            banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
            banner[:] = (0, 0, 180)  # Red background
            
            # Blend into top of frame
            annotated[0:banner_h, :] = cv2.addWeighted(banner, 0.85, annotated[0:banner_h, :], 0.15, 0)

            # Warning text with pulsing duration
            msg = f"WARNING: TRAFFIC CONGESTION DETECTED | Duration: {duration:.1f}s | High Density Sustained"
            cv2.putText(
                annotated,
                msg,
                (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns statistical summary of all detected congestion incidents.
        
        Returns:
            Dict[str, Any]: Summary metrics.
        """
        total_duration = sum(e.get("duration_sec", 0.0) for e in self.events)
        max_duration = max((e.get("duration_sec", 0.0) for e in self.events), default=0.0)
        avg_duration = round(total_duration / len(self.events), 2) if self.events else 0.0

        total_video_time = round(self.total_frames_processed / self.fps, 2) if self.fps > 0 else 0.0
        congested_pct = (
            round((total_duration / total_video_time) * 100.0, 2)
            if total_video_time > 0 else 0.0
        )

        return {
            "total_frames_analyzed": self.total_frames_processed,
            "total_congestion_events": len(self.events),
            "total_congested_time_sec": round(total_duration, 2),
            "average_event_duration_sec": avg_duration,
            "maximum_event_duration_sec": round(max_duration, 2),
            "percent_time_congested": congested_pct,
            "events": self.events,
        }

    def export_json(self, output_path: Union[Path, str]) -> Path:
        """Exports all congestion events and summary statistics to JSON on Windows."""
        dest = Path(output_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "summary": self.get_summary(),
            "events": self.events,
            "frame_history": self.history,
        }

        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Exported congestion detection JSON to: %s", dest)
        return dest

    def export_csv(self, output_path: Union[Path, str]) -> Path:
        """Exports list of congestion incidents to a CSV file on Windows."""
        dest = Path(output_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "event_id",
            "start_frame",
            "start_time_sec",
            "end_frame",
            "end_time_sec",
            "duration_frames",
            "duration_sec",
            "peak_vehicle_count",
            "status",
        ]

        with open(dest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ev in self.events:
                writer.writerow(ev)

        logger.info("Exported congestion events CSV to: %s", dest)
        return dest

    def reset(self) -> None:
        """Resets detector state."""
        self.is_congested = False
        self.consecutive_high_frames = 0
        self.consecutive_recovery_frames = 0
        self.total_frames_processed = 0
        self.active_event = None
        self.events.clear()
        self.event_counter = 0
        self.history.clear()
