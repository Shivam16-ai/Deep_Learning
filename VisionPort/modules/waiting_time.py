"""
Module: waiting_time.py
Task 15: Traffic waiting-time estimation.

Monitors vehicle stationary durations within intersection queue zones,
calculating average delay times and queue dissipation rates.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from utils.logger import setup_logger
from utils.config import get_path, load_config

logger = setup_logger("Task15_WaitingTime")


class WaitingTimeEstimator:
    """
    Estimates vehicle idling and waiting times at traffic junctions.
    """

    def __init__(
        self,
        stationary_speed_threshold_kmh: float = 3.0,
        fps: float = 30.0,
    ):
        """
        Initializes the waiting time estimator.
        
        Args:
            stationary_speed_threshold_kmh (float): Speeds below this are considered waiting/stopped.
            fps (float): Video stream frame rate used for frame-to-time conversion.
        """
        self.stationary_threshold = stationary_speed_threshold_kmh
        self.fps = fps
        # Map of track_id -> frames accumulated stationary
        self.waiting_frames: Dict[int, int] = {}
        # Map of track_id -> total waiting time in seconds
        self.completed_waits: Dict[int, float] = {}

        logger.info(
            "WaitingTimeEstimator initialized | Stationary speed threshold: < %.1f km/h | FPS: %.1f",
            stationary_speed_threshold_kmh, fps
        )

    def update_vehicle_state(self, track_id: int, current_speed_kmh: float) -> float:
        """
        Updates the waiting status of a single tracked vehicle.
        
        Args:
            track_id (int): Identifier for vehicle.
            current_speed_kmh (float): Current estimated speed in km/h.
            
        Returns:
            float: Cumulative seconds vehicle has been waiting.
        """
        if current_speed_kmh < self.stationary_threshold:
            self.waiting_frames[track_id] = self.waiting_frames.get(track_id, 0) + 1
        else:
            if track_id in self.waiting_frames and self.waiting_frames[track_id] > 0:
                elapsed_sec = self.waiting_frames[track_id] / self.fps
                self.completed_waits[track_id] = elapsed_sec
                self.waiting_frames[track_id] = 0

        current_wait_sec = (self.waiting_frames.get(track_id, 0)) / self.fps
        return round(current_wait_sec, 1)

    def get_waiting_statistics(self) -> Dict[str, Any]:
        """
        Computes aggregate statistics of traffic delays.
        
        Returns:
            Dict[str, Any]: Aggregate delay summary containing:
                - 'currently_waiting_count': int
                - 'avg_waiting_time_sec': float
                - 'max_waiting_time_sec': float
        """
        active_waits = [
            frames / self.fps
            for frames in self.waiting_frames.values()
            if frames > 0
        ]
        all_waits = active_waits + list(self.completed_waits.values())

        avg_wait = sum(all_waits) / len(all_waits) if all_waits else 0.0
        max_wait = max(all_waits) if all_waits else 0.0

        stats = {
            "currently_waiting_count": len(active_waits),
            "avg_waiting_time_sec": round(avg_wait, 1),
            "max_waiting_time_sec": round(max_wait, 1),
        }
        return stats

    def export_report(self, output_dir: Optional[Path | str] = None) -> Optional[Path]:
        """
        Exports the current waiting time statistics to a JSON report file in output/reports/.
        
        Args:
            output_dir (Optional[Path | str]): Destination directory for the report.
            
        Returns:
            Optional[Path]: Path to written report file, or None on error.
        """
        if output_dir is None:
            try:
                cfg = load_config()
                target_dir = get_path("reports_dir", cfg)
            except Exception:
                target_dir = Path("output/reports").resolve()
        else:
            target_dir = Path(output_dir).resolve()

        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = target_dir / f"waiting_time_report_{timestamp}.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.get_waiting_statistics(),
            "individual_waits_seconds": {str(k): round(v, 1) for k, v in self.completed_waits.items()},
        }

        try:
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.info("Exported waiting time report to: %s", report_file)
            return report_file
        except Exception as exc:
            logger.exception("Failed to export waiting time report: %s", exc)
            return None
