"""
Module: speed_estimation.py
Task 13: Approximate vehicle speed estimation.

Estimates the speed of individually tracked vehicles by measuring the
Euclidean pixel displacement of their centroids between successive frames
and converting that displacement to km/h using a user-supplied pixel-to-meter
calibration factor.

IMPORTANT – calibration disclaimer
------------------------------------
Real-world speed accuracy depends entirely on the correctness of the
``pixels_per_meter`` value.  This value is scene-specific and must be
measured by the operator using a physical reference object of known size
visible in the camera frame (e.g. a standard lane width of 3.5 m, a road
marking, or a vehicle of known length).

When no calibration has been supplied (``calibrated = False``) the module
still computes pixel-displacement-based estimates but clearly labels every
output value as *uncalibrated* (px/s, not real km/h).

Speed history and smoothing
----------------------------
Each track maintains a fixed-size deque of recent raw speed measurements.
The smoothed speed is the median of that window, which is robust to single
noisy outliers (e.g. caused by detection jitter in a single frame).

Limitations
-----------
1. Perspective distortion – objects far from the camera appear to move
   faster per pixel; a flat homography or per-zone calibration improves
   accuracy.
2. Camera motion – any camera shake adds to the apparent vehicle motion.
3. Sampling rate – speed is computed only when two consecutive detections
   for the same ID exist; a gap (missed detection) inflates the apparent
   displacement.
4. Minimum history – speed is only reported after a track has been seen for
   at least `min_frames_for_speed` frames to avoid spurious single-frame
   spikes.
"""

from __future__ import annotations

import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

from utils.logger import setup_logger
from utils.config import load_config

logger = setup_logger("SpeedEstimation")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KMH_FACTOR = 3.6          # 1 m/s = 3.6 km/h
_DEFAULT_PPM = 8.5          # pixels per metre – placeholder, NOT accurate
_MIN_PIXEL_DISP = 1.0       # ignore sub-pixel movements (stationary noise)
_SMOOTHING_WINDOW = 7       # number of samples in the median filter window
_MIN_FRAMES = 3             # minimum consecutive detections before reporting speed
_MAX_SPEED_KMH = 250.0      # implausible-speed ceiling (measurement artefact guard)


# ---------------------------------------------------------------------------
# SpeedEstimator
# ---------------------------------------------------------------------------


class SpeedEstimator:
    """
    Estimates vehicle speeds from tracker centroid histories.

    Parameters
    ----------
    pixels_per_meter:
        Calibration ratio: how many image pixels correspond to 1 real metre.
        **Must be measured in the actual camera scene.**  Pass ``None`` to
        operate in uncalibrated (pixel-based) mode.
    fps:
        Video frame rate used to convert frame-to-frame displacement into
        a velocity.
    smoothing_window:
        Number of recent raw speed samples used for the median filter.
    min_frames_for_speed:
        A track must have been seen for at least this many frames before a
        speed estimate is emitted.
    max_speed_kmh:
        Any estimate above this value is discarded as a measurement artefact.
    calibrated:
        If ``False`` (default when ``pixels_per_meter`` is not provided by
        the user), outputs are labelled as pixel-based estimates rather than
        real km/h.
    """

    def __init__(
        self,
        pixels_per_meter: Optional[float] = None,
        fps: float = 30.0,
        smoothing_window: int = _SMOOTHING_WINDOW,
        min_frames_for_speed: int = _MIN_FRAMES,
        max_speed_kmh: float = _MAX_SPEED_KMH,
    ) -> None:
        # Determine calibration status
        if pixels_per_meter is not None and float(pixels_per_meter) > 0:
            self.pixels_per_meter: float = float(pixels_per_meter)
            self.calibrated: bool = True
        else:
            # Try the new top-level speed_estimation config key first,
            # then fall back to the legacy traffic_analysis.speed_estimation path.
            # If both are absent or null, run uncalibrated.
            self.pixels_per_meter = _DEFAULT_PPM
            self.calibrated = False
            try:
                cfg = load_config()
                # New top-level key (added in Phase 4 config update)
                cfg_ppm = cfg.get("speed_estimation", {}).get("pixels_per_meter", None)
                # Legacy fallback
                if cfg_ppm is None:
                    cfg_ppm = (
                        cfg.get("traffic_analysis", {})
                           .get("speed_estimation", {})
                           .get("pixels_per_meter", None)
                    )
                if cfg_ppm is not None and float(cfg_ppm) > 0:
                    self.pixels_per_meter = float(cfg_ppm)
                    self.calibrated = True
                # else: stay uncalibrated with default placeholder
            except Exception:
                pass  # config unavailable – stay uncalibrated

        self.fps:                  float = max(float(fps), 1.0)
        self.smoothing_window:     int   = max(int(smoothing_window), 1)
        self.min_frames_for_speed: int   = max(int(min_frames_for_speed), 2)
        self.max_speed_kmh:        float = float(max_speed_kmh)

        # Per-track state
        # speed_history[tid] -> deque of raw speed values (km/h or px/s)
        self._speed_history: Dict[int, Deque[float]] = {}
        # centroid history: last observed (cx, cy) per track (for frame-delta calc)
        self._last_centroid: Dict[int, Tuple[float, float]] = {}
        # smoothed speed cache
        self._smoothed_speed: Dict[int, float] = {}
        # all measurements ever recorded (for export)
        self._records: List[Dict[str, Any]] = []

        unit = "km/h (calibrated)" if self.calibrated else "px/s (UNCALIBRATED)"
        logger.info(
            "SpeedEstimator initialized | %.2f px/m | FPS: %.1f | unit: %s | "
            "window: %d | min_frames: %d",
            self.pixels_per_meter, self.fps, unit,
            self.smoothing_window, self.min_frames_for_speed,
        )
        if not self.calibrated:
            logger.warning(
                "SpeedEstimator is running in UNCALIBRATED mode.  "
                "Provide a measured pixels_per_meter value for real-world km/h estimates."
            )

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    @staticmethod
    def pixel_displacement(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        """Euclidean distance in pixels between two centroid positions."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def raw_speed(
        self,
        displacement_px: float,
        frames_elapsed: int = 1,
    ) -> float:
        """
        Converts a pixel displacement over ``frames_elapsed`` frames into a
        speed value.

        Returns km/h when calibrated, or px/s when uncalibrated.
        """
        if frames_elapsed <= 0 or displacement_px < _MIN_PIXEL_DISP:
            return 0.0

        time_s = frames_elapsed / self.fps

        if self.calibrated:
            dist_m   = displacement_px / self.pixels_per_meter
            speed    = (dist_m / time_s) * _KMH_FACTOR   # km/h
        else:
            speed    = displacement_px / time_s            # px/s

        return round(speed, 2)

    def update(
        self,
        track_id: int,
        centroid: Tuple[float, float],
        frames_seen: int,
        frame_idx: int = 0,
    ) -> Optional[float]:
        """
        Updates the speed estimate for a single tracked vehicle.

        Called once per frame per active track.

        Parameters
        ----------
        track_id:
            Unique ByteTrack integer ID.
        centroid:
            Current (cx, cy) centroid in pixels.
        frames_seen:
            Total number of frames this track has been active.
        frame_idx:
            Current frame index (used for record-keeping only).

        Returns
        -------
        Optional[float]
            Smoothed speed (km/h or px/s), or ``None`` if insufficient history.
        """
        tid = int(track_id)

        # First observation — store centroid and return None
        if tid not in self._last_centroid:
            self._last_centroid[tid]  = centroid
            self._speed_history[tid]  = deque(maxlen=self.smoothing_window)
            return None

        # Compute raw speed from displacement since last call
        disp = self.pixel_displacement(self._last_centroid[tid], centroid)
        raw  = self.raw_speed(disp, frames_elapsed=1)
        self._last_centroid[tid] = centroid

        # Plausibility gate
        if raw > self.max_speed_kmh and self.calibrated:
            logger.debug("Track %d: speed %.1f exceeds ceiling %.1f; discarded.", tid, raw, self.max_speed_kmh)
            return self._smoothed_speed.get(tid)

        self._speed_history[tid].append(raw)

        # Only emit speed after minimum history
        if frames_seen < self.min_frames_for_speed:
            return None
        if len(self._speed_history[tid]) < 2:
            return None

        smoothed = round(float(np.median(list(self._speed_history[tid]))), 1)
        self._smoothed_speed[tid] = smoothed

        # Record measurement
        self._records.append({
            "frame_idx":   frame_idx,
            "track_id":    tid,
            "displacement_px": round(disp, 2),
            "raw_speed":   raw,
            "smoothed_speed": smoothed,
            "unit":        "km/h" if self.calibrated else "px/s",
            "calibrated":  self.calibrated,
        })

        return smoothed

    def get_speed(self, track_id: int) -> Optional[float]:
        """Returns the latest smoothed speed for a track (None if unknown)."""
        return self._smoothed_speed.get(int(track_id))

    def get_all_speeds(self) -> Dict[int, float]:
        """Returns a snapshot of the latest smoothed speed per track ID."""
        return dict(self._smoothed_speed)

    def get_average_speed(self) -> float:
        """Average smoothed speed across all tracks with valid estimates."""
        vals = list(self._smoothed_speed.values())
        if not vals:
            return 0.0
        return round(float(np.mean(vals)), 1)

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracked: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Overlays speed labels on each tracked vehicle bounding box.

        Parameters
        ----------
        frame:
            BGR image (not modified in-place).
        tracked:
            List of track dicts from VehicleTracker.track_frame().

        Returns
        -------
        np.ndarray
            Annotated BGR image.
        """
        import cv2
        out = frame.copy()
        unit = "km/h" if self.calibrated else "px/s*"

        for obj in tracked:
            tid = obj["track_id"]
            spd = self.get_speed(tid)
            if spd is None:
                label = f"ID:{tid} --{unit}"
            else:
                label = f"ID:{tid} {spd:.0f}{unit}"

            x1, y1 = int(obj["bbox"][0]), int(obj["bbox"][1])
            x2     = int(obj["bbox"][2])

            # Speed tag below the top-left corner
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            tag_y = min(frame.shape[0] - 5, y1 + th + 20)
            cv2.rectangle(out, (x1, tag_y - th - 4), (x1 + tw + 4, tag_y + 2),
                          (30, 30, 30), -1)
            cv2.putText(
                out, label, (x1 + 2, tag_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (255, 255, 0), 1, cv2.LINE_AA,
            )

        # Uncalibrated watermark
        if not self.calibrated:
            warn = "* SPEED UNCALIBRATED (px/s)"
            cv2.putText(
                out, warn,
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 100, 255), 2, cv2.LINE_AA,
            )

        return out

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Returns a session-level speed summary dictionary."""
        speeds = list(self._smoothed_speed.values())
        return {
            "calibrated":          self.calibrated,
            "pixels_per_meter":    self.pixels_per_meter,
            "fps":                 self.fps,
            "unit":                "km/h" if self.calibrated else "px/s",
            "tracks_with_speed":   len(speeds),
            "average_speed":       round(float(np.mean(speeds)), 1) if speeds else 0.0,
            "max_speed":           round(float(np.max(speeds)),  1) if speeds else 0.0,
            "min_speed":           round(float(np.min(speeds)),  1) if speeds else 0.0,
            "smoothing_window":    self.smoothing_window,
            "min_frames_required": self.min_frames_for_speed,
            "records":             self._records,
            "limitations": [
                "Accuracy requires a measured pixels_per_meter for the specific camera scene.",
                "Perspective distortion causes overestimation for far-field objects.",
                "Any camera shake adds apparent motion to all measurements.",
                "Missed detections (gaps) inflate displacement over the gap interval.",
                "Speed is unavailable until a track has been seen for min_frames_for_speed frames.",
            ],
        }

    def export_json(self, output_path: Union[Path, str]) -> Path:
        """Exports the speed summary to a JSON file."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self.get_summary(), fh, indent=2)
        logger.info("Exported speed estimation JSON to: %s", out)
        return out

    def export_csv(self, output_path: Union[Path, str]) -> Path:
        """Exports per-measurement records to a CSV file."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "frame_idx", "track_id", "displacement_px",
            "raw_speed", "smoothed_speed", "unit", "calibrated",
        ]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self._records:
                writer.writerow({k: rec.get(k, "") for k in fieldnames})
        logger.info("Exported speed estimation CSV to: %s", out)
        return out


# ---------------------------------------------------------------------------
# Integrated Video Processing Mode (Task 12 + Task 13)
# ---------------------------------------------------------------------------

def process_video_with_tracking_and_speed(
    video_path: Union[Path, str],
    output_video_path: Optional[Union[Path, str]] = None,
    conf_threshold: float = 0.40,
    pixels_per_meter: Optional[float] = None,
    device: str = "auto",
    max_frames: Optional[int] = None,
    draw_trajectory: bool = True,
) -> Dict[str, Any]:
    """
    Integrated video-processing mode for Tasks 12 & 13:
    1. Reads the video.
    2. Runs YOLO detection.
    3. Tracks individual vehicles with persistent IDs (ByteTrack).
    4. Estimates calibrated speed when calibration is available (or px/s if uncalibrated).
    5. Draws bounding boxes, tracking IDs, motion trails, and speed labels.
    6. Saves processed video to output/processed_videos/tracked_traffic.mp4.
    7. Exports vehicle_tracking_report and speed_estimation_report (JSON + CSV).

    Parameters
    ----------
    video_path:
        Path to the input video file.
    output_video_path:
        Destination video file path. Defaults to output/processed_videos/tracked_traffic.mp4.
    conf_threshold:
        YOLO detection confidence threshold.
    pixels_per_meter:
        Physical scale calibration ratio (px/m).
    device:
        Inference device ('cpu', 'cuda', 'auto').
    max_frames:
        Optional frame limit.
    draw_trajectory:
        Whether to render motion trails.

    Returns
    -------
    Dict[str, Any]
        Summary containing paths, frame count, average FPS, and statistics.
    """
    import time
    import cv2
    from utils.config import get_path
    from modules.vehicle_tracking import VehicleTracker

    src = Path(video_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input video not found: {src}")

    try:
        cfg = load_config()
        videos_out_dir = get_path("processed_videos_dir", cfg)
        reports_dir    = get_path("reports_dir", cfg)
    except Exception:
        videos_out_dir = Path("output/processed_videos").resolve()
        reports_dir    = Path("output/reports").resolve()

    videos_out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if output_video_path is None:
        out_video = videos_out_dir / "tracked_traffic.mp4"
    else:
        out_video = Path(output_video_path).resolve()
        out_video.parent.mkdir(parents=True, exist_ok=True)

    tracker = VehicleTracker(conf_threshold=conf_threshold, device=device)
    estimator = SpeedEstimator(pixels_per_meter=pixels_per_meter)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {src}")

    from utils.video_compat import create_compatible_video_writer, finalize_video_file

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = create_compatible_video_writer(out_video, fps, width, height)

    estimator.fps = fps
    frames_done = 0
    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frames_done >= max_frames:
            break

        tracked = tracker.track_frame(frame, frame_idx=frames_done)

        for obj in tracked:
            tid    = obj["track_id"]
            cx, cy = obj["centroid"]
            estimator.update(
                track_id=tid,
                centroid=(cx, cy),
                frames_seen=obj["frames_seen"],
                frame_idx=frames_done,
            )

        annotated = tracker.annotate_frame(frame, tracked, draw_trajectory=draw_trajectory)
        annotated = estimator.annotate_frame(annotated, tracked)
        writer.write(annotated)
        frames_done += 1

    cap.release()
    writer.release()
    finalize_video_file(out_video)

    elapsed = time.perf_counter() - t0
    avg_fps = round(frames_done / max(elapsed, 0.001), 2)

    tracking_json = tracker.export_json(reports_dir / "vehicle_tracking_report.json")
    tracking_csv  = tracker.export_csv(reports_dir  / "vehicle_tracking_report.csv")
    speed_json    = estimator.export_json(reports_dir / "speed_estimation_report.json")
    speed_csv     = estimator.export_csv(reports_dir  / "speed_estimation_report.csv")

    summary = {
        "source_video": str(src),
        "output_video": str(out_video),
        "frames_processed": frames_done,
        "average_fps": avg_fps,
        "calibrated": estimator.calibrated,
        "speed_unit": "km/h" if estimator.calibrated else "px/s",
        "average_speed": estimator.get_average_speed(),
        "tracking_summary": tracker.get_summary(),
        "speed_summary": estimator.get_summary(),
        "files_generated": [
            str(out_video),
            str(tracking_json),
            str(tracking_csv),
            str(speed_json),
            str(speed_csv),
        ],
    }

    logger.info(
        "Integrated tracking and speed estimation complete: %d frames -> %s",
        frames_done, out_video,
    )
    return summary
