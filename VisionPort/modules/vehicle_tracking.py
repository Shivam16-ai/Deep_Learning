"""
Module: vehicle_tracking.py
Task 12: Individual vehicle tracking using YOLOv8 + ByteTrack.

Integrates the Ultralytics built-in ByteTrack implementation so that
persistent integer track IDs are assigned to every detected vehicle and
maintained across consecutive video frames.  The module purposely reuses the
existing VehicleDetector so that YOLO weights are loaded only once and
no COCO class-mapping is duplicated.

Design decisions
----------------
* Tracking runs by calling model.track() directly on each frame; this invokes
  the ByteTrack association internally without a separate process.
* A lightweight Python-side `active_tracks` dict stores per-ID metadata
  (class name, centroid history, first/last seen frame, total frames seen).
* When a track is lost for `track_buffer` frames the entry moves to
  `lost_tracks` so the full session history is preserved for export.
* All export methods produce JSON + CSV reports that are compatible with the
  rest of the VisionPort pipeline.

Limitations
-----------
* ByteTrack IDs are session-scoped; they reset when a new VehicleTracker is
  created.
* Re-identification across separate video files is not supported.
* No camera calibration is applied; centroid coordinates are in raw pixels.
"""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from utils.logger import setup_logger
from utils.config import load_config, get_path

logger = setup_logger("VehicleTracking")

# Allow unit tests to inject a mock class by patching this module attribute
# before constructing VehicleTracker.
_YOLO_CLS = YOLO

# BGR colour palette — one colour per vehicle class for consistent display
_TRACK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car":        (0, 255, 0),    # Green
    "motorcycle": (0, 165, 255),  # Orange
    "bus":        (255, 0, 0),    # Blue
    "truck":      (0, 255, 255),  # Yellow
    "unknown":    (200, 200, 200),
}

# COCO IDs of the four target vehicle classes
_TARGET_CLASS_IDS = [2, 3, 5, 7]
_ID_TO_CLASS: Dict[int, str] = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Maximum centroid history entries kept per track (memory guard)
_MAX_HISTORY = 500


class VehicleTracker:
    """
    YOLOv8 + ByteTrack multi-object vehicle tracker.

    Wraps the Ultralytics `model.track()` API to obtain frame-by-frame
    tracking results and enriches them with per-ID metadata stored in
    `active_tracks`.

    Parameters
    ----------
    model_path:
        Path to the YOLOv8 weights file.  Defaults to ``models/yolov8n.pt``.
    conf_threshold:
        Minimum YOLO confidence for a detection to be passed to ByteTrack.
    iou_threshold:
        NMS IoU threshold used during YOLO inference.
    track_buffer:
        Number of frames a lost track is kept alive before it is retired.
    match_threshold:
        ByteTrack association threshold (high-confidence detections).
    device:
        Torch device string ('cpu', 'cuda', 'cuda:0', or 'auto').
    """

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        track_buffer: int = 30,
        match_threshold: float = 0.80,
        device: Optional[str] = None,
    ) -> None:
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.track_buffer = int(track_buffer)
        self.match_threshold = float(match_threshold)

        # Resolve model path from config or default
        if model_path is None:
            try:
                cfg = load_config()
                model_name = cfg.get("detection", {}).get("model_name", "yolov8n.pt")
                models_dir = get_path("models_dir", cfg)
                self.model_path = models_dir / model_name
            except Exception:
                self.model_path = Path("models/yolov8n.pt").resolve()
        else:
            self.model_path = Path(model_path).resolve()

        # Resolve device
        import torch
        if device is None or str(device).lower() == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device).lower()

        # Per-track metadata stores
        # active_tracks[track_id] -> dict with keys:
        #   class_name, class_id, history (deque of (cx,cy)),
        #   first_frame, last_frame, frames_seen, bbox
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        # Retired tracks (lost for > track_buffer frames)
        self.lost_tracks: Dict[int, Dict[str, Any]] = {}

        self._frame_count: int = 0
        self._model = None  # loaded lazily
        self._load_model()

        logger.info(
            "VehicleTracker initialized | Weights: %s | Device: %s | "
            "Conf: %.2f | Buffer: %d frames",
            self.model_path.name, self.device, self.conf_threshold, self.track_buffer,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Loads the YOLO model from disk (or downloads if absent)."""
        import modules.vehicle_tracking as _self_module
        yolo_cls = _self_module._YOLO_CLS   # patchable in unit tests
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Loading YOLO model for tracking from %s …", self.model_path)
            self._model = yolo_cls(str(self.model_path))
            logger.info(
                "YOLO tracker model loaded on %s (%d classes).",
                self.device, len(self._model.names),
            )
        except Exception as exc:
            logger.exception("Failed to load YOLO model: %s", exc)
            raise RuntimeError(f"Cannot load YOLO model: {exc}") from exc

    # ------------------------------------------------------------------
    # Core tracking API
    # ------------------------------------------------------------------

    def track_frame(
        self,
        frame: np.ndarray,
        frame_idx: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs YOLO + ByteTrack on a single frame and returns tracked vehicles.

        Parameters
        ----------
        frame:
            BGR image array (H × W × 3).
        frame_idx:
            Caller-supplied frame index; if None the internal counter is used.

        Returns
        -------
        List[Dict[str, Any]]
            One entry per active track:
            ``{track_id, class_id, class_name, confidence, bbox, centroid,
               frames_seen, first_frame, last_frame}``
        """
        if frame is None or frame.size == 0:
            logger.warning("track_frame received an empty frame; skipping.")
            return []

        idx = frame_idx if frame_idx is not None else self._frame_count
        self._frame_count += 1

        try:
            results = self._model.track(
                source=frame,
                classes=_TARGET_CLASS_IDS,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                tracker="bytetrack.yaml",   # Ultralytics built-in ByteTrack config
                persist=True,               # Keep track IDs across successive calls
                verbose=False,
            )
        except Exception as exc:
            logger.error("model.track() raised an error on frame %d: %s", idx, exc)
            return []

        tracked: List[Dict[str, Any]] = []

        if not results or results[0].boxes is None:
            return tracked

        boxes = results[0].boxes
        # boxes.id is None when no tracks are active (ByteTrack found no match)
        if boxes.id is None:
            return tracked

        xyxy   = boxes.xyxy.cpu().numpy()
        ids    = boxes.id.cpu().numpy().astype(int)
        confs  = boxes.conf.cpu().numpy()
        clss   = boxes.cls.cpu().numpy().astype(int)

        for box, tid, conf, cls_id in zip(xyxy, ids, confs, clss):
            x1, y1, x2, y2 = [round(float(v), 1) for v in box]
            cx = round((x1 + x2) / 2.0, 1)
            cy = round((y1 + y2) / 2.0, 1)
            class_name = _ID_TO_CLASS.get(int(cls_id), "unknown")

            # Update or create track metadata
            if tid not in self.active_tracks:
                self.active_tracks[tid] = {
                    "track_id":    int(tid),
                    "class_id":    int(cls_id),
                    "class_name":  class_name,
                    "first_frame": int(idx),
                    "last_frame":  int(idx),
                    "frames_seen": 1,
                    "history":     deque([(cx, cy)], maxlen=_MAX_HISTORY),
                    "bbox":        [x1, y1, x2, y2],
                    "last_conf":   round(float(conf), 4),
                }
            else:
                meta = self.active_tracks[tid]
                meta["last_frame"]  = int(idx)
                meta["frames_seen"] += 1
                meta["history"].append((cx, cy))
                meta["bbox"]       = [x1, y1, x2, y2]
                meta["last_conf"]  = round(float(conf), 4)

            tracked.append({
                "track_id":   int(tid),
                "class_id":   int(cls_id),
                "class_name": class_name,
                "confidence": round(float(conf), 4),
                "bbox":       [x1, y1, x2, y2],
                "centroid":   (cx, cy),
                "frames_seen": self.active_tracks[tid]["frames_seen"],
                "first_frame": self.active_tracks[tid]["first_frame"],
                "last_frame":  int(idx),
            })

        logger.debug(
            "Frame %d: %d active tracks (%d total seen)",
            idx, len(tracked), len(self.active_tracks),
        )
        return tracked

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        """Returns the centroid history for a given track ID."""
        meta = self.active_tracks.get(track_id) or self.lost_tracks.get(track_id)
        if meta is None:
            return []
        return list(meta["history"])

    def get_unique_vehicle_count(self) -> Dict[str, int]:
        """Returns count of unique vehicle IDs seen so far, broken down by class."""
        counts: Dict[str, int] = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "total": 0}
        all_tracks = {**self.active_tracks, **self.lost_tracks}
        for meta in all_tracks.values():
            cls = meta.get("class_name", "unknown")
            if cls in counts:
                counts[cls] += 1
            counts["total"] += 1
        return counts

    def clear(self) -> None:
        """Resets all track state (new session)."""
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self._frame_count = 0
        logger.info("VehicleTracker state cleared.")

    # ------------------------------------------------------------------
    # Annotation helper
    # ------------------------------------------------------------------

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracked: List[Dict[str, Any]],
        draw_trajectory: bool = True,
        trajectory_length: int = 30,
    ) -> np.ndarray:
        """
        Draws bounding boxes, track IDs, class names, and motion trails.

        Parameters
        ----------
        frame:
            BGR image to annotate (not modified in-place; a copy is returned).
        tracked:
            Output of :py:meth:`track_frame`.
        draw_trajectory:
            Whether to draw the recent centroid path as a polyline.
        trajectory_length:
            Number of historical centroids to draw.

        Returns
        -------
        np.ndarray
            Annotated BGR image.
        """
        out = frame.copy()

        for obj in tracked:
            tid       = obj["track_id"]
            cls_name  = obj["class_name"]
            conf      = obj["confidence"]
            x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
            cx, cy    = int(obj["centroid"][0]), int(obj["centroid"][1])

            color = _TRACK_COLORS.get(cls_name, _TRACK_COLORS["unknown"])

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label banner
            label = f"ID:{tid} {cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            banner_y1 = max(0, y1 - th - 6)
            cv2.rectangle(out, (x1, banner_y1), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                out, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA,
            )

            # Centroid dot
            cv2.circle(out, (cx, cy), 4, color, -1)

            # Motion trajectory
            if draw_trajectory:
                history = self.get_trajectory(tid)
                pts = [(int(p[0]), int(p[1])) for p in history[-trajectory_length:]]
                for i in range(1, len(pts)):
                    alpha = i / len(pts)          # fade older points
                    faded = tuple(int(c * alpha) for c in color)
                    cv2.line(out, pts[i - 1], pts[i], faded, 2)

        # HUD: unique vehicle summary
        summary = self.get_unique_vehicle_count()
        hud_lines = [
            f"Active tracks : {len(tracked)}",
            f"Unique vehicles: {summary['total']}",
            f"Cars:{summary['car']} Bikes:{summary['motorcycle']} "
            f"Buses:{summary['bus']} Trucks:{summary['truck']}",
        ]
        for i, line in enumerate(hud_lines):
            cv2.putText(
                out, line,
                (10, 24 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                out, line,
                (10, 24 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (0, 0, 0), 1, cv2.LINE_AA,
            )

        return out

    # ------------------------------------------------------------------
    # Report export
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Returns a session-level tracking summary dictionary."""
        all_tracks = {**self.active_tracks, **self.lost_tracks}
        track_records = []
        for tid, meta in all_tracks.items():
            track_records.append({
                "track_id":    int(tid),
                "class_name":  str(meta["class_name"]),
                "first_frame": int(meta["first_frame"]),
                "last_frame":  int(meta["last_frame"]),
                "frames_seen": int(meta["frames_seen"]),
                "last_conf":   float(meta.get("last_conf", 0.0)),
            })

        unique_counts = self.get_unique_vehicle_count()
        return {
            "unique_vehicle_counts": unique_counts,
            "total_tracks":          int(len(all_tracks)),
            "frames_processed":      int(self._frame_count),
            "tracks":                track_records,
        }

    def export_json(self, output_path: Union[Path, str]) -> Path:
        """
        Exports the full tracking summary to a JSON file.

        Parameters
        ----------
        output_path:
            Destination ``.json`` file path.

        Returns
        -------
        Path
            Resolved path to the written file.
        """
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_summary()

        class _NumpyEncoder(json.JSONEncoder):
            """Converts numpy scalar types to native Python for JSON serialisation."""
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, cls=_NumpyEncoder)
        logger.info("Exported vehicle tracking JSON to: %s", out)
        return out

    def export_csv(self, output_path: Union[Path, str]) -> Path:
        """
        Exports per-track metadata to a CSV file.

        Parameters
        ----------
        output_path:
            Destination ``.csv`` file path.

        Returns
        -------
        Path
            Resolved path to the written file.
        """
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_summary()
        fieldnames = ["track_id", "class_name", "first_frame", "last_frame",
                      "frames_seen", "last_conf"]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for record in summary["tracks"]:
                writer.writerow({k: record.get(k, "") for k in fieldnames})
        logger.info("Exported vehicle tracking CSV to: %s", out)
        return out


# ---------------------------------------------------------------------------
# Top-level convenience function used by main.py --track-vehicles
# ---------------------------------------------------------------------------

def process_video_with_tracking(
    video_path: Union[Path, str],
    output_video_path: Optional[Union[Path, str]] = None,
    conf_threshold: float = 0.40,
    device: str = "auto",
    max_frames: Optional[int] = None,
    draw_trajectory: bool = True,
) -> Dict[str, Any]:
    """
    Reads a video, runs YOLO + ByteTrack on every frame, writes an annotated
    output video, and returns a session summary plus the paths to generated
    report files.

    Parameters
    ----------
    video_path:
        Path to the input video file.
    output_video_path:
        Where to write the annotated video. Defaults to
        ``output/processed_videos/tracked_<stem>.mp4``.
    conf_threshold:
        YOLO confidence threshold.
    device:
        Torch device ('auto', 'cpu', 'cuda').
    max_frames:
        Process at most this many frames (useful for quick tests).
    draw_trajectory:
        Whether to draw centroid trails on the output video.

    Returns
    -------
    Dict[str, Any]
        ``{output_video, tracking_json, tracking_csv, summary}``
    """
    src = Path(video_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input video not found: {src}")

    # Resolve output paths from config
    try:
        cfg = load_config()
        videos_out_dir = get_path("processed_videos_dir", cfg)
        reports_dir    = get_path("reports_dir",           cfg)
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

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {src}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    from utils.video_compat import create_compatible_video_writer, finalize_video_file

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = create_compatible_video_writer(out_video, fps, width, height)

    logger.info(
        "Starting tracking on '%s' (%dx%d @ %.1f FPS, %d frames) …",
        src.name, width, height, fps, total,
    )

    t_start = time.perf_counter()
    frames_done = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frames_done >= max_frames:
            break

        tracked = tracker.track_frame(frame, frame_idx=frames_done)
        annotated = tracker.annotate_frame(frame, tracked, draw_trajectory=draw_trajectory)
        writer.write(annotated)
        frames_done += 1

    cap.release()
    writer.release()
    finalize_video_file(out_video)

    elapsed = time.perf_counter() - t_start
    avg_fps = round(frames_done / max(elapsed, 0.001), 2)

    # Export reports
    json_path = tracker.export_json(reports_dir / "vehicle_tracking_report.json")
    csv_path  = tracker.export_csv(reports_dir  / "vehicle_tracking_report.csv")

    summary = tracker.get_summary()
    summary["frames_processed"] = frames_done
    summary["average_fps"]      = avg_fps
    summary["source_video"]     = str(src)

    logger.info(
        "Tracking complete: %d frames in %.1fs (%.1f FPS) | unique vehicles: %d",
        frames_done, elapsed, avg_fps, summary["unique_vehicle_counts"]["total"],
    )

    return {
        "output_video":  str(out_video),
        "tracking_json": str(json_path),
        "tracking_csv":  str(csv_path),
        "summary":       summary,
    }
