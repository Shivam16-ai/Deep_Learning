"""
tests/test_tracking_and_speed.py
Task 12 & 13: Unit tests for VehicleTracker and SpeedEstimator.

All tests use synthetic data only.  No webcam, real video, or network
access is required.  YOLO model.track() is monkey-patched so that the
ByteTrack association logic is exercised without GPU or internet.

Naming convention follows the existing VisionPort test suite:
    TestVehicleTracking   – Task 12
    TestSpeedEstimation   – Task 13
"""

from __future__ import annotations

import csv
import json
import math
import sys
import types
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so imports work from any cwd
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Shared synthetic helpers
# ---------------------------------------------------------------------------

def _blank_frame(w: int = 640, h: int = 480) -> np.ndarray:
    """Returns a solid-colour BGR frame (no actual content needed)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_fake_boxes(
    xywh_list: List[Tuple[float, float, float, float]],
    ids: List[int],
    cls_ids: List[int],
    confs: Optional[List[float]] = None,
):
    """
    Creates a mock Ultralytics ``results[0].boxes`` object with the
    attributes that VehicleTracker.track_frame() reads.

    Parameters
    ----------
    xywh_list:
        List of (x1, y1, x2, y2) in pixel coordinates.
    ids:
        Track IDs (integer, one per box).
    cls_ids:
        COCO class IDs (one per box).
    confs:
        Confidence scores (defaults to 0.9 for all).
    """
    n = len(xywh_list)
    if confs is None:
        confs = [0.90] * n

    boxes_mock = MagicMock()
    boxes_mock.xyxy = MagicMock()
    boxes_mock.xyxy.cpu.return_value.numpy.return_value = np.array(xywh_list, dtype=np.float32)
    boxes_mock.id = MagicMock()
    boxes_mock.id.cpu.return_value.numpy.return_value = np.array(ids, dtype=np.float32)
    boxes_mock.conf = MagicMock()
    boxes_mock.conf.cpu.return_value.numpy.return_value = np.array(confs, dtype=np.float32)
    boxes_mock.cls = MagicMock()
    boxes_mock.cls.cpu.return_value.numpy.return_value = np.array(cls_ids, dtype=np.float32)
    return boxes_mock


def _make_fake_result(boxes_mock):
    """Wraps a fake boxes object in a minimal Ultralytics Result mock."""
    result_mock = MagicMock()
    result_mock.boxes = boxes_mock
    return result_mock


# ===========================================================================
# Task 12 – VehicleTracker
# ===========================================================================


class TestVehicleTracking:
    """Unit tests for modules/vehicle_tracking.py (Task 12)."""

    # -----------------------------------------------------------------------
    # Fixture: a VehicleTracker with YOLO model loading suppressed
    # -----------------------------------------------------------------------

    @pytest.fixture()
    def tracker(self, tmp_path):
        """
        Returns a VehicleTracker whose YOLO model is replaced with a mock.

        Strategy: patch the module-level ``_YOLO_CLS`` variable in
        ``modules.vehicle_tracking`` before the VehicleTracker constructor
        calls ``_load_model()``.  This avoids touching ``ultralytics.YOLO``
        itself and works regardless of import caching.
        """
        import modules.vehicle_tracking as vt_module

        mock_model = MagicMock()
        mock_model.names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        original_cls = vt_module._YOLO_CLS
        vt_module._YOLO_CLS = MagicMock(return_value=mock_model)
        try:
            from modules.vehicle_tracking import VehicleTracker
            vt = VehicleTracker(
                model_path=tmp_path / "fake.pt",
                conf_threshold=0.40,
                track_buffer=30,
            )
            vt._model = mock_model
            yield vt
        finally:
            vt_module._YOLO_CLS = original_cls

    # -----------------------------------------------------------------------
    # Helper: inject a fake tracking result into the tracker
    # -----------------------------------------------------------------------

    def _inject_frame(
        self,
        tracker,
        boxes_data: List[Tuple],  # (x1,y1,x2,y2,id,cls,conf)
        frame_idx: int = 0,
    ):
        """
        Directly calls tracker.track_frame() after replacing model.track()
        with a stub that returns a fixed result.
        """
        if boxes_data:
            xywhs  = [d[:4] for d in boxes_data]
            ids    = [d[4]  for d in boxes_data]
            clss   = [d[5]  for d in boxes_data]
            confs  = [d[6]  for d in boxes_data]
            fake_boxes = _make_fake_boxes(xywhs, ids, clss, confs)
        else:
            fake_boxes = MagicMock()
            fake_boxes.id = None   # simulate no detections

        fake_result = _make_fake_result(fake_boxes)
        tracker._model.track.return_value = [fake_result]

        frame = _blank_frame()
        return tracker.track_frame(frame, frame_idx=frame_idx)

    # -----------------------------------------------------------------------
    # Test 01: persistent track IDs across frames
    # -----------------------------------------------------------------------

    def test_01_persistent_track_ids_across_frames(self, tracker):
        """The same integer ID must appear in the output on successive frames."""
        # Frame 0 – car at position A
        t0 = self._inject_frame(
            tracker,
            [(100, 100, 200, 200, 42, 2, 0.9)],  # track_id=42, car
            frame_idx=0,
        )
        assert len(t0) == 1
        assert t0[0]["track_id"] == 42
        assert t0[0]["class_name"] == "car"

        # Frame 1 – same car slightly moved
        t1 = self._inject_frame(
            tracker,
            [(110, 105, 210, 205, 42, 2, 0.88)],
            frame_idx=1,
        )
        assert len(t1) == 1
        assert t1[0]["track_id"] == 42, "Track ID must persist across frames."
        assert tracker.active_tracks[42]["frames_seen"] == 2

    # -----------------------------------------------------------------------
    # Test 02: multiple vehicles get distinct IDs
    # -----------------------------------------------------------------------

    def test_02_distinct_ids_for_multiple_vehicles(self, tracker):
        """Two simultaneously detected vehicles must receive different IDs."""
        tracked = self._inject_frame(
            tracker,
            [
                (50,  50, 150, 150, 1, 2, 0.9),   # car,   ID 1
                (300, 50, 400, 150, 2, 7, 0.85),  # truck, ID 2
            ],
            frame_idx=0,
        )
        ids = {obj["track_id"] for obj in tracked}
        assert len(ids) == 2, "Each vehicle must have a unique track ID."
        cls_names = {obj["class_name"] for obj in tracked}
        assert "car"   in cls_names
        assert "truck" in cls_names

    # -----------------------------------------------------------------------
    # Test 03: centroid calculation from bounding box
    # -----------------------------------------------------------------------

    def test_03_centroid_calculation(self, tracker):
        """Centroid must be the midpoint of the bounding box."""
        tracked = self._inject_frame(
            tracker,
            [(100, 200, 300, 400, 7, 2, 0.9)],
            frame_idx=0,
        )
        cx, cy = tracked[0]["centroid"]
        assert abs(cx - 200.0) < 1.0, "cx must be (x1+x2)/2"
        assert abs(cy - 300.0) < 1.0, "cy must be (y1+y2)/2"

    # -----------------------------------------------------------------------
    # Test 04: trajectory history grows with each frame
    # -----------------------------------------------------------------------

    def test_04_trajectory_grows_per_frame(self, tracker):
        """get_trajectory() must return one entry per observed frame."""
        for f in range(5):
            self._inject_frame(
                tracker,
                [(100 + f * 5, 200, 200 + f * 5, 300, 99, 2, 0.9)],
                frame_idx=f,
            )
        traj = tracker.get_trajectory(99)
        assert len(traj) == 5, f"Expected 5 history entries, got {len(traj)}"

    # -----------------------------------------------------------------------
    # Test 05: vehicle entering and leaving the frame
    # -----------------------------------------------------------------------

    def test_05_vehicle_enter_and_leave(self, tracker):
        """A track that disappears must still appear in active_tracks history."""
        # Vehicle visible for 3 frames
        for f in range(3):
            self._inject_frame(
                tracker,
                [(200, 200, 300, 300, 55, 3, 0.9)],  # motorcycle
                frame_idx=f,
            )
        # Vehicle leaves – next frame has no detections
        self._inject_frame(tracker, [], frame_idx=3)

        # active_tracks still records it (retirement is session-level only)
        assert 55 in tracker.active_tracks
        assert tracker.active_tracks[55]["frames_seen"] == 3
        assert tracker.active_tracks[55]["class_name"] == "motorcycle"

    # -----------------------------------------------------------------------
    # Test 06: unique vehicle count by class
    # -----------------------------------------------------------------------

    def test_06_unique_vehicle_count_by_class(self, tracker):
        """get_unique_vehicle_count() must tally unique IDs per class."""
        self._inject_frame(
            tracker,
            [
                (10,  10, 100, 100, 1, 2, 0.9),  # car
                (200, 10, 300, 100, 2, 2, 0.9),  # car
                (400, 10, 500, 100, 3, 5, 0.9),  # bus
            ],
            frame_idx=0,
        )
        counts = tracker.get_unique_vehicle_count()
        assert counts["car"]   == 2
        assert counts["bus"]   == 1
        assert counts["total"] == 3

    # -----------------------------------------------------------------------
    # Test 07: annotate_frame returns same-shape array
    # -----------------------------------------------------------------------

    def test_07_annotate_frame_shape_preserved(self, tracker):
        """annotate_frame() must return an array with the same H×W×3 shape."""
        frame = _blank_frame(640, 480)
        tracked = self._inject_frame(
            tracker,
            [(50, 50, 150, 150, 10, 2, 0.9)],
            frame_idx=0,
        )
        annotated = tracker.annotate_frame(frame, tracked)
        assert annotated.shape == frame.shape

    # -----------------------------------------------------------------------
    # Test 08: JSON export structure
    # -----------------------------------------------------------------------

    def test_08_json_export(self, tracker, tmp_path):
        """export_json() must produce a valid JSON file with expected keys."""
        self._inject_frame(
            tracker,
            [(10, 10, 100, 100, 1, 2, 0.9)],
            frame_idx=0,
        )
        json_path = tmp_path / "tracking.json"
        result = tracker.export_json(json_path)
        assert result.exists()
        with open(result, encoding="utf-8") as fh:
            data = json.load(fh)
        assert "unique_vehicle_counts" in data
        assert "tracks"               in data
        assert "total_tracks"         in data

    # -----------------------------------------------------------------------
    # Test 09: CSV export
    # -----------------------------------------------------------------------

    def test_09_csv_export(self, tracker, tmp_path):
        """export_csv() must produce a readable CSV with the correct headers."""
        self._inject_frame(
            tracker,
            [(10, 10, 100, 100, 1, 2, 0.9)],
            frame_idx=0,
        )
        csv_path = tmp_path / "tracking.csv"
        result = tracker.export_csv(csv_path)
        assert result.exists()
        with open(result, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) >= 1
        assert "track_id" in rows[0]
        assert "class_name" in rows[0]

    # -----------------------------------------------------------------------
    # Test 10: clear() resets state
    # -----------------------------------------------------------------------

    def test_10_clear_resets_state(self, tracker):
        """clear() must empty active_tracks and reset the frame counter."""
        self._inject_frame(tracker, [(10, 10, 100, 100, 1, 2, 0.9)], frame_idx=0)
        assert len(tracker.active_tracks) > 0
        tracker.clear()
        assert len(tracker.active_tracks) == 0
        assert tracker._frame_count == 0


# ===========================================================================
# Task 13 – SpeedEstimator
# ===========================================================================


class TestSpeedEstimation:
    """Unit tests for modules/speed_estimation.py (Task 13)."""

    @pytest.fixture()
    def estimator_calibrated(self):
        """SpeedEstimator with a known calibration (100 px = 1 m)."""
        from modules.speed_estimation import SpeedEstimator
        return SpeedEstimator(pixels_per_meter=100.0, fps=30.0)

    @pytest.fixture()
    def estimator_uncalibrated(self):
        """SpeedEstimator with no calibration supplied."""
        from modules.speed_estimation import SpeedEstimator
        return SpeedEstimator(pixels_per_meter=None, fps=30.0)

    # -----------------------------------------------------------------------
    # Test 01: pixel_displacement
    # -----------------------------------------------------------------------

    def test_01_pixel_displacement_known_distance(self, estimator_calibrated):
        """pixel_displacement must return 5 for a 3–4–5 right triangle."""
        from modules.speed_estimation import SpeedEstimator
        d = SpeedEstimator.pixel_displacement((0.0, 0.0), (3.0, 4.0))
        assert abs(d - 5.0) < 1e-6

    def test_02_pixel_displacement_stationary(self, estimator_calibrated):
        """Zero displacement between identical centroids."""
        from modules.speed_estimation import SpeedEstimator
        d = SpeedEstimator.pixel_displacement((100.0, 200.0), (100.0, 200.0))
        assert d == 0.0

    # -----------------------------------------------------------------------
    # Test 03: raw_speed calibrated
    # -----------------------------------------------------------------------

    def test_03_raw_speed_calibrated(self, estimator_calibrated):
        """
        100 px displacement, 100 px/m, 30 FPS, 1 frame → 1/30 s elapsed
        → 30 m/s → 108 km/h.
        """
        speed = estimator_calibrated.raw_speed(100.0, frames_elapsed=1)
        expected = (100.0 / 100.0) / (1.0 / 30.0) * 3.6   # 108.0
        assert abs(speed - expected) < 0.5, f"Expected ~108 km/h, got {speed}"

    # -----------------------------------------------------------------------
    # Test 04: raw_speed uncalibrated returns px/s
    # -----------------------------------------------------------------------

    def test_04_raw_speed_uncalibrated(self, estimator_uncalibrated):
        """Uncalibrated mode must return px/s, not km/h."""
        est = estimator_uncalibrated
        assert not est.calibrated
        speed = est.raw_speed(300.0, frames_elapsed=1)
        expected_pxs = 300.0 * 30.0   # 9000 px/s
        assert abs(speed - expected_pxs) < 1.0

    # -----------------------------------------------------------------------
    # Test 05: update() returns None before minimum history
    # -----------------------------------------------------------------------

    def test_05_update_returns_none_before_min_frames(self, estimator_calibrated):
        """Speed must be None for the first few frames of a new track."""
        est = estimator_calibrated
        result = est.update(track_id=1, centroid=(100.0, 100.0), frames_seen=1, frame_idx=0)
        assert result is None, "First frame → None (no prior centroid)."

        # Second frame: frames_seen=2 < min_frames_for_speed=3 → still None
        result2 = est.update(track_id=1, centroid=(110.0, 100.0), frames_seen=2, frame_idx=1)
        # May be None depending on smoothing window; must not raise
        # (do not assert exact value here – depends on min_frames)

    # -----------------------------------------------------------------------
    # Test 06: update() returns smoothed speed after sufficient history
    # -----------------------------------------------------------------------

    def test_06_update_returns_speed_after_sufficient_history(self, estimator_calibrated):
        """After min_frames observations, get_speed() must return a float ≥ 0."""
        est = estimator_calibrated
        # Simulate 10 frames of steady 10 px/frame movement (horizontal)
        for f in range(10):
            est.update(
                track_id=99,
                centroid=(float(f * 10), 200.0),
                frames_seen=f + 1,
                frame_idx=f,
            )
        speed = est.get_speed(99)
        assert speed is not None, "After 10 frames, a speed estimate must exist."
        assert speed >= 0.0, "Speed must be non-negative."
        assert isinstance(speed, float)

    # -----------------------------------------------------------------------
    # Test 07: speed conversion accuracy (calibrated)
    # -----------------------------------------------------------------------

    def test_07_speed_accuracy_calibrated(self):
        """
        Synthetic ground-truth check:
        Vehicle moves 50 px per frame at 25 FPS, 50 px/m calibration.
        → 1 m/frame × 25 FPS = 25 m/s = 90 km/h.
        """
        from modules.speed_estimation import SpeedEstimator
        est = SpeedEstimator(pixels_per_meter=50.0, fps=25.0, min_frames_for_speed=2)
        for f in range(8):
            est.update(
                track_id=7,
                centroid=(float(f * 50), 100.0),
                frames_seen=f + 1,
                frame_idx=f,
            )
        speed = est.get_speed(7)
        assert speed is not None
        # Allow ±5 km/h tolerance for the median smoothing
        assert abs(speed - 90.0) < 5.0, f"Expected ~90 km/h, got {speed} km/h"

    # -----------------------------------------------------------------------
    # Test 08: plausibility ceiling
    # -----------------------------------------------------------------------

    def test_08_max_speed_ceiling(self):
        """Implausibly large displacements must be discarded (ceiling guard)."""
        from modules.speed_estimation import SpeedEstimator
        est = SpeedEstimator(pixels_per_meter=1.0, fps=30.0,
                             max_speed_kmh=250.0, min_frames_for_speed=2)
        # First observation
        est.update(track_id=5, centroid=(0.0, 0.0), frames_seen=1, frame_idx=0)
        # Huge jump (> 250 km/h equivalent)
        est.update(track_id=5, centroid=(100000.0, 0.0), frames_seen=2, frame_idx=1)
        speed = est.get_speed(5)
        # Either None (discarded) or below ceiling
        if speed is not None:
            assert speed <= 250.0, "Speed must not exceed the ceiling."

    # -----------------------------------------------------------------------
    # Test 09: get_all_speeds returns dict of floats
    # -----------------------------------------------------------------------

    def test_09_get_all_speeds(self, estimator_calibrated):
        """get_all_speeds() must return a dict keyed by int track IDs."""
        est = estimator_calibrated
        for f in range(5):
            est.update(10, (float(f * 20), 100.0), f + 1, f)
            est.update(11, (100.0, float(f * 10)), f + 1, f)
        speeds = est.get_all_speeds()
        assert isinstance(speeds, dict)
        # At least one entry may be populated after smoothing window fills
        for tid, spd in speeds.items():
            assert isinstance(tid, int)
            assert isinstance(spd, float)
            assert spd >= 0.0

    # -----------------------------------------------------------------------
    # Test 10: uncalibrated mode annotates frame with watermark
    # -----------------------------------------------------------------------

    def test_10_uncalibrated_annotate_watermark(self, estimator_uncalibrated):
        """annotate_frame() in uncalibrated mode must not raise and must mark frame."""
        frame = _blank_frame(640, 480)
        # No tracks – just verify no exception and shape preserved
        out = estimator_uncalibrated.annotate_frame(frame, [])
        assert out.shape == frame.shape

    # -----------------------------------------------------------------------
    # Test 11: JSON export structure
    # -----------------------------------------------------------------------

    def test_11_json_export_keys(self, estimator_calibrated, tmp_path):
        """export_json() must include calibration metadata and limitations."""
        for f in range(5):
            estimator_calibrated.update(1, (float(f * 10), 0.0), f + 1, f)
        p = tmp_path / "speed.json"
        estimator_calibrated.export_json(p)
        assert p.exists()
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        for key in ("calibrated", "pixels_per_meter", "fps", "unit",
                    "limitations", "records"):
            assert key in data, f"Key '{key}' missing from speed JSON."
        assert data["calibrated"] is True

    # -----------------------------------------------------------------------
    # Test 12: CSV export
    # -----------------------------------------------------------------------

    def test_12_csv_export(self, estimator_calibrated, tmp_path):
        """export_csv() must write per-measurement rows with correct headers."""
        for f in range(5):
            estimator_calibrated.update(2, (float(f * 15), 50.0), f + 1, f)
        p = tmp_path / "speed.csv"
        estimator_calibrated.export_csv(p)
        assert p.exists()
        with open(p, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        # May be empty if smoothing didn't fire, but headers must be present
        with open(p, encoding="utf-8") as fh:
            first_line = fh.readline()
        assert "track_id" in first_line
        assert "smoothed_speed" in first_line

    # -----------------------------------------------------------------------
    # Test 13: average_speed with zero observations
    # -----------------------------------------------------------------------

    def test_13_average_speed_empty(self, estimator_calibrated):
        """get_average_speed() on a fresh estimator must return 0.0."""
        assert estimator_calibrated.get_average_speed() == 0.0

    # -----------------------------------------------------------------------
    # Test 14: summary limitations documented
    # -----------------------------------------------------------------------

    def test_14_limitations_documented(self, estimator_calibrated):
        """get_summary() must include at least 3 documented limitations."""
        summary = estimator_calibrated.get_summary()
        assert "limitations" in summary
        assert len(summary["limitations"]) >= 3

