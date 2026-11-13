"""
Module: test_traffic_analytics.py
Phase 3 Test Suite:
- Task 9: Vehicle counting and category breakdown.
- Task 10: Traffic density classification and road ROI.
- Task 11: Congestion detection, temporal persistence, and alerts.
- Integrated Traffic Analytics pipeline, JSON and CSV export verification.
"""

import sys
import unittest
import tempfile
import json
import csv
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import get_project_root, load_config, get_path
from modules.vehicle_counting import VehicleCounter
from modules.traffic_density import TrafficDensityCalculator
from modules.congestion_detection import CongestionDetector
from modules.traffic_analytics import TrafficAnalyticsPipeline


class TestTrafficAnalyticsPhase3(unittest.TestCase):
    """Unit and integration tests for Phase 3 Traffic Analytics on Windows."""

    @classmethod
    def setUpClass(cls):
        cls.root = get_project_root()
        cls.config = load_config()
        cls.images_dir = get_path("dataset_images_dir", cls.config)
        cls.videos_dir = get_path("dataset_videos_dir", cls.config)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_vehicle_counter_tally_and_categories(self):
        """Task 9: Verifies vehicle counting by category and total calculations."""
        counter = VehicleCounter()
        sample_detections = [
            {"bbox": [100, 100, 200, 200], "class_name": "car", "confidence": 0.85},
            {"bbox": [250, 150, 320, 210], "class_name": "car", "confidence": 0.72},
            {"bbox": [350, 200, 400, 260], "class_name": "motorcycle", "confidence": 0.65},
            {"bbox": [450, 100, 600, 350], "class_name": "bus", "confidence": 0.91},
            {"bbox": [650, 120, 800, 320], "class_name": "truck", "confidence": 0.78},
        ]

        metrics = counter.count_frame(sample_detections, frame_idx=0)
        self.assertEqual(metrics["total_vehicles"], 5)
        self.assertEqual(metrics["counts"]["car"], 2)
        self.assertEqual(metrics["counts"]["motorcycle"], 1)
        self.assertEqual(metrics["counts"]["bus"], 1)
        self.assertEqual(metrics["counts"]["truck"], 1)

    def test_02_vehicle_counter_roi_filtering(self):
        """Task 9: Verifies that vehicles outside the defined ROI are excluded."""
        # Rectangular ROI in center: x in [200, 600], y in [200, 600]
        counter = VehicleCounter(roi_bbox=(200, 200, 600, 600))
        sample_detections = [
            {"bbox": [300, 300, 350, 350], "class_name": "car", "confidence": 0.8},   # Inside
            {"bbox": [50, 50, 90, 90], "class_name": "car", "confidence": 0.9},        # Outside
            {"bbox": [400, 400, 550, 500], "class_name": "truck", "confidence": 0.7}, # Inside
        ]

        metrics = counter.count_frame(sample_detections, frame_idx=0)
        self.assertEqual(metrics["total_vehicles"], 2)
        self.assertEqual(metrics["counts"]["car"], 1)
        self.assertEqual(metrics["counts"]["truck"], 1)

    def test_03_vehicle_counter_json_csv_export(self):
        """Task 9: Verifies exporting counts to JSON and CSV formats."""
        counter = VehicleCounter()
        counter.count_frame([{"bbox": [10, 10, 50, 50], "class_name": "car", "confidence": 0.8}], frame_idx=0)
        counter.count_frame([{"bbox": [20, 20, 60, 60], "class_name": "bus", "confidence": 0.9}], frame_idx=1)

        json_file = self.temp_path / "counts.json"
        csv_file = self.temp_path / "counts.csv"

        counter.export_json(json_file)
        counter.export_csv(csv_file)

        self.assertTrue(json_file.exists())
        self.assertTrue(csv_file.exists())

        # Validate JSON content
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("summary", data)
            self.assertEqual(len(data["frame_history"]), 2)

        # Validate CSV content
        with open(csv_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["car"], "1")
            self.assertEqual(rows[1]["bus"], "1")

    def test_04_traffic_density_classification_tiers(self):
        """Task 10: Verifies LOW, MEDIUM, and HIGH classification and transitions."""
        calc = TrafficDensityCalculator(low_threshold=2, medium_threshold=5, high_threshold=10)

        # 1 vehicle -> LOW
        low_res = calc.calculate_density([{"bbox": [0, 0, 10, 10]}], frame_idx=0)
        self.assertEqual(low_res["density_level"], "LOW")

        # 4 vehicles -> MEDIUM
        med_res = calc.calculate_density([{"bbox": [0, 0, 10, 10]}] * 4, frame_idx=1)
        self.assertEqual(med_res["density_level"], "MEDIUM")

        # 8 vehicles -> HIGH
        high_res = calc.calculate_density([{"bbox": [0, 0, 10, 10]}] * 8, frame_idx=2)
        self.assertEqual(high_res["density_level"], "HIGH")

        # Verify state transition log was recorded
        self.assertGreaterEqual(len(calc.transitions), 2)

    def test_05_traffic_density_limitations_and_export(self):
        """Task 10: Verifies documented limitations and export."""
        calc = TrafficDensityCalculator()
        limitations = calc.get_limitations()
        self.assertIsInstance(limitations, list)
        self.assertGreaterEqual(len(limitations), 4)

        json_out = self.temp_path / "density.json"
        csv_out = self.temp_path / "density.csv"
        calc.calculate_density([{"bbox": [10, 10, 50, 50]}], frame_idx=0)
        calc.export_json(json_out)
        calc.export_csv(csv_out)

        self.assertTrue(json_out.exists())
        self.assertTrue(csv_out.exists())

    def test_06_congestion_detector_temporal_persistence(self):
        """Task 11: Verifies that single temporary spikes DO NOT trigger congestion."""
        # Threshold: 3 consecutive HIGH frames required
        detector = CongestionDetector(consecutive_frames_threshold=3, recovery_frames_threshold=2, fps=30.0)

        # Frame 0: HIGH (single spike)
        eval0 = detector.evaluate_frame({"density_level": "HIGH", "vehicle_count": 25}, frame_idx=0)
        self.assertFalse(eval0["is_congested"], "Single frame must not trigger congestion.")

        # Frame 1: LOW (spike subsides)
        eval1 = detector.evaluate_frame({"density_level": "LOW", "vehicle_count": 2}, frame_idx=1)
        self.assertFalse(eval1["is_congested"])
        self.assertEqual(len(detector.events), 0)

        # Now simulate sustained HIGH for 3 consecutive frames
        detector.evaluate_frame({"density_level": "HIGH", "vehicle_count": 20}, frame_idx=2)
        detector.evaluate_frame({"density_level": "HIGH", "vehicle_count": 22}, frame_idx=3)
        eval4 = detector.evaluate_frame({"density_level": "HIGH", "vehicle_count": 24}, frame_idx=4)

        self.assertTrue(eval4["is_congested"], "Sustained high density must trigger congestion.")
        self.assertTrue(eval4["alert_generated"])

        # Simulate recovery
        detector.evaluate_frame({"density_level": "LOW", "vehicle_count": 3}, frame_idx=5)
        eval6 = detector.evaluate_frame({"density_level": "LOW", "vehicle_count": 2}, frame_idx=6)
        self.assertFalse(eval6["is_congested"], "Traffic must recover after recovery threshold.")
        self.assertEqual(len(detector.events), 1)
        self.assertEqual(detector.events[0]["status"], "RESOLVED")
        self.assertGreater(detector.events[0]["duration_sec"], 0.0)

    def test_07_integrated_analytics_pipeline(self):
        """Verifies end-to-end integration with real YOLO inference on a real traffic image."""
        valid_images = [
            p for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        self.assertGreater(len(valid_images), 0)

        import cv2
        test_img = cv2.imread(str(valid_images[0]))
        self.assertIsNotNone(test_img)

        pipeline = TrafficAnalyticsPipeline(conf_threshold=0.30, device="cpu")
        annotated_frame, metrics = pipeline.process_frame(test_img, frame_idx=0)

        self.assertEqual(annotated_frame.shape, test_img.shape)
        self.assertIn("counted_vehicles", metrics)
        self.assertIn("counts_by_class", metrics)
        self.assertIn("density_level", metrics)
        self.assertIn("is_congested", metrics)


if __name__ == "__main__":
    unittest.main()

