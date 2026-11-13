"""
Module: test_vehicle_detection.py
Tasks 6, 7, 8:
- Task 6: Car detection using YOLO.
- Task 7: Bike (motorcycle), bus, and truck detection.
- Task 8: Vehicle labels, bounding boxes, and confidence scores.

Integration and unit tests verifying YOLOv8 vehicle detection pipeline on Windows.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import get_project_root, load_config, get_path
from modules.vehicle_detection import VehicleDetector


class TestVehicleDetection(unittest.TestCase):
    """Test suite for YOLOv8 vehicle detection tasks (Tasks 6, 7, and 8)."""

    @classmethod
    def setUpClass(cls):
        """Initializes detector and locates test images."""
        cls.root = get_project_root()
        cls.config = load_config()
        cls.images_dir = get_path("dataset_images_dir", cls.config)
        
        # Initialize detector on CPU for test environment reproducibility
        cls.detector = VehicleDetector(conf_threshold=0.30, device="cpu")

        # Find available real traffic images
        cls.valid_images = [
            p for p in cls.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]

    def test_01_model_loaded_and_classes_mapped(self):
        """Verifies model initialization and COCO class ID mapping."""
        self.assertIsNotNone(self.detector.model, "YOLO model should be loaded.")
        self.assertEqual(self.detector.TARGET_CLASSES["car"], 2)
        self.assertEqual(self.detector.TARGET_CLASSES["motorcycle"], 3)
        self.assertEqual(self.detector.TARGET_CLASSES["bus"], 5)
        self.assertEqual(self.detector.TARGET_CLASSES["truck"], 7)

    def test_02_detect_cars_exclusively(self):
        """Task 6: Verifies that detect_cars only yields car detections (class_id=2)."""
        self.assertGreater(len(self.valid_images), 0, "At least one traffic image must exist.")
        import cv2

        test_img_path = self.valid_images[0]
        frame = cv2.imread(str(test_img_path))
        self.assertIsNotNone(frame, f"Failed to read {test_img_path}")

        cars = self.detector.detect_cars(frame, conf=0.25)
        self.assertIsInstance(cars, list)
        for det in cars:
            self.assertEqual(det["class_id"], 2, "Only car class ID (2) should be returned.")
            self.assertEqual(det["class_name"], "car")
            self.assertIn("bbox", det)
            self.assertEqual(len(det["bbox"]), 4)
            self.assertIn("confidence", det)
            self.assertGreaterEqual(det["confidence"], 0.25)

    def test_03_detect_multi_class(self):
        """Task 7 & 8: Verifies multi-class vehicle detection, bounding boxes and confidences."""
        import cv2

        test_img_path = self.valid_images[0]
        frame = cv2.imread(str(test_img_path))

        detections = self.detector.detect_multi_class(frame, conf=0.25)
        self.assertIsInstance(detections, list)

        allowed_ids = {2, 3, 5, 7}
        for det in detections:
            self.assertIn(det["class_id"], allowed_ids)
            self.assertIn(det["class_name"], {"car", "motorcycle", "bus", "truck"})
            self.assertGreaterEqual(det["confidence"], 0.25)
            x1, y1, x2, y2 = det["bbox"]
            self.assertLessEqual(x1, x2)
            self.assertLessEqual(y1, y2)

    def test_04_confidence_filtering(self):
        """Verifies that detections strictly respect the confidence cutoff."""
        import cv2

        test_img_path = self.valid_images[0]
        frame = cv2.imread(str(test_img_path))

        low_conf_dets = self.detector.detect_multi_class(frame, conf=0.20)
        high_conf_dets = self.detector.detect_multi_class(frame, conf=0.60)

        # Higher threshold must return equal or fewer detections
        self.assertLessEqual(len(high_conf_dets), len(low_conf_dets))
        for d in high_conf_dets:
            self.assertGreaterEqual(d["confidence"], 0.60)

    def test_05_annotate_frame(self):
        """Task 8: Verifies bounding box and label rendering on image."""
        import cv2

        test_img_path = self.valid_images[0]
        frame = cv2.imread(str(test_img_path))
        detections = self.detector.detect_multi_class(frame, conf=0.30)

        annotated = self.detector.annotate_frame(frame, detections)
        self.assertEqual(annotated.shape, frame.shape)
        self.assertIsInstance(annotated, np.ndarray)

    def test_06_process_single_image_end_to_end(self):
        """Verifies end-to-end processing of a real traffic photograph."""
        test_img_path = self.valid_images[0]
        result = self.detector.process_image(test_img_path, conf=0.30)

        self.assertIn("image_name", result)
        self.assertIn("total_vehicles", result)
        self.assertIn("counts", result)
        self.assertIn("output_path", result)
        self.assertTrue(Path(result["output_path"]).exists())
        self.assertGreater(Path(result["output_path"]).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

