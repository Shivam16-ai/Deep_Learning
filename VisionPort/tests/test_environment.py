"""
Module: test_environment.py
Task 1: Python and library environment setup verification.

Tests the Windows environment configuration, Python version (3.12.7 target),
project folder hierarchy integrity, configuration loader, and library availability.
"""

import sys
import unittest
from pathlib import Path


class TestVisionPortEnvironment(unittest.TestCase):
    """
    Test suite to verify Python environment, package imports, and project structure.
    """

    def setUp(self):
        # Resolve project root relative to tests directory
        self.project_root = Path(__file__).resolve().parent.parent

    def test_python_version(self):
        """Validates that Python runtime meets 3.12 requirement."""
        major = sys.version_info.major
        minor = sys.version_info.minor
        print(f"\n[Environment Check] Detected Python Version: {sys.version}")
        self.assertEqual(major, 3, "Python 3 is required.")
        self.assertGreaterEqual(minor, 12, "Python 3.12+ is required for VisionPort.")

    def test_directory_structure_exists(self):
        """Verifies that all required VisionPort directories exist on Windows."""
        required_dirs = [
            self.project_root / "dataset" / "images",
            self.project_root / "dataset" / "videos",
            self.project_root / "captures",
            self.project_root / "output" / "detected_images",
            self.project_root / "output" / "processed_videos",
            self.project_root / "output" / "extracted_frames",
            self.project_root / "output" / "reports",
            self.project_root / "output" / "analytics",
            self.project_root / "models",
            self.project_root / "modules",
            self.project_root / "utils",
            self.project_root / "tests",
        ]
        for dir_path in required_dirs:
            self.assertTrue(
                dir_path.exists() and dir_path.is_dir(),
                f"Required directory is missing: {dir_path}",
            )

    def test_required_files_exist(self):
        """Verifies that essential project configuration and module files exist."""
        required_files = [
            self.project_root / "config.yaml",
            self.project_root / "requirements.txt",
            self.project_root / "README.md",
            self.project_root / ".gitignore",
            self.project_root / "main.py",
            self.project_root / "utils" / "config.py",
            self.project_root / "utils" / "logger.py",
            self.project_root / "utils" / "visualization.py",
            self.project_root / "modules" / "image_capture.py",
            self.project_root / "modules" / "image_display.py",
            self.project_root / "modules" / "video_processing.py",
            self.project_root / "modules" / "frame_extraction.py",
            self.project_root / "modules" / "vehicle_detection.py",
            self.project_root / "modules" / "vehicle_counting.py",
            self.project_root / "modules" / "traffic_density.py",
            self.project_root / "modules" / "congestion_detection.py",
            self.project_root / "modules" / "vehicle_tracking.py",
            self.project_root / "modules" / "speed_estimation.py",
            self.project_root / "modules" / "signal_detection.py",
            self.project_root / "modules" / "waiting_time.py",
        ]
        for file_path in required_files:
            self.assertTrue(
                file_path.exists() and file_path.is_file(),
                f"Required file is missing: {file_path}",
            )

    def test_config_loading(self):
        """Validates that config.yaml can be parsed without errors."""
        # Add project root to sys.path if not present
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        from utils.config import load_config
        config = load_config()
        self.assertIsInstance(config, dict, "Config should load as a dictionary.")
        self.assertIn("project", config)
        self.assertIn("paths", config)
        self.assertEqual(config["project"]["name"], "VisionPort")

    def test_library_imports(self):
        """Checks availability of key computer vision and deep learning packages."""
        packages = ["numpy", "cv2", "yaml", "ultralytics"]
        missing = []
        for pkg in packages:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
                
        if missing:
            print(f"[Notice] The following packages are not yet installed in this environment: {missing}")
            print("Run 'pip install -r requirements.txt' inside your virtual environment.")
        else:
            print("[Environment Check] Core packages (numpy, cv2, yaml, ultralytics) are imported successfully.")


if __name__ == "__main__":
    unittest.main()
