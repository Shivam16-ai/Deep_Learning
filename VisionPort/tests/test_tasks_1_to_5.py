"""
Module: test_tasks_1_to_5.py
Integration and unit test suite for Tasks 1 through 5:
- Task 1: Python and library environment setup
- Task 2: Traffic image capture
- Task 3: Image reading, inspection, and display
- Task 4: Traffic video playback and iteration
- Task 5: Video frame extraction
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config, get_project_root
from utils.sample_data import generate_sample_assets
from modules.image_capture import capture_traffic_image
from modules.image_display import load_and_inspect_image, convert_color_space, display_image
from modules.video_processing import VideoProcessor, play_traffic_video
from modules.frame_extraction import extract_frames_from_video


class TestFoundationalTasks1To5(unittest.TestCase):
    """Verifies Tasks 1 through 5 end-to-end on Windows."""

    @classmethod
    def setUpClass(cls):
        """Prepare sample traffic image and video assets."""
        cls.root = get_project_root()
        cls.sample_image, cls.sample_video = generate_sample_assets(force=False)

    def test_task_01_environment_and_config(self):
        """Task 1: Verify environment, YAML config parsing, and directories."""
        config = load_config()
        self.assertIn("project", config)
        self.assertEqual(config["project"]["name"], "VisionPort")
        self.assertTrue((self.root / "dataset" / "images").exists())
        self.assertTrue((self.root / "dataset" / "videos").exists())

    def test_task_02_image_capture(self):
        """Task 2: Verify traffic image capture and saving to captures/."""
        output_dir = self.root / "captures"
        saved_path = capture_traffic_image(
            device_index=0,
            output_dir=output_dir,
            filename_prefix="test_capture",
            simulate_if_no_camera=True,
            skip_hardware_probe=True,
        )
        self.assertIsNotNone(saved_path, "Image capture should return valid path.")
        self.assertTrue(saved_path.exists(), f"Captured image does not exist: {saved_path}")
        self.assertGreater(saved_path.stat().st_size, 1000, "Image file should contain non-trivial data.")

    def test_task_03_image_read_and_display(self):
        """Task 3: Verify image loading, metadata inspection, color conversion, and headless display."""
        img, meta = load_and_inspect_image(self.sample_image)
        self.assertIsNotNone(img, "Failed to load image.")
        self.assertIsNotNone(meta, "Metadata inspection failed.")
        self.assertEqual(meta["width"], 1280)
        self.assertEqual(meta["height"], 720)
        self.assertEqual(meta["channels"], 3)
        self.assertGreater(meta["mean_intensity"], 0.0)

        # Test color space conversion
        gray = convert_color_space(img, target="GRAY")
        self.assertEqual(len(gray.shape), 2, "Grayscale image must have 2 dimensions.")

        rgb = convert_color_space(img, target="RGB")
        self.assertEqual(rgb.shape, img.shape, "RGB image shape must match input.")

        # Test display routine (headless mode for test automation)
        preview_out = self.root / "output" / "detected_images" / "test_inspection_preview.jpg"
        success = display_image(img, headless=True, save_copy_to=preview_out)
        self.assertTrue(success, "Image display/processing failed.")
        self.assertTrue(preview_out.exists(), "Inspection preview file was not saved.")

    def test_task_04_video_playback(self):
        """Task 4: Verify video metadata extraction, frame iteration, and playback."""
        processor = VideoProcessor(self.sample_video)
        self.assertEqual(processor.width, 1280)
        self.assertEqual(processor.height, 720)
        self.assertEqual(processor.fps, 30.0)
        self.assertGreaterEqual(processor.frame_count, 60)

        # Iterate 10 frames
        frames = list(processor.iter_frames(max_frames=10))
        self.assertEqual(len(frames), 10)
        self.assertEqual(frames[0][1].shape, (720, 1280, 3))

        # Test playback in headless mode
        playback_ok = play_traffic_video(self.sample_video, max_frames=15, headless=True)
        self.assertTrue(playback_ok, "Video playback returned failure.")

    def test_task_05_frame_extraction(self):
        """Task 5: Verify frame extraction at regular step intervals."""
        extract_dir = self.root / "output" / "extracted_frames"
        extracted = extract_frames_from_video(
            video_path=self.sample_video,
            output_dir=extract_dir,
            frame_interval=15,  # Extract every 15th frame (4 frames out of 60)
            max_frames=4,
        )
        self.assertEqual(len(extracted), 4, f"Expected 4 extracted frames, got {len(extracted)}")
        for frame_file in extracted:
            self.assertTrue(frame_file.exists(), f"Extracted frame missing: {frame_file}")
            self.assertGreater(frame_file.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
