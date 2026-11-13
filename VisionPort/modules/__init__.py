"""
VisionPort Core Processing Modules.
Contains modular components mapped directly to the 15 project assignments:
- image_capture: Task 2
- image_display: Task 3
- video_processing: Task 4
- frame_extraction: Task 5
- vehicle_detection: Tasks 6, 7, 8
- vehicle_counting: Task 9
- traffic_density: Task 10
- congestion_detection: Task 11
- vehicle_tracking: Task 12
- speed_estimation: Task 13
- signal_detection: Task 14
- waiting_time: Task 15
"""

from .image_capture import capture_traffic_image
from .image_display import display_image, load_and_inspect_image
from .video_processing import play_traffic_video, VideoProcessor
from .frame_extraction import extract_frames_from_video
from .vehicle_detection import VehicleDetector
from .vehicle_counting import VehicleCounter
from .traffic_density import TrafficDensityCalculator
from .congestion_detection import CongestionDetector
from .vehicle_tracking import VehicleTracker
from .speed_estimation import SpeedEstimator
from .signal_detection import TrafficSignalDetector
from .waiting_time import WaitingTimeEstimator

__all__ = [
    "capture_traffic_image",
    "display_image",
    "load_and_inspect_image",
    "play_traffic_video",
    "VideoProcessor",
    "extract_frames_from_video",
    "VehicleDetector",
    "VehicleCounter",
    "TrafficDensityCalculator",
    "CongestionDetector",
    "VehicleTracker",
    "SpeedEstimator",
    "TrafficSignalDetector",
    "WaitingTimeEstimator",
]
