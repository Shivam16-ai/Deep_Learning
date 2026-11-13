"""
VisionPort – Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning
Main Entry Point (CLI)

Usage on Windows:
    python main.py --help
    python main.py --check-env
    python main.py --list-tasks
    python main.py --capture-image
    python main.py --display-image --path "dataset/images/sample.jpg"
    python main.py --play-video --path "dataset/videos/sample_traffic.mp4"
    python main.py --extract-frames --path "dataset/videos/sample_traffic.mp4"
    python main.py --detect-cars --path "dataset/images/photo-1589828155685-83225f7d91f3.webp"
    python main.py --detect-vehicles --path "dataset/images/aerial-view-of-bengaluru-city-traffic-near-town-hall-bengaluru-india.webp"
    python main.py --detect-all
    python main.py --detect-video --path "dataset/videos/sample_traffic.mp4"
    python main.py --count-vehicles --path "dataset/images/aerial-view-of-bengaluru-city-traffic-near-town-hall-bengaluru-india.webp"
    python main.py --traffic-density --path "dataset/images/aerial-view-of-bengaluru-city-traffic-near-town-hall-bengaluru-india.webp"
    python main.py --detect-congestion --path "dataset/images/aerial-view-of-bengaluru-city-traffic-near-town-hall-bengaluru-india.webp"
    python main.py --analyze-traffic --path "dataset/videos/sample_traffic.mp4"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path for direct script invocation
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config, get_path
from utils.logger import setup_logger

logger = setup_logger("VisionPort_Main")

TASK_MAPPING = {
    1: ("Python and library environment setup", "tests/test_environment.py"),
    2: ("Traffic image capture", "modules/image_capture.py"),
    3: ("Image reading and displaying using OpenCV", "modules/image_display.py"),
    4: ("Traffic video playback", "modules/video_processing.py"),
    5: ("Video frame extraction", "modules/frame_extraction.py"),
    6: ("Car detection using YOLO", "modules/vehicle_detection.py (Implemented)"),
    7: ("Bike, bus and truck detection", "modules/vehicle_detection.py (Implemented)"),
    8: ("Vehicle labels and confidence scores", "modules/vehicle_detection.py & utils/visualization.py"),
    9: ("Vehicle counting", "modules/vehicle_counting.py (Implemented)"),
    10: ("Traffic density calculation", "modules/traffic_density.py (Implemented)"),
    11: ("Traffic congestion detection", "modules/congestion_detection.py (Implemented)"),
    12: ("Individual vehicle tracking", "modules/vehicle_tracking.py (ByteTrack – Implemented)"),
    13: ("Approximate vehicle speed estimation", "modules/speed_estimation.py (Implemented)"),
    14: ("Traffic signal status detection", "modules/signal_detection.py"),
    15: ("Traffic waiting-time estimation", "modules/waiting_time.py"),
}


def list_tasks() -> None:
    """Prints the project 15-task mapping table."""
    print("=" * 80)
    print(" VISIONPORT - 15 TASK ARCHITECTURE MAPPING")
    print("=" * 80)
    for task_num, (desc, module_path) in TASK_MAPPING.items():
        print(f" Task {task_num:02d}: {desc:<42} -> {module_path}")
    print("=" * 80)


def check_environment() -> None:
    """Runs the environment and project structure test suite."""
    logger.info("Running VisionPort environment and configuration audit...")
    import unittest
    from tests.test_environment import TestVisionPortEnvironment

    suite = unittest.TestLoader().loadTestsFromTestCase(TestVisionPortEnvironment)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        logger.info("All environment and structure checks PASSED successfully.")
    else:
        logger.error("Environment checks encountered failures.")


def run_detect_cars(image_path: str, conf: float, device: str) -> None:
    """Task 6 CLI runner: Detects cars in an image."""
    from modules.vehicle_detection import VehicleDetector
    import cv2

    detector = VehicleDetector(conf_threshold=conf, device=device)
    src_file = Path(image_path).resolve()
    frame = cv2.imread(str(src_file))
    if frame is None:
        logger.error("Failed to load image from %s", src_file)
        return

    cars = detector.detect_cars(frame, conf=conf)
    print(f"\n[Task 6: Car Detection Results for {src_file.name}]")
    print(f"Total cars detected: {len(cars)}")
    for i, car in enumerate(cars, 1):
        print(f"  Car #{i:02d}: Box={car['bbox']} | Confidence={car['confidence']:.4f}")

    annotated = detector.annotate_frame(frame, cars)
    cfg = load_config()
    out_dir = get_path("detected_images_dir", cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"cars_{src_file.stem}.jpg"
    cv2.imwrite(str(out_file), annotated)
    print(f"Annotated car detection image saved to: {out_file}\n")


def run_detect_vehicles(image_path: str, conf: float, device: str) -> None:
    """Tasks 7 & 8 CLI runner: Detects cars, motorcycles, buses, and trucks."""
    from modules.vehicle_detection import VehicleDetector

    detector = VehicleDetector(conf_threshold=conf, device=device)
    res = detector.process_image(image_path, conf=conf)
    print(f"\n[Tasks 7 & 8: Multi-Class Vehicle Detection Results for {res['image_name']}]")
    print(f"Inference time: {res['inference_time_ms']} ms")
    print(f"Total vehicles detected: {res['total_vehicles']}")
    print("Breakdown:")
    for cls_name, count in res["counts"].items():
        print(f"  - {cls_name.capitalize():<12}: {count}")

    print("\nDetailed detections:")
    for i, det in enumerate(res["detections"], 1):
        print(f"  #{i:02d} [{det['class_name'].upper()}] Conf: {det['confidence']:.4f} | BBox: {det['bbox']}")

    print(f"\nAnnotated image saved to: {res['output_path']}\n")


def run_detect_all(conf: float, device: str) -> None:
    """Processes all images inside dataset/images/ and writes report."""
    from modules.vehicle_detection import VehicleDetector

    cfg = load_config()
    img_dir = get_path("dataset_images_dir", cfg)
    reports_dir = get_path("reports_dir", cfg)
    reports_dir.mkdir(parents=True, exist_ok=True)

    detector = VehicleDetector(conf_threshold=conf, device=device)
    summary = detector.process_image_directory(input_dir=img_dir, conf=conf)

    # Save JSON report
    json_path = reports_dir / "vehicle_detection_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Generate Markdown report
    md_path = reports_dir / "vehicle_detection_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# VisionPort – Phase 2: Vehicle Detection Report\n\n")
        f.write(f"- **Total Images Processed:** {summary['total_images_processed']}\n")
        f.write(f"- **Total Vehicles Detected:** {summary['total_vehicles_detected']}\n")
        f.write(f"- **Average Inference Time:** {summary['average_inference_time_ms']} ms\n")
        f.write(f"- **Detection Threshold:** {conf}\n\n")
        f.write("### Vehicle Category Distribution\n\n")
        f.write("| Vehicle Category | Count |\n")
        f.write("| :--- | :--- |\n")
        for cls_name, cnt in summary["class_counts"].items():
            f.write(f"| {cls_name.capitalize()} | {cnt} |\n")
        f.write(f"| **Total** | **{summary['total_vehicles_detected']}** |\n\n")

        f.write("### Per-Image Detection Breakdown\n\n")
        f.write("| Image | Dimensions | Cars | Bikes | Buses | Trucks | Total | Inference Time |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in summary["results"]:
            dims = f"{r['dimensions']['width']}x{r['dimensions']['height']}"
            c = r["counts"]
            f.write(
                f"| {r['image_name']} | {dims} | {c.get('car', 0)} | {c.get('motorcycle', 0)} | "
                f"{c.get('bus', 0)} | {c.get('truck', 0)} | **{r['total_vehicles']}** | {r['inference_time_ms']} ms |\n"
            )

    print("\n" + "=" * 80)
    print(" VISIONPORT - BATCH VEHICLE DETECTION COMPLETE")
    print("=" * 80)
    print(f"Processed images: {summary['total_images_processed']}")
    print(f"Total vehicles:  {summary['total_vehicles_detected']}")
    for cls_name, cnt in summary["class_counts"].items():
        print(f"  {cls_name.capitalize():<12}: {cnt}")
    print(f"Average time:    {summary['average_inference_time_ms']} ms/image")
    print(f"JSON Report:     {json_path}")
    print(f"Markdown Report: {md_path}")
    print("=" * 80 + "\n")


def run_detect_video(video_path: str, conf: float, device: str, max_frames: int = None) -> None:
    """Processes video stream with YOLO vehicle detection."""
    from modules.vehicle_detection import VehicleDetector

    detector = VehicleDetector(conf_threshold=conf, device=device)
    summary = detector.process_video(video_path=video_path, conf=conf, max_frames=max_frames)
    print("\n" + "=" * 80)
    print(" VISIONPORT - VIDEO VEHICLE DETECTION COMPLETE")
    print("=" * 80)
    print(f"Source video:     {summary['source_video']}")
    print(f"Output video:     {summary['output_video']}")
    print(f"Frames processed: {summary['frames_processed']}")
    print(f"Processing time:  {summary['processing_time_sec']}s ({summary['average_fps']} FPS)")
    print(f"Total detections: {summary['total_detections']}")
    print("Breakdown:")
    for cls_name, cnt in summary["cumulative_counts"].items():
        print(f"  {cls_name.capitalize():<12}: {cnt}")
    print("=" * 80 + "\n")


def run_count_vehicles(target_path: str, conf: float, device: str) -> None:
    """Task 9 CLI runner: Counts vehicles by category and exports reports."""
    from modules.vehicle_detection import VehicleDetector
    from modules.vehicle_counting import VehicleCounter
    import cv2

    cfg = load_config()
    reports_dir = get_path("reports_dir", cfg)
    detected_img_dir = get_path("detected_images_dir", cfg)

    src = Path(target_path).resolve()
    detector = VehicleDetector(conf_threshold=conf, device=device)
    counter = VehicleCounter()

    frame = cv2.imread(str(src))
    if frame is None:
        logger.error("Failed to load image from %s", src)
        return

    detections = detector.detect_multi_class(frame, conf=conf)
    metrics = counter.count_frame(detections, frame_idx=0)

    print("\n" + "=" * 80)
    print(f" TASK 9: VEHICLE COUNTING RESULTS - {src.name}")
    print("=" * 80)
    print(f"Total Vehicles: {metrics['total_vehicles']}")
    for k, v in metrics["counts"].items():
        print(f"  - {k.capitalize():<12}: {v}")

    annotated = detector.annotate_frame(frame, detections)
    annotated = counter.annotate_frame(annotated, metrics)
    out_img = detected_img_dir / f"counted_{src.stem}.jpg"
    cv2.imwrite(str(out_img), annotated)
    print(f"Annotated frame saved to: {out_img}")

    json_out = counter.export_json(reports_dir / "vehicle_counting_report.json")
    csv_out = counter.export_csv(reports_dir / "vehicle_counting_report.csv")
    print(f"Exported JSON: {json_out}")
    print(f"Exported CSV:  {csv_out}")
    print("=" * 80 + "\n")


def run_traffic_density(target_path: str, conf: float, device: str) -> None:
    """Task 10 CLI runner: Computes traffic density and level classification."""
    from modules.vehicle_detection import VehicleDetector
    from modules.traffic_density import TrafficDensityCalculator
    import cv2

    cfg = load_config()
    reports_dir = get_path("reports_dir", cfg)
    detected_img_dir = get_path("detected_images_dir", cfg)

    src = Path(target_path).resolve()
    detector = VehicleDetector(conf_threshold=conf, device=device)
    density_calc = TrafficDensityCalculator()

    frame = cv2.imread(str(src))
    if frame is None:
        logger.error("Failed to load image from %s", src)
        return

    detections = detector.detect_multi_class(frame, conf=conf)
    density = density_calc.calculate_density(detections, frame_shape=(frame.shape[0], frame.shape[1]))

    print("\n" + "=" * 80)
    print(f" TASK 10: TRAFFIC DENSITY ANALYSIS - {src.name}")
    print("=" * 80)
    print(f"Vehicle Count:     {density['vehicle_count']}")
    print(f"Density Tier:      {density['density_level']}")
    print(f"Density Score:     {density['density_score']:.3f} / 1.000")
    print(f"Road Occupancy:    {density['occupancy_percent']}%")
    print(f"Estimated Density: {density['density_per_sq_km']} vehicles/sq.km")

    print("\nDocumented Limitations of Count-Based Estimation:")
    for lim in TrafficDensityCalculator.get_limitations():
        print(f"  * {lim['limitation']}: {lim['impact']}")

    annotated = detector.annotate_frame(frame, detections)
    annotated = density_calc.annotate_frame(annotated, density)
    out_img = detected_img_dir / f"density_{src.stem}.jpg"
    cv2.imwrite(str(out_img), annotated)
    print(f"\nAnnotated frame saved to: {out_img}")

    json_out = density_calc.export_json(reports_dir / "traffic_density_report.json")
    csv_out = density_calc.export_csv(reports_dir / "traffic_density_report.csv")
    print(f"Exported JSON: {json_out}")
    print(f"Exported CSV:  {csv_out}")
    print("=" * 80 + "\n")


def run_detect_congestion(target_path: str, conf: float, device: str) -> None:
    """Task 11 CLI runner: Evaluates congestion and monitors alerts."""
    from modules.vehicle_detection import VehicleDetector
    from modules.traffic_density import TrafficDensityCalculator
    from modules.congestion_detection import CongestionDetector
    import cv2

    cfg = load_config()
    reports_dir = get_path("reports_dir", cfg)
    detected_img_dir = get_path("detected_images_dir", cfg)

    src = Path(target_path).resolve()
    detector = VehicleDetector(conf_threshold=conf, device=device)
    density_calc = TrafficDensityCalculator()
    congestion = CongestionDetector(consecutive_frames_threshold=1)  # immediate threshold for single image

    frame = cv2.imread(str(src))
    if frame is None:
        logger.error("Failed to load image from %s", src)
        return

    detections = detector.detect_multi_class(frame, conf=conf)
    density = density_calc.calculate_density(detections, frame_shape=(frame.shape[0], frame.shape[1]))
    cong_metrics = congestion.evaluate_frame(density, frame_idx=0)
    congestion.finalize()

    print("\n" + "=" * 80)
    print(f" TASK 11: CONGESTION EVALUATION - {src.name}")
    print("=" * 80)
    print(f"Traffic Status:    {cong_metrics['status']}")
    print(f"Is Congested:      {cong_metrics['is_congested']}")
    print(f"Alert Generated:   {cong_metrics['alert_generated']}")
    if cong_metrics['alert_message']:
        print(f"Alert Message:     {cong_metrics['alert_message']}")

    annotated = detector.annotate_frame(frame, detections)
    annotated = density_calc.annotate_frame(annotated, density)
    annotated = congestion.annotate_frame(annotated, cong_metrics)
    out_img = detected_img_dir / f"congestion_{src.stem}.jpg"
    cv2.imwrite(str(out_img), annotated)
    print(f"\nAnnotated frame saved to: {out_img}")

    json_out = congestion.export_json(reports_dir / "congestion_events_report.json")
    csv_out = congestion.export_csv(reports_dir / "congestion_events_report.csv")
    print(f"Exported JSON: {json_out}")
    print(f"Exported CSV:  {csv_out}")
    print("=" * 80 + "\n")


def run_track_vehicles(video_path: str, conf: float, device: str, max_frames: Optional[int]) -> None:
    """Task 12 CLI runner: tracks vehicles across all frames of a video."""
    from modules.vehicle_tracking import process_video_with_tracking

    cfg = load_config()
    videos_out_dir = get_path("processed_videos_dir", cfg)
    out_video = videos_out_dir / "tracked_traffic.mp4"

    result = process_video_with_tracking(
        video_path=video_path,
        output_video_path=out_video,
        conf_threshold=conf,
        device=device,
        max_frames=max_frames,
        draw_trajectory=True,
    )
    summary = result["summary"]
    unique  = summary["unique_vehicle_counts"]

    print("\n" + "=" * 80)
    print(" VISIONPORT - TASK 12: VEHICLE TRACKING COMPLETE")
    print("=" * 80)
    print(f"Source Video:       {video_path}")
    print(f"Output Video:       {result['output_video']}")
    print(f"Frames Processed:   {summary['frames_processed']}")
    print(f"Average FPS:        {summary['average_fps']}")
    print(f"\nUnique Vehicles Tracked: {unique['total']}")
    print(f"  Cars:        {unique['car']}")
    print(f"  Motorcycles: {unique['motorcycle']}")
    print(f"  Buses:       {unique['bus']}")
    print(f"  Trucks:      {unique['truck']}")
    print(f"\nTracking JSON: {result['tracking_json']}")
    print(f"Tracking CSV:  {result['tracking_csv']}")
    print("=" * 80 + "\n")


def run_estimate_speed(video_path: str, conf: float, device: str, max_frames: Optional[int]) -> None:
    """Task 13 CLI runner: tracks vehicles and overlays speed estimates on output video."""
    from modules.speed_estimation import process_video_with_tracking_and_speed

    cfg = load_config()
    videos_out_dir = get_path("processed_videos_dir", cfg)
    out_video = videos_out_dir / "tracked_traffic.mp4"

    res = process_video_with_tracking_and_speed(
        video_path=video_path,
        output_video_path=out_video,
        conf_threshold=conf,
        device=device,
        max_frames=max_frames,
        draw_trajectory=True,
    )

    print("\n" + "=" * 80)
    print(" VISIONPORT - TASK 13: SPEED ESTIMATION COMPLETE")
    print("=" * 80)
    print(f"Speed unit:         {res['speed_unit']}")
    if not res["calibrated"]:
        print("  NOTE: Set pixels_per_meter in config.yaml for real-world km/h.")
    print(f"Output Video:       {res['output_video']}")
    print(f"Frames Processed:   {res['frames_processed']}  ({res['average_fps']} FPS)")
    print(f"Average Speed:      {res['average_speed']} {res['speed_unit']}")
    print("\nGenerated Reports:")
    for f in res["files_generated"]:
        print(f"  * {f}")
    print("=" * 80 + "\n")


def run_analyze_traffic(target_path: Optional[str], conf: float, device: str, max_frames: Optional[int] = None) -> None:
    """Integrated Phase 3 Pipeline CLI runner: executes complete analytics stack."""
    from modules.traffic_analytics import TrafficAnalyticsPipeline

    cfg = load_config()
    pipeline = TrafficAnalyticsPipeline(conf_threshold=conf, device=device)

    # Determine whether target is a video or image or directory
    if target_path is None:
        videos_dir = get_path("dataset_videos_dir", cfg)
        sample_vid = videos_dir / "sample_traffic.mp4"
        if sample_vid.exists():
            target_path = str(sample_vid)
        else:
            images_dir = get_path("dataset_images_dir", cfg)
            target_path = str(images_dir)

    target = Path(target_path).resolve()
    if target.is_dir():
        print(f"\n[Running Integrated Traffic Analytics on Image Directory: {target}]")
        summary = pipeline.process_images_batch(images_dir=target)
        print("\n" + "=" * 80)
        print(" VISIONPORT - INTEGRATED BATCH TRAFFIC ANALYTICS COMPLETE")
        print("=" * 80)
        print(f"Images Analyzed:     {summary['total_images_processed']}")
        print(f"Peak Vehicle Volume: {summary['counting_summary']['peak_vehicles_in_single_frame']}")
        print(f"Average Vehicles:    {summary['counting_summary']['average_vehicles_per_frame']}")
        print(f"Total Congestion Events: {summary['congestion_summary']['total_congestion_events']}")
        print("Reports Generated in output/reports/ and output/analytics/")
        print("=" * 80 + "\n")
    else:
        # Check if video extension
        if target.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            print(f"\n[Running Integrated Traffic Analytics on Video: {target.name}]")
            result = pipeline.process_video(video_path=target, max_frames=max_frames)
            summary = result["summary"]
            print("\n" + "=" * 80)
            print(" VISIONPORT - INTEGRATED VIDEO TRAFFIC ANALYTICS COMPLETE")
            print("=" * 80)
            print(f"Output Video:        {result['output_video']}")
            print(f"Frames Processed:    {summary['frames_processed']}")
            print(f"Average FPS:         {summary['average_fps']}")
            print(f"Peak Vehicles:       {summary['counting_summary']['peak_vehicles_in_single_frame']}")
            print(f"Congestion Events:   {summary['congestion_summary']['total_congestion_events']}")
            print("\nGenerated Reports and Outputs:")
            for fpath in result["files_generated"]:
                print(f"  * {fpath}")
            print("=" * 80 + "\n")
        else:
            # Single image analysis
            import cv2
            frame = cv2.imread(str(target))
            if frame is None:
                logger.error("Failed to decode image from %s", target)
                return
            annotated_frame, metrics = pipeline.process_frame(frame, frame_idx=0)
            pipeline.congestion_detector.finalize()
            out_dir = get_path("detected_images_dir", cfg)
            out_file = out_dir / f"analytics_{target.stem}.jpg"
            cv2.imwrite(str(out_file), annotated_frame)
            print("\n" + "=" * 80)
            print(f" INTEGRATED TRAFFIC ANALYTICS - {target.name}")
            print("=" * 80)
            print(f"Counted Vehicles:   {metrics['counted_vehicles']}")
            print(f"Density Tier:       {metrics['density_level']}")
            print(f"Congestion Status:  {metrics['congestion_status']}")
            print(f"Saved Image:        {out_file}")
            print("=" * 80 + "\n")


def main() -> None:
    """Main CLI parser and execution controller."""
    parser = argparse.ArgumentParser(
        description="VisionPort – Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Task 1 & Architecture
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run Task 1 environment and project integrity checks.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Display the 15 project assignments and their module locations.",
    )

    # Tasks 2-5 Foundational Computer Vision
    parser.add_argument(
        "--capture-image",
        action="store_true",
        help="Run Task 2: Capture a traffic frame from camera to captures/.",
    )
    parser.add_argument(
        "--display-image",
        action="store_true",
        help="Run Task 3: Load and display a traffic image using OpenCV.",
    )
    parser.add_argument(
        "--play-video",
        action="store_true",
        help="Run Task 4: Play back a traffic video stream.",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Run Task 5: Extract video frames to output/extracted_frames/.",
    )

    # Tasks 6-8 Deep Learning Vehicle Detection
    parser.add_argument(
        "--detect-cars",
        action="store_true",
        help="Run Task 6: Detect only cars in the specified image.",
    )
    parser.add_argument(
        "--detect-vehicles",
        action="store_true",
        help="Run Tasks 7 & 8: Detect cars, motorcycles, buses, and trucks.",
    )
    parser.add_argument(
        "--detect-all",
        action="store_true",
        help="Run vehicle detection across all images in dataset/images/.",
    )
    parser.add_argument(
        "--detect-video",
        action="store_true",
        help="Run vehicle detection on a traffic video stream.",
    )

    # Tasks 12–13 Vehicle Tracking & Speed Estimation
    parser.add_argument(
        "--track-vehicles",
        action="store_true",
        help="Run Task 12: Track individual vehicles with ByteTrack across a video.",
    )
    parser.add_argument(
        "--estimate-speed",
        action="store_true",
        help="Run Task 13: Estimate vehicle speeds (requires --path to a video).",
    )

    # Tasks 9-11 Traffic Analytics
    parser.add_argument(
        "--count-vehicles",
        action="store_true",
        help="Run Task 9: Count vehicles by category and export reports.",
    )
    parser.add_argument(
        "--traffic-density",
        action="store_true",
        help="Run Task 10: Calculate traffic density tier and occupancy.",
    )
    parser.add_argument(
        "--detect-congestion",
        action="store_true",
        help="Run Task 11: Evaluate traffic congestion and alerts.",
    )
    parser.add_argument(
        "--analyze-traffic",
        action="store_true",
        help="Run Integrated Phase 3 Traffic Analytics across video or image batch.",
    )

    # General configuration parameters
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Target file or directory path for processing.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence score threshold for detections (default: 0.40).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: 'auto', 'cpu', or 'cuda' (default: 'auto').",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum video frames to process.",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config()
        logger.info(
            "Loaded configuration for project: %s (v%s)",
            config["project"]["name"], config["project"]["version"]
        )
    except Exception as exc:
        logger.warning("Could not load configuration: %s", exc)

    if args.list_tasks:
        list_tasks()
        return

    if args.check_env:
        check_environment()
        return

    # Task 2
    if args.capture_image:
        from modules.image_capture import capture_traffic_image
        saved = capture_traffic_image()
        if saved:
            print(f"Captured image saved to: {saved}")
        return

    # Task 3
    if args.display_image:
        if not args.path:
            logger.error("Please provide an image path using --path <path_to_image>")
            return
        from modules.image_display import display_image
        display_image(args.path)
        return

    # Task 4
    if args.play_video:
        if not args.path:
            logger.error("Please provide a video path using --path <path_to_video>")
            return
        from modules.video_processing import play_traffic_video
        play_traffic_video(args.path)
        return

    # Task 5
    if args.extract_frames:
        if not args.path:
            logger.error("Please provide a video path using --path <path_to_video>")
            return
        from modules.frame_extraction import extract_frames_from_video
        frames = extract_frames_from_video(args.path)
        print(f"Extracted {len(frames)} frames.")
        return

    # Task 6
    if args.detect_cars:
        if not args.path:
            logger.error("Please provide an image path using --path <path_to_image>")
            return
        run_detect_cars(args.path, conf=args.conf, device=args.device)
        return

    # Tasks 7 & 8
    if args.detect_vehicles:
        if not args.path:
            logger.error("Please provide an image path using --path <path_to_image>")
            return
        run_detect_vehicles(args.path, conf=args.conf, device=args.device)
        return

    # Batch images
    if args.detect_all:
        run_detect_all(conf=args.conf, device=args.device)
        return

    # Video detection
    if args.detect_video:
        vid_path = args.path
        if not vid_path:
            cfg = load_config()
            vids_dir = get_path("dataset_videos_dir", cfg)
            sample_vid = vids_dir / "sample_traffic.mp4"
            if sample_vid.exists():
                vid_path = str(sample_vid)
            else:
                logger.error("Please provide a video path using --path <path_to_video>")
                return
        run_detect_video(vid_path, conf=args.conf, device=args.device, max_frames=args.max_frames)
        return

    # Task 9
    if args.count_vehicles:
        if not args.path:
            logger.error("Please provide an image or video path using --path <path>")
            return
        run_count_vehicles(args.path, conf=args.conf, device=args.device)
        return

    # Task 10
    if args.traffic_density:
        if not args.path:
            logger.error("Please provide an image or video path using --path <path>")
            return
        run_traffic_density(args.path, conf=args.conf, device=args.device)
        return

    # Task 11
    if args.detect_congestion:
        if not args.path:
            logger.error("Please provide an image or video path using --path <path>")
            return
        run_detect_congestion(args.path, conf=args.conf, device=args.device)
        return

    # Task 12 – Vehicle Tracking
    if args.track_vehicles:
        vid_path = args.path
        if not vid_path:
            cfg = load_config()
            vids_dir = get_path("dataset_videos_dir", cfg)
            sample_vid = vids_dir / "sample_traffic.mp4"
            if sample_vid.exists():
                vid_path = str(sample_vid)
            else:
                logger.error("Please provide a video path using --path <path_to_video>")
                return
        run_track_vehicles(vid_path, conf=args.conf, device=args.device, max_frames=args.max_frames)
        return

    # Task 13 – Speed Estimation
    if args.estimate_speed:
        vid_path = args.path
        if not vid_path:
            cfg = load_config()
            vids_dir = get_path("dataset_videos_dir", cfg)
            sample_vid = vids_dir / "sample_traffic.mp4"
            if sample_vid.exists():
                vid_path = str(sample_vid)
            else:
                logger.error("Please provide a video path using --path <path_to_video>")
                return
        run_estimate_speed(vid_path, conf=args.conf, device=args.device, max_frames=args.max_frames)
        return

    # Phase 3 Integrated Analytics
    if args.analyze_traffic:
        run_analyze_traffic(target_path=args.path, conf=args.conf, device=args.device, max_frames=args.max_frames)
        return

    # Default action if no flags provided:
    list_tasks()
    print("\nTip: Run 'python main.py --help' to see all available CLI options.")
    print("Tip: Run 'python main.py --analyze-traffic' to run integrated traffic analytics.")


if __name__ == "__main__":
    main()
