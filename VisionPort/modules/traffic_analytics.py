"""
Module: traffic_analytics.py
Phase 3 Integrated Analytics Pipeline for VisionPort.

Orchestrates:
- Task 6, 7, 8: Pretrained YOLOv8 vehicle detection.
- Task 9: Vehicle counting and category breakdown (cars, motorcycles, buses, trucks).
- Task 10: Traffic density calculation (LOW, MEDIUM, HIGH) with ROI occupancy.
- Task 11: Congestion detection with temporal persistence, alerts, and duration tracking.

Renders composite HUD overlays, saves annotated video streams, and exports
specialized and unified JSON and CSV analytics reports.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import csv
import json
import time
import cv2
import numpy as np

from utils.logger import setup_logger
from utils.config import load_config, get_path
from modules.vehicle_detection import VehicleDetector
from modules.vehicle_counting import VehicleCounter
from modules.traffic_density import TrafficDensityCalculator
from modules.congestion_detection import CongestionDetector

logger = setup_logger("TrafficAnalyticsPipeline")


class TrafficAnalyticsPipeline:
    """
    Unified traffic monitoring pipeline integrating YOLO detection,
    counting, density estimation, and congestion alerting.
    """

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        conf_threshold: float = 0.40,
        device: str = "auto",
        low_density_thresh: int = 5,
        medium_density_thresh: int = 15,
        high_density_thresh: int = 25,
        congestion_consecutive_frames: int = 5,
        roi_polygon: Optional[List[Tuple[int, int]]] = None,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    ):
        """Initializes all sub-modules with coordinated thresholds and geometry."""
        self.detector = VehicleDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device,
        )
        self.counter = VehicleCounter(roi_polygon=roi_polygon, roi_bbox=roi_bbox)
        self.density_calc = TrafficDensityCalculator(
            low_threshold=low_density_thresh,
            medium_threshold=medium_density_thresh,
            high_threshold=high_density_thresh,
            roi_polygon=roi_polygon,
            roi_bbox=roi_bbox,
        )
        self.congestion_detector = CongestionDetector(
            consecutive_frames_threshold=congestion_consecutive_frames,
            recovery_frames_threshold=3,
        )

        logger.info("TrafficAnalyticsPipeline successfully initialized.")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        timestamp_sec: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a single video frame through the complete analytics stack:
        1. YOLO vehicle detection.
        2. Vehicle counting by category.
        3. Traffic density calculation.
        4. Congestion evaluation and alert check.
        5. Composite HUD annotation.
        
        Args:
            frame: Input image array in BGR format.
            frame_idx: Sequential frame index.
            timestamp_sec: Video timestamp in seconds.
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (Annotated frame, frame metrics dictionary).
        """
        # Step 1: Detect vehicles
        detections = self.detector.detect_multi_class(frame)

        # Step 2: Count vehicles
        counting_metrics = self.counter.count_frame(
            detections=detections,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
        )

        # Step 3: Calculate density
        density_metrics = self.density_calc.calculate_density(
            detections=detections,
            frame_shape=(frame.shape[0], frame.shape[1]),
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
        )

        # Step 4: Evaluate congestion
        congestion_metrics = self.congestion_detector.evaluate_frame(
            density_metrics=density_metrics,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
        )

        # Step 5: Render composite annotations
        # 5a. Draw vehicle bounding boxes
        annotated = self.detector.annotate_frame(frame, detections)
        # 5b. Overlay counting HUD panel
        annotated = self.counter.annotate_frame(annotated, counting_metrics)
        # 5c. Overlay density status badge
        annotated = self.density_calc.annotate_frame(annotated, density_metrics)
        # 5d. Overlay congestion alert banner if active
        annotated = self.congestion_detector.annotate_frame(annotated, congestion_metrics)

        composite_metrics = {
            "frame_index": frame_idx,
            "timestamp_sec": counting_metrics["timestamp_sec"],
            "detections_count": len(detections),
            "counted_vehicles": counting_metrics["total_vehicles"],
            "counts_by_class": counting_metrics["counts"],
            "density_level": density_metrics["density_level"],
            "density_score": density_metrics["density_score"],
            "occupancy_percent": density_metrics["occupancy_percent"],
            "is_congested": congestion_metrics["is_congested"],
            "congestion_status": congestion_metrics["status"],
            "active_congestion_duration": congestion_metrics["active_duration_sec"],
            "alert_generated": congestion_metrics["alert_generated"],
        }
        return annotated, composite_metrics

    def process_video(
        self,
        video_path: Union[Path, str],
        output_video_path: Optional[Union[Path, str]] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs complete analytics across a video stream, writes processed video to disk,
        and generates all reports in output/reports/ and output/analytics/.
        
        Args:
            video_path: Path to input traffic video file.
            output_video_path: Path to output MP4 video file.
            max_frames: Maximum frames to process (useful for tests or quick verification).
            
        Returns:
            Dict[str, Any]: Comprehensive analytics summary.
        """
        src = Path(video_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Video file not found: {src}")

        # Resolve output directories
        cfg = load_config()
        reports_dir = get_path("reports_dir", cfg)
        analytics_dir = get_path("analytics_dir", cfg)
        videos_dir = get_path("processed_videos_dir", cfg)

        reports_dir.mkdir(parents=True, exist_ok=True)
        analytics_dir.mkdir(parents=True, exist_ok=True)
        videos_dir.mkdir(parents=True, exist_ok=True)

        if output_video_path is None:
            out_video = videos_dir / f"analytics_{src.stem}.mp4"
        else:
            out_video = Path(output_video_path).resolve()
            out_video.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream: {src}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.congestion_detector.fps = fps
        total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        from utils.video_compat import create_compatible_video_writer, finalize_video_file

        writer = create_compatible_video_writer(out_video, fps, width, height)

        logger.info(
            "Starting integrated traffic analytics on '%s' (%dx%d @ %.1f FPS, %d frames)...",
            src.name, width, height, fps, total_source_frames
        )

        all_frame_metrics: List[Dict[str, Any]] = []
        frame_idx = 0
        t_start = time.perf_counter()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                ts = round(frame_idx / fps, 3)
                annotated_frame, metrics = self.process_frame(frame, frame_idx=frame_idx, timestamp_sec=ts)
                writer.write(annotated_frame)
                all_frame_metrics.append(metrics)

                frame_idx += 1
                if max_frames and frame_idx >= max_frames:
                    logger.info("Reached maximum frames limit of %d.", max_frames)
                    break
        finally:
            cap.release()
            writer.release()
            finalize_video_file(out_video)
            self.congestion_detector.finalize()

        elapsed_sec = round(time.perf_counter() - t_start, 2)
        avg_fps = round(frame_idx / elapsed_sec, 2) if elapsed_sec > 0 else 0.0

        # Export individual module reports
        counting_json = self.counter.export_json(reports_dir / "vehicle_counting_report.json")
        counting_csv = self.counter.export_csv(reports_dir / "vehicle_counting_report.csv")

        density_json = self.density_calc.export_json(reports_dir / "traffic_density_report.json")
        density_csv = self.density_calc.export_csv(reports_dir / "traffic_density_report.csv")

        congestion_json = self.congestion_detector.export_json(reports_dir / "congestion_events_report.json")
        congestion_csv = self.congestion_detector.export_csv(reports_dir / "congestion_events_report.csv")

        # Export unified combined analytics
        combined_json = analytics_dir / "combined_traffic_analytics.json"
        combined_csv = analytics_dir / "combined_traffic_analytics.csv"

        unified_summary = {
            "source_video": str(src),
            "output_video": str(out_video),
            "resolution": f"{width}x{height}",
            "frames_processed": frame_idx,
            "processing_time_sec": elapsed_sec,
            "average_fps": avg_fps,
            "counting_summary": self.counter.get_summary(),
            "density_summary": self.density_calc.get_summary(),
            "congestion_summary": self.congestion_detector.get_summary(),
            "density_estimation_limitations": self.density_calc.get_limitations(),
        }

        with open(combined_json, "w", encoding="utf-8") as f:
            json.dump({
                "summary": unified_summary,
                "timeline": all_frame_metrics,
            }, f, indent=2)

        # Export unified CSV
        csv_headers = [
            "frame_index",
            "timestamp_sec",
            "counted_vehicles",
            "car_count",
            "motorcycle_count",
            "bus_count",
            "truck_count",
            "density_level",
            "occupancy_percent",
            "is_congested",
            "congestion_status",
            "congestion_duration_sec",
        ]
        with open(combined_csv, "w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=csv_headers)
            writer_csv.writeheader()
            for m in all_frame_metrics:
                writer_csv.writerow({
                    "frame_index": m["frame_index"],
                    "timestamp_sec": m["timestamp_sec"],
                    "counted_vehicles": m["counted_vehicles"],
                    "car_count": m["counts_by_class"].get("car", 0),
                    "motorcycle_count": m["counts_by_class"].get("motorcycle", 0),
                    "bus_count": m["counts_by_class"].get("bus", 0),
                    "truck_count": m["counts_by_class"].get("truck", 0),
                    "density_level": m["density_level"],
                    "occupancy_percent": m["occupancy_percent"],
                    "is_congested": m["is_congested"],
                    "congestion_status": m["congestion_status"],
                    "congestion_duration_sec": m["active_congestion_duration"],
                })

        logger.info(
            "Integrated Traffic Analytics complete: %d frames -> %s (FPS: %.1f)",
            frame_idx, out_video, avg_fps
        )
        return {
            "summary": unified_summary,
            "output_video": str(out_video),
            "files_generated": [
                str(out_video),
                str(counting_json),
                str(counting_csv),
                str(density_json),
                str(density_csv),
                str(congestion_json),
                str(congestion_csv),
                str(combined_json),
                str(combined_csv),
            ],
        }

    def process_images_batch(
        self,
        images_dir: Union[Path, str],
        output_dir: Optional[Union[Path, str]] = None,
    ) -> Dict[str, Any]:
        """
        Runs integrated analytics across a directory of real traffic photographs.
        
        Args:
            images_dir: Directory containing input images.
            output_dir: Destination directory for annotated output images.
            
        Returns:
            Dict[str, Any]: Aggregated batch analytics summary.
        """
        in_path = Path(images_dir).resolve()
        cfg = load_config()
        reports_dir = get_path("reports_dir", cfg)
        analytics_dir = get_path("analytics_dir", cfg)
        if output_dir is None:
            out_img_dir = get_path("detected_images_dir", cfg)
        else:
            out_img_dir = Path(output_dir).resolve()

        reports_dir.mkdir(parents=True, exist_ok=True)
        analytics_dir.mkdir(parents=True, exist_ok=True)
        out_img_dir.mkdir(parents=True, exist_ok=True)

        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = sorted([p for p in in_path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])

        logger.info("Running integrated analytics on %d images from %s...", len(images), in_path)

        batch_results: List[Dict[str, Any]] = []
        for idx, img_file in enumerate(images):
            frame = cv2.imread(str(img_file))
            if frame is None:
                continue

            annotated_frame, metrics = self.process_frame(frame, frame_idx=idx, timestamp_sec=round(idx * 1.0, 2))
            metrics["image_name"] = img_file.name

            out_file = out_img_dir / f"analytics_{img_file.stem}.jpg"
            cv2.imwrite(str(out_file), annotated_frame)
            metrics["saved_image"] = str(out_file)
            batch_results.append(metrics)

        self.congestion_detector.finalize()

        # Export reports
        self.counter.export_json(reports_dir / "vehicle_counting_report.json")
        self.counter.export_csv(reports_dir / "vehicle_counting_report.csv")
        self.density_calc.export_json(reports_dir / "traffic_density_report.json")
        self.density_calc.export_csv(reports_dir / "traffic_density_report.csv")
        self.congestion_detector.export_json(reports_dir / "congestion_events_report.json")
        self.congestion_detector.export_csv(reports_dir / "congestion_events_report.csv")

        # Combined reports
        combined_json = analytics_dir / "combined_traffic_analytics.json"
        combined_csv = analytics_dir / "combined_traffic_analytics.csv"

        unified_data = {
            "total_images_processed": len(batch_results),
            "counting_summary": self.counter.get_summary(),
            "density_summary": self.density_calc.get_summary(),
            "congestion_summary": self.congestion_detector.get_summary(),
            "density_limitations": self.density_calc.get_limitations(),
            "batch_timeline": batch_results,
        }

        with open(combined_json, "w", encoding="utf-8") as f:
            json.dump(unified_data, f, indent=2)

        csv_headers = [
            "image_name",
            "frame_index",
            "counted_vehicles",
            "car_count",
            "motorcycle_count",
            "bus_count",
            "truck_count",
            "density_level",
            "occupancy_percent",
            "is_congested",
            "congestion_status",
        ]
        with open(combined_csv, "w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=csv_headers)
            writer_csv.writeheader()
            for b in batch_results:
                writer_csv.writerow({
                    "image_name": b["image_name"],
                    "frame_index": b["frame_index"],
                    "counted_vehicles": b["counted_vehicles"],
                    "car_count": b["counts_by_class"].get("car", 0),
                    "motorcycle_count": b["counts_by_class"].get("motorcycle", 0),
                    "bus_count": b["counts_by_class"].get("bus", 0),
                    "truck_count": b["counts_by_class"].get("truck", 0),
                    "density_level": b["density_level"],
                    "occupancy_percent": b["occupancy_percent"],
                    "is_congested": b["is_congested"],
                    "congestion_status": b["congestion_status"],
                })

        logger.info("Batch analytics successfully written to %s and %s", combined_json, combined_csv)
        return unified_data

