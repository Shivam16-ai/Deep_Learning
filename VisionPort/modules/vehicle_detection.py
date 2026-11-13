"""
Module: vehicle_detection.py
Tasks 6, 7, 8:
- Task 6: Car detection using YOLO.
- Task 7: Bike (motorcycle), bus, and truck detection.
- Task 8: Vehicle labels and confidence scores.

Provides complete, production-grade vehicle detection utilizing Ultralytics YOLOv8
pretrained on COCO, with hardware acceleration (CUDA/CPU), batch processing for images,
and video inference pipelines on Windows.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from utils.logger import setup_logger
from utils.config import load_config, get_path
from utils.visualization import draw_detections

logger = setup_logger("VehicleDetection")


class VehicleDetector:
    """
    YOLOv8-based vehicle detector for traffic surveillance and monitoring.
    
    Supports:
    - Task 6: Single-class car detection (COCO class ID 2).
    - Task 7: Multi-class vehicle detection (Car: 2, Motorcycle: 3, Bus: 5, Truck: 7).
    - Task 8: Bounding box coordinates, class labels, and confidence score extraction.
    """

    # COCO Class mapping for target vehicles
    TARGET_CLASSES: Dict[str, int] = {
        "car": 2,
        "motorcycle": 3,
        "bus": 5,
        "truck": 7,
    }

    # Inverse mapping from ID to standardized name
    ID_TO_CLASS: Dict[int, str] = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """
        Initializes the VehicleDetector with model weights, thresholds, and execution device.
        
        Args:
            model_path: Path to the YOLO weights file. Defaults to config or models/yolov8n.pt.
            conf_threshold: Minimum confidence score for a valid detection (0.0 to 1.0).
            iou_threshold: Intersection over Union threshold for Non-Maximum Suppression.
            device: Computing device ('cpu', 'cuda', 'cuda:0', or 'auto').
        """
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.model: Optional[YOLO] = None

        # Resolve model path
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

        # Determine computing device
        if device is None or device.lower() == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device.lower()

        logger.info(
            "Initializing VehicleDetector | Weights: %s | Device: %s | Conf: %.2f | IoU: %.2f",
            self.model_path.name, self.device, self.conf_threshold, self.iou_threshold
        )
        self.load_model()

    def load_model(self) -> None:
        """
        Loads the Ultralytics YOLO model from model_path.
        If weights are absent, Ultralytics downloads yolov8n.pt and saves to the models directory.
        """
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Loading YOLO model from %s...", self.model_path)
            self.model = YOLO(str(self.model_path))
            
            # Verify model is ready
            logger.info(
                "YOLO model successfully loaded on %s (%s classes in model registry).",
                self.device, len(self.model.names)
            )
        except Exception as exc:
            logger.exception("Failed to load YOLO model from %s: %s", self.model_path, exc)
            raise RuntimeError(f"Unable to load YOLO model from {self.model_path}: {exc}") from exc

    def _format_detections(
        self,
        boxes,
        allowed_class_ids: List[int],
        conf_thresh: float,
    ) -> List[Dict[str, Any]]:
        """
        Extracts and formats bounding box coordinates, labels, and confidences from YOLO boxes.
        
        Args:
            boxes: Ultralytics result.boxes object.
            allowed_class_ids: Filter list of integer COCO class IDs.
            conf_thresh: Confidence cutoff threshold.
            
        Returns:
            List[Dict[str, Any]]: Structured detections.
        """
        detections: List[Dict[str, Any]] = []
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(xyxy, confs, classes):
            if cls_id in allowed_class_ids and conf >= conf_thresh:
                x1, y1, x2, y2 = [round(float(v), 2) for v in box]
                class_name = self.ID_TO_CLASS.get(int(cls_id), self.model.names.get(int(cls_id), "vehicle"))
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "confidence": round(float(conf), 4),
                })

        return detections

    def detect_cars(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Task 6: Detects exclusively cars in an input frame.
        
        Args:
            frame: Input image array in BGR format.
            conf: Optional confidence threshold override.
            
        Returns:
            List[Dict[str, Any]]: Detected car bounding boxes and confidences.
        """
        if self.model is None:
            self.load_model()

        threshold = conf if conf is not None else self.conf_threshold
        car_class_id = self.TARGET_CLASSES["car"]

        # Run YOLO inference filtering directly by COCO class ID 2
        results = self.model(
            source=frame,
            classes=[car_class_id],
            conf=threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            detections = self._format_detections(
                results[0].boxes,
                allowed_class_ids=[car_class_id],
                conf_thresh=threshold,
            )

        logger.debug("Task 6: Detected %d cars (conf >= %.2f)", len(detections), threshold)
        return detections

    def detect_multi_class(
        self,
        frame: np.ndarray,
        classes: Optional[List[str]] = None,
        conf: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tasks 7 & 8: Detects multiple vehicle classes (cars, motorcycles, buses, trucks)
        along with labels and confidence scores.
        
        Args:
            frame: Input image array in BGR format.
            classes: Optional list of class names to filter (e.g., ['car', 'motorcycle', 'bus', 'truck']).
                     Defaults to all 4 target classes.
            conf: Optional confidence threshold override.
            
        Returns:
            List[Dict[str, Any]]: Structured detections containing:
                - 'bbox': [x1, y1, x2, y2]
                - 'class_id': int
                - 'class_name': str
                - 'confidence': float
        """
        if self.model is None:
            self.load_model()

        threshold = conf if conf is not None else self.conf_threshold

        # Resolve target class IDs
        if classes is None:
            target_ids = list(self.TARGET_CLASSES.values())
        else:
            target_ids = [
                self.TARGET_CLASSES[c.lower()]
                for c in classes
                if c.lower() in self.TARGET_CLASSES
            ]
            if not target_ids:
                logger.warning("No valid vehicle classes found in %s; using default classes.", classes)
                target_ids = list(self.TARGET_CLASSES.values())

        # Run inference with Ultralytics YOLO
        results = self.model(
            source=frame,
            classes=target_ids,
            conf=threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            detections = self._format_detections(
                results[0].boxes,
                allowed_class_ids=target_ids,
                conf_thresh=threshold,
            )

        logger.debug(
            "Task 7/8: Detected %d vehicles across classes %s (conf >= %.2f)",
            len(detections), [self.ID_TO_CLASS[cid] for cid in target_ids], threshold
        )
        return detections

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        show_labels: bool = True,
        show_conf: bool = True,
    ) -> np.ndarray:
        """
        Draws vehicle bounding boxes, class labels, and confidence scores on the frame.
        
        Args:
            frame: Input image array in BGR format.
            detections: List of detection dictionaries.
            show_labels: Whether to show class name.
            show_conf: Whether to show confidence score.
            
        Returns:
            np.ndarray: Annotated image array.
        """
        return draw_detections(
            frame=frame,
            detections=detections,
            show_labels=show_labels,
            show_conf=show_conf,
        )

    def process_image(
        self,
        image_path: Union[Path, str],
        output_path: Optional[Union[Path, str]] = None,
        classes: Optional[List[str]] = None,
        conf: Optional[float] = None,
        save_annotated: bool = True,
    ) -> Dict[str, Any]:
        """
        Processes a single traffic image: reads file, executes YOLO vehicle detection,
        annotates the frame, and saves the result inside output/detected_images/.
        
        Args:
            image_path: Path to the input image file.
            output_path: Optional path to save annotated output.
            classes: Classes to detect ('car', 'motorcycle', 'bus', 'truck').
            conf: Confidence threshold override.
            save_annotated: Whether to write the annotated image to disk.
            
        Returns:
            Dict[str, Any]: Detection metadata, vehicle counts, list of detections, and timing.
        """
        src_path = Path(image_path).resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"Input image not found: {src_path}")

        frame = cv2.imread(str(src_path))
        if frame is None:
            raise ValueError(f"Failed to decode image: {src_path}")

        t0 = time.perf_counter()
        detections = self.detect_multi_class(frame, classes=classes, conf=conf)
        inference_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Count per vehicle category
        counts: Dict[str, int] = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        for det in detections:
            cls_name = det["class_name"]
            counts[cls_name] = counts.get(cls_name, 0) + 1

        annotated_frame = self.annotate_frame(frame, detections)

        saved_file: Optional[Path] = None
        if save_annotated:
            if output_path is None:
                try:
                    cfg = load_config()
                    out_dir = get_path("detected_images_dir", cfg)
                except Exception:
                    out_dir = Path("output/detected_images").resolve()
                out_dir.mkdir(parents=True, exist_ok=True)
                saved_file = out_dir / f"detected_{src_path.stem}.jpg"
            else:
                saved_file = Path(output_path).resolve()
                saved_file.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(saved_file), annotated_frame)
            logger.info("Saved annotated detection image to: %s", saved_file)

        result = {
            "image_name": src_path.name,
            "image_path": str(src_path),
            "output_path": str(saved_file) if saved_file else None,
            "dimensions": {"width": frame.shape[1], "height": frame.shape[0]},
            "inference_time_ms": inference_time_ms,
            "total_vehicles": len(detections),
            "counts": counts,
            "detections": detections,
        }
        return result

    def process_image_directory(
        self,
        input_dir: Union[Path, str],
        output_dir: Optional[Union[Path, str]] = None,
        classes: Optional[List[str]] = None,
        conf: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Batch processes all images in a directory, executes YOLO detection,
        saves all annotated images, and returns an aggregated detection summary.
        
        Args:
            input_dir: Directory containing input traffic images.
            output_dir: Directory to store annotated images. Defaults to output/detected_images.
            classes: Vehicle classes to filter.
            conf: Confidence threshold override.
            
        Returns:
            Dict[str, Any]: Comprehensive batch summary report.
        """
        in_path = Path(input_dir).resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {in_path}")

        # Supported image extensions
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_files = sorted([
            p for p in in_path.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions
        ])

        if not image_files:
            logger.warning("No valid images found in %s", in_path)
            return {"total_images": 0, "results": [], "summary_counts": {}}

        if output_dir is None:
            try:
                cfg = load_config()
                out_path = get_path("detected_images_dir", cfg)
            except Exception:
                out_path = Path("output/detected_images").resolve()
        else:
            out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        logger.info("Processing %d images from %s with YOLOv8...", len(image_files), in_path)

        results: List[Dict[str, Any]] = []
        total_counts: Dict[str, int] = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        total_time_ms = 0.0

        for img_file in image_files:
            dest_file = out_path / f"detected_{img_file.stem}.jpg"
            img_result = self.process_image(
                image_path=img_file,
                output_path=dest_file,
                classes=classes,
                conf=conf,
                save_annotated=True,
            )
            results.append(img_result)
            total_time_ms += img_result["inference_time_ms"]
            for cls_k, cls_cnt in img_result["counts"].items():
                total_counts[cls_k] = total_counts.get(cls_k, 0) + cls_cnt

        total_vehicles = sum(total_counts.values())
        avg_time = round(total_time_ms / len(image_files), 2) if image_files else 0.0

        batch_summary = {
            "input_directory": str(in_path),
            "output_directory": str(out_path),
            "total_images_processed": len(image_files),
            "total_vehicles_detected": total_vehicles,
            "class_counts": total_counts,
            "average_inference_time_ms": avg_time,
            "results": results,
        }

        logger.info(
            "Batch detection complete: %d images | %d total vehicles detected (Cars: %d, Bikes: %d, Buses: %d, Trucks: %d)",
            len(image_files), total_vehicles, total_counts["car"], total_counts["motorcycle"],
            total_counts["bus"], total_counts["truck"]
        )
        return batch_summary

    def process_video(
        self,
        video_path: Union[Path, str],
        output_path: Optional[Union[Path, str]] = None,
        classes: Optional[List[str]] = None,
        conf: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs YOLO vehicle detection on a traffic video stream frame-by-frame,
        annotates bounding boxes and labels, and writes the output video to disk.
        
        Args:
            video_path: Input traffic video file path.
            output_path: Output video file path. Defaults to output/processed_videos/detected_<name>.mp4.
            classes: Target classes to detect.
            conf: Confidence threshold override.
            max_frames: Maximum number of frames to process (useful for testing).
            
        Returns:
            Dict[str, Any]: Video processing summary statistics.
        """
        src_video = Path(video_path).resolve()
        if not src_video.exists():
            raise FileNotFoundError(f"Source video not found: {src_video}")

        # Resolve output path
        if output_path is None:
            try:
                cfg = load_config()
                out_dir = get_path("processed_videos_dir", cfg)
            except Exception:
                out_dir = Path("output/processed_videos").resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            dest_video = out_dir / f"detected_{src_video.stem}.mp4"
        else:
            dest_video = Path(output_path).resolve()
            dest_video.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(src_video))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {src_video}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        from utils.video_compat import create_compatible_video_writer, finalize_video_file

        writer = create_compatible_video_writer(dest_video, fps, width, height)

        logger.info(
            "Processing video '%s' (%dx%d @ %.1f FPS, %d frames)...",
            src_video.name, width, height, fps, total_frames
        )

        frame_idx = 0
        total_detections = 0
        counts: Dict[str, int] = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        t_start = time.perf_counter()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                detections = self.detect_multi_class(frame, classes=classes, conf=conf)
                total_detections += len(detections)
                for det in detections:
                    cls_name = det["class_name"]
                    counts[cls_name] = counts.get(cls_name, 0) + 1

                annotated = self.annotate_frame(frame, detections)

                # Add timestamp and frame counter overlay
                cv2.putText(
                    annotated,
                    f"Frame: {frame_idx + 1} | Vehicles: {len(detections)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                writer.write(annotated)
                frame_idx += 1

                if max_frames and frame_idx >= max_frames:
                    logger.info("Reached maximum frame limit of %d frames.", max_frames)
                    break
        finally:
            cap.release()
            writer.release()
            finalize_video_file(dest_video)

        elapsed_sec = round(time.perf_counter() - t_start, 2)
        avg_fps = round(frame_idx / elapsed_sec, 2) if elapsed_sec > 0 else 0.0

        summary = {
            "source_video": str(src_video),
            "output_video": str(dest_video),
            "resolution": f"{width}x{height}",
            "frames_processed": frame_idx,
            "total_frames_in_source": total_frames,
            "processing_time_sec": elapsed_sec,
            "average_fps": avg_fps,
            "total_detections": total_detections,
            "cumulative_counts": counts,
        }

        logger.info(
            "Video processing complete: %d frames written to %s in %.2fs (%.1f FPS)",
            frame_idx, dest_video, elapsed_sec, avg_fps
        )
        return summary
