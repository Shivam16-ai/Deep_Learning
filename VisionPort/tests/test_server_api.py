"""
Tests for VisionPort Server REST API and Frontend static file integrity.
"""

from pathlib import Path
import pytest

import server


class TestVisionPortServerAPI:
    """Automated test suite for VisionPort backend API and static structure."""

    def test_01_api_health(self):
        result = server.get_health()
        assert result["status"] == "online"
        assert "project" in result and "version" in result["project"]
        assert "yolo_model_available" in result and result["yolo_model_available"] is True

    def test_02_api_dashboard(self):
        result = server.get_dashboard_summary()
        assert "total_vehicles_detected" in result
        assert "current_density" in result
        assert "class_counts" in result
        assert isinstance(result["class_counts"], dict)

    def test_03_api_dataset_images(self):
        result = server.list_dataset_images()
        assert "images" in result
        assert len(result["images"]) >= 5
        first = result["images"][0]
        assert "filename" in first and "url" in first

    def test_04_api_dataset_videos(self):
        result = server.list_dataset_videos()
        assert "videos" in result
        assert len(result["videos"]) >= 1
        first = result["videos"][0]
        assert "filename" in first and "url" in first

    def test_05_api_analytics(self):
        result = server.get_analytics()
        assert "density" in result
        assert "congestion" in result
        assert "counting" in result

    def test_06_api_vehicles(self):
        result = server.get_vehicles()
        assert "tracking" in result
        assert "speed" in result
        assert "unique_vehicle_counts" in result["tracking"]

    def test_07_api_alerts(self):
        result = server.get_alerts()
        assert "alerts" in result
        assert "total_active" in result

    def test_08_api_signals(self):
        result = server.get_signals()
        assert result["status"] in ("RED", "YELLOW", "GREEN")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_09_api_waiting_time(self):
        result = server.get_waiting_time()
        assert "average_waiting_time_sec" in result
        assert "vehicles_in_queue" in result
        assert "configured_queue_zone" in result

    def test_10_api_reports(self):
        result = server.list_reports()
        assert "reports" in result
        assert isinstance(result["reports"], list)

    def test_11_api_settings_get_and_update(self):
        current_cfg = server.get_settings()
        assert "yolo" in current_cfg or "detection" in current_cfg

        update_payload = {
            "yolo": {"confidence_threshold": 0.40},
            "density": {"low_threshold": 3},
        }
        update_res = server.update_settings(update_payload)
        assert update_res["status"] == "saved"

        new_cfg = server.get_settings()
        assert new_cfg["yolo"]["confidence_threshold"] == 0.40

    def test_12_frontend_static_structure_completeness(self):
        frontend_dir = Path(server.FRONTEND_DIR)
        assert (frontend_dir / "index.html").exists()

        # All 10 individual pages
        pages = [
            "monitoring.html",
            "video-analysis.html",
            "detection.html",
            "tracking.html",
            "analytics.html",
            "alerts.html",
            "signals.html",
            "waiting-time.html",
            "reports.html",
            "settings.html",
        ]
        for p in pages:
            page_path = frontend_dir / "pages" / p
            assert page_path.exists(), f"Missing frontend page: {p}"

        # CSS files
        for c in ["style.css", "components.css", "dashboard.css", "responsive.css"]:
            assert (frontend_dir / "css" / c).exists(), f"Missing CSS file: {c}"

        # JS files
        js_files = [
            "api.js",
            "websocket.js",
            "app.js",
            "dashboard.js",
            "monitoring.js",
            "detection.js",
            "tracking.js",
            "analytics.js",
            "alerts.js",
            "signals.js",
            "waiting-time.js",
            "reports.js",
            "settings.js",
        ]
        for j in js_files:
            assert (frontend_dir / "js" / j).exists(), f"Missing JS file: {j}"
