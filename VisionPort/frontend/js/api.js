/**
 * VisionPort API Client
 * Enterprise REST API connector for VisionPort Deep Learning Backend
 */

const API_CONFIG = {
  // Automatically detects server origin or falls back to standard port 8000
  BASE_URL: window.location.origin.includes("http")
    ? window.location.origin
    : "http://127.0.0.1:8000",
  TIMEOUT_MS: 45000,
};

class VisionPortAPI {
  constructor(baseUrl = API_CONFIG.BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  /**
   * Internal fetch wrapper with timeout and error extraction
   */
  async _fetch(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), API_CONFIG.TIMEOUT_MS);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (!response.ok) {
        let errDetail = response.statusText;
        try {
          const errData = await response.json();
          errDetail = errData.detail || JSON.stringify(errData);
        } catch (_) {}
        throw new Error(`API Error [${response.status}]: ${errDetail}`);
      }

      return await response.json();
    } catch (err) {
      clearTimeout(timer);
      if (err.name === "AbortError") {
        throw new Error(`Request timed out after ${API_CONFIG.TIMEOUT_MS / 1000}s`);
      }
      throw err;
    }
  }

  // System & Health
  async getHealth() {
    return this._fetch("/api/health");
  }

  // Dashboard Overview
  async getDashboard() {
    return this._fetch("/api/dashboard");
  }

  // Dataset Media
  async getDatasetImages() {
    return this._fetch("/api/dataset/images");
  }

  async getDatasetVideos() {
    return this._fetch("/api/dataset/videos");
  }

  // Task 6-8: Vehicle Detection on Image
  async detectImage({ file, datasetImage, conf = 0.40 }) {
    if (file) {
      const formData = new FormData();
      formData.append("file", file);
      return this._fetch(`/api/detection/image?conf=${conf}`, {
        method: "POST",
        body: formData,
      });
    } else if (datasetImage) {
      return this._fetch(`/api/detection/image?dataset_image=${encodeURIComponent(datasetImage)}&conf=${conf}`, {
        method: "POST",
      });
    }
    throw new Error("Must provide either a File object or datasetImage name");
  }

  // Video Analysis (Tasks 6-13 integrated)
  async analyzeVideo({ file, datasetVideo, conf = 0.40, maxFrames = 120 }) {
    const params = new URLSearchParams({
      conf: conf.toString(),
      max_frames: maxFrames.toString(),
      track: "true",
    });

    if (file) {
      const formData = new FormData();
      formData.append("file", file);
      return this._fetch(`/api/detection/video?${params}`, {
        method: "POST",
        body: formData,
      });
    } else if (datasetVideo) {
      params.append("dataset_video", datasetVideo);
      return this._fetch(`/api/detection/video?${params}`, {
        method: "POST",
      });
    }
    throw new Error("Must provide either a File object or datasetVideo name");
  }

  // Live Monitoring Control
  async startMonitoring(sourceType = "video", sourceVideo = "sample_traffic.mp4") {
    const params = new URLSearchParams({
      source_type: sourceType,
      source_video: sourceVideo,
    });
    return this._fetch(`/api/monitoring/start?${params}`, { method: "POST" });
  }

  async stopMonitoring() {
    return this._fetch("/api/monitoring/stop", { method: "POST" });
  }

  // Analytics & Aggregates
  async getAnalytics() {
    return this._fetch("/api/analytics");
  }

  async getVehicles() {
    return this._fetch("/api/vehicles");
  }

  async getAlerts() {
    return this._fetch("/api/alerts");
  }

  async getSignals() {
    return this._fetch("/api/signals");
  }

  async getWaitingTime() {
    return this._fetch("/api/waiting-time");
  }

  // Reports
  async getReports() {
    return this._fetch("/api/reports");
  }

  // Configuration / Settings
  async getSettings() {
    return this._fetch("/api/settings");
  }

  async updateSettings(configObj) {
    return this._fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(configObj),
    });
  }
}

// Export singleton instance
const api = new VisionPortAPI();

