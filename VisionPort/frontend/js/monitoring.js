/**
 * VisionPort Live Traffic Monitoring Controller
 * Real-time WebSocket video stream & telemetry HUD
 */

let isMonitoring = false;
let isPaused = false;
let overlayEnabled = true;

document.addEventListener("DOMContentLoaded", () => {
  initMonitoringControls();
  initWebSocketStream();
  populateVideoSources();
});

function initMonitoringControls() {
  const startBtn = document.getElementById("btn-start-monitor");
  const stopBtn = document.getElementById("btn-stop-monitor");
  const pauseBtn = document.getElementById("btn-pause-monitor");
  const fullscreenBtn = document.getElementById("btn-fullscreen");
  const overlayCheckbox = document.getElementById("toggle-overlay");

  if (startBtn) {
    startBtn.addEventListener("click", handleStartMonitoring);
  }
  if (stopBtn) {
    stopBtn.addEventListener("click", handleStopMonitoring);
  }
  if (pauseBtn) {
    pauseBtn.addEventListener("click", handleTogglePause);
  }
  if (fullscreenBtn) {
    fullscreenBtn.addEventListener("click", handleToggleFullscreen);
  }
  if (overlayCheckbox) {
    overlayCheckbox.addEventListener("change", (e) => {
      overlayEnabled = e.target.checked;
    });
  }
}

async function populateVideoSources() {
  const selectEl = document.getElementById("source-select");
  if (!selectEl) return;

  try {
    const data = await api.getDatasetVideos();
    if (data && data.videos && data.videos.length > 0) {
      data.videos.forEach((vid) => {
        const opt = document.createElement("option");
        opt.value = vid.filename;
        opt.textContent = `Video: ${vid.filename} (${vid.size_mb} MB)`;
        selectEl.appendChild(opt);
      });
    }
  } catch (err) {
    console.warn("Could not load dataset videos:", err);
  }
}

async function handleStartMonitoring() {
  const sourceSelect = document.getElementById("source-select");
  const selectedSource = sourceSelect ? sourceSelect.value : "sample_traffic.mp4";
  const sourceType = selectedSource.startsWith("cam") ? "camera" : "video";

  try {
    await api.startMonitoring(sourceType, selectedSource);
    wsClient.connect();

    isMonitoring = true;
    isPaused = false;
    updateButtonStates();

    const placeholder = document.getElementById("stream-placeholder");
    const streamImg = document.getElementById("live-stream-img");
    if (placeholder) placeholder.style.display = "none";
    if (streamImg) streamImg.style.display = "block";

    showToast("Live monitoring started", "info");
  } catch (err) {
    showToast(`Failed to start: ${err.message}`, "danger");
  }
}

async function handleStopMonitoring() {
  try {
    await api.stopMonitoring();
    isMonitoring = false;
    isPaused = false;
    updateButtonStates();

    const placeholder = document.getElementById("stream-placeholder");
    const streamImg = document.getElementById("live-stream-img");
    if (streamImg) streamImg.style.display = "none";
    if (placeholder) placeholder.style.display = "flex";

    showToast("Live monitoring stopped", "info");
  } catch (err) {
    showToast(`Failed to stop: ${err.message}`, "danger");
  }
}

function handleTogglePause() {
  isPaused = !isPaused;
  const pauseBtn = document.getElementById("btn-pause-monitor");
  if (pauseBtn) {
    pauseBtn.innerHTML = isPaused
      ? `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Resume`
      : `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg> Pause`;
  }
}

function handleToggleFullscreen() {
  const container = document.getElementById("video-monitor-container");
  if (!container) return;

  if (!document.fullscreenElement) {
    container.requestFullscreen().catch((err) => {
      console.warn("Fullscreen error:", err);
    });
  } else {
    document.exitFullscreen();
  }
}

function updateButtonStates() {
  const startBtn = document.getElementById("btn-start-monitor");
  const stopBtn = document.getElementById("btn-stop-monitor");
  const pauseBtn = document.getElementById("btn-pause-monitor");

  if (startBtn) startBtn.disabled = isMonitoring;
  if (stopBtn) stopBtn.disabled = !isMonitoring;
  if (pauseBtn) pauseBtn.disabled = !isMonitoring;
}

function initWebSocketStream() {
  const streamImg = document.getElementById("live-stream-img");

  wsClient.on("telemetry", (data) => {
    if (!isMonitoring || isPaused) return;

    // Render frame
    if (streamImg && data.frame_base64) {
      streamImg.src = data.frame_base64;
    }

    // Update telemetry side panel & HUD
    setElText("hud-fps", "16 FPS");
    setElText("hud-frame", `Frame #${data.frame_idx || 0}`);

    setElText("monitor-total-vehicles", data.total_vehicles || 0);
    setElText("monitor-cars", data.counts?.car || 0);
    setElText("monitor-motorcycles", data.counts?.motorcycle || 0);
    setElText("monitor-buses", data.counts?.bus || 0);
    setElText("monitor-trucks", data.counts?.truck || 0);

    const densityBadge = document.getElementById("monitor-density");
    if (densityBadge) {
      const dens = data.density_level || "LOW";
      densityBadge.textContent = dens;
      densityBadge.className = `badge badge-${dens.toLowerCase()}`;
    }

    const congBadge = document.getElementById("monitor-congestion");
    if (congBadge) {
      const cong = data.congestion_status || "NORMAL";
      congBadge.textContent = cong;
      congBadge.className = `badge ${cong === "CONGESTED" ? "badge-high" : "badge-low"}`;
    }

    const signalBadge = document.getElementById("monitor-signal");
    if (signalBadge) {
      signalBadge.textContent = data.signal_status || "GREEN";
      signalBadge.className = `badge ${data.signal_status === "RED" ? "badge-high" : "badge-low"}`;
    }
  });

  wsClient.on("connect", () => {
    setElText("ws-connection-status", "Live Stream Connected");
  });

  wsClient.on("disconnect", () => {
    setElText("ws-connection-status", "Stream Disconnected");
  });
}

