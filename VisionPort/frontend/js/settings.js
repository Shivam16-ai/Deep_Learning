/**
 * VisionPort Settings Controller
 * Handles configuration reading and saving
 */

document.addEventListener("DOMContentLoaded", () => {
  loadSettings();
  setupSettingsHandlers();
});

async function loadSettings() {
  try {
    const config = await api.getSettings();
    populateForm(config);
  } catch (err) {
    console.error("Failed to load settings:", err);
    showToast("Could not load backend settings.", "error");
  }
}

function populateForm(cfg) {
  if (!cfg) return;

  // YOLO detection
  const yolo = cfg.yolo || {};
  setVal("conf-threshold", yolo.confidence_threshold ?? 0.4);
  setVal("conf-threshold-val", yolo.confidence_threshold ?? 0.4);
  setVal("iou-threshold", yolo.iou_threshold ?? 0.45);
  setVal("model-name", yolo.model_name || "yolov8n.pt");

  // Classes
  const targetClasses = (yolo.target_classes || ["car", "motorcycle", "bus", "truck"]).map((c) =>
    c.toLowerCase()
  );
  setCheck("cls-car", targetClasses.includes("car"));
  setCheck("cls-motorcycle", targetClasses.includes("motorcycle"));
  setCheck("cls-bus", targetClasses.includes("bus"));
  setCheck("cls-truck", targetClasses.includes("truck"));

  // Tracking
  const tracking = cfg.tracking || {};
  setVal("track-buffer", tracking.track_buffer ?? 30);
  setVal("track-thresh", tracking.track_thresh ?? 0.5);

  // Speed Estimation
  const speed = cfg.speed_estimation || {};
  setVal("ppm-scale", speed.pixels_per_meter ?? 15.0);
  setVal("speed-fps", speed.fps ?? 30);

  // Traffic Density
  const density = cfg.density || {};
  setVal("thresh-low", density.low_threshold ?? 3);
  setVal("thresh-medium", density.medium_threshold ?? 7);

  // Waiting Time / Queue Zone
  const queue = cfg.waiting_time || {};
  setVal("queue-ymin", (queue.queue_zone?.y_min ?? 0.6) * 100);
  setVal("queue-ymax", (queue.queue_zone?.y_max ?? 0.95) * 100);
}

function setupSettingsHandlers() {
  const confSlider = document.getElementById("conf-threshold");
  const confVal = document.getElementById("conf-threshold-val");
  if (confSlider && confVal) {
    confSlider.addEventListener("input", (e) => {
      confVal.value = e.target.value;
    });
    confVal.addEventListener("change", (e) => {
      confSlider.value = e.target.value;
    });
  }

  const form = document.getElementById("settings-form");
  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      await saveSettings();
    });
  }

  const resetBtn = document.getElementById("reset-settings-btn");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      loadSettings();
      showToast("Reset to saved configuration", "info");
    });
  }
}

async function saveSettings() {
  const targetClasses = [];
  if (getCheck("cls-car")) targetClasses.push("car");
  if (getCheck("cls-motorcycle")) targetClasses.push("motorcycle");
  if (getCheck("cls-bus")) targetClasses.push("bus");
  if (getCheck("cls-truck")) targetClasses.push("truck");

  const payload = {
    yolo: {
      model_name: getVal("model-name"),
      confidence_threshold: parseFloat(getVal("conf-threshold")),
      iou_threshold: parseFloat(getVal("iou-threshold")),
      target_classes: targetClasses,
    },
    tracking: {
      track_buffer: parseInt(getVal("track-buffer"), 10),
      track_thresh: parseFloat(getVal("track-thresh")),
    },
    speed_estimation: {
      pixels_per_meter: parseFloat(getVal("ppm-scale")),
      fps: parseInt(getVal("speed-fps"), 10),
    },
    density: {
      low_threshold: parseInt(getVal("thresh-low"), 10),
      medium_threshold: parseInt(getVal("thresh-medium"), 10),
    },
    waiting_time: {
      queue_zone: {
        y_min: parseFloat(getVal("queue-ymin")) / 100,
        y_max: parseFloat(getVal("queue-ymax")) / 100,
      },
    },
  };

  const saveBtn = document.getElementById("save-settings-btn");
  if (saveBtn) {
    saveBtn.disabled = true;
    saveBtn.innerHTML = `<span>Saving...</span>`;
  }

  try {
    const res = await api.updateSettings(payload);
    showToast(res.message || "Settings saved successfully", "success");
  } catch (err) {
    console.error("Save settings error:", err);
    showToast("Failed to save settings: " + err.message, "error");
  } finally {
    if (saveBtn) {
      saveBtn.disabled = false;
      saveBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path>
          <polyline points="17 21 17 13 7 13 7 21"></polyline>
          <polyline points="7 3 7 8 15 8"></polyline>
        </svg>
        Save Configuration
      `;
    }
  }
}

function setVal(id, val) {
  const el = document.getElementById(id);
  if (el) el.value = val;
}

function getVal(id) {
  const el = document.getElementById(id);
  return el ? el.value : "";
}

function setCheck(id, checked) {
  const el = document.getElementById(id);
  if (el) el.checked = !!checked;
}

function getCheck(id) {
  const el = document.getElementById(id);
  return el ? el.checked : false;
}

