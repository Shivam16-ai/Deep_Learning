/**
 * VisionPort Overview Dashboard Controller
 * Connects to /api/dashboard and renders charts with Chart.js
 */

let categoryChartInstance = null;
let densityChartInstance = null;
let activityChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
  loadDashboardData();
  // Refresh every 15 seconds
  setInterval(loadDashboardData, 15000);
});

async function loadDashboardData() {
  const loadingOverlay = document.getElementById("dashboard-loading");
  const errorBanner = document.getElementById("dashboard-error");

  try {
    const data = await api.getDashboard();
    if (errorBanner) errorBanner.style.display = "none";
    if (loadingOverlay) loadingOverlay.style.display = "none";

    renderStatCards(data);
    renderCategoryChart(data.class_counts);
    renderDensityChart(data.density_history);
    renderActivityChart(data.class_counts);
    renderRecentAlerts(data.recent_congestion_events);
    renderRecentActivity(data.recent_activity);
  } catch (err) {
    console.error("Failed to load dashboard data:", err);
    if (errorBanner) {
      errorBanner.style.display = "flex";
      errorBanner.querySelector(".error-text").textContent =
        `Backend disconnected: ${err.message}. Showing last known or empty state.`;
    }
    if (loadingOverlay) loadingOverlay.style.display = "none";
  }
}

function renderStatCards(data) {
  // Update stat values
  setElText("stat-total-vehicles", data.total_vehicles_detected || 0);
  setElText("stat-cars", data.class_counts?.car || 0);
  setElText("stat-motorcycles", data.class_counts?.motorcycle || 0);
  setElText("stat-buses", data.class_counts?.bus || 0);
  setElText("stat-trucks", data.class_counts?.truck || 0);

  // Density
  const densityEl = document.getElementById("stat-density");
  if (densityEl) {
    const level = data.current_density || "LOW";
    densityEl.textContent = level;
    densityEl.className = `stat-card-value badge-${level.toLowerCase()}`;
  }

  // Active alerts
  setElText("stat-alerts", data.active_congestion_alerts || 0);

  // Speed
  const speedVal = data.average_speed_kmh || 0;
  const unit = data.speed_unit || "km/h";
  const calText = data.speed_calibrated ? "" : " (Uncalibrated)";
  setElText("stat-speed", `${speedVal.toFixed(1)} ${unit}${calText}`);
}

function renderCategoryChart(classCounts) {
  const ctx = document.getElementById("categoryChart");
  if (!ctx || !window.Chart) return;

  const counts = classCounts || { car: 0, motorcycle: 0, bus: 0, truck: 0 };
  const labels = ["Cars", "Motorcycles", "Buses", "Trucks"];
  const values = [counts.car || 0, counts.motorcycle || 0, counts.bus || 0, counts.truck || 0];

  if (categoryChartInstance) {
    categoryChartInstance.data.datasets[0].data = values;
    categoryChartInstance.update();
    return;
  }

  categoryChartInstance = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: labels,
      datasets: [
        {
          data: values,
          backgroundColor: ["#3b82f6", "#06b6d4", "#10b981", "#f59e0b"],
          borderWidth: 2,
          borderColor: "#131c2e",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#94a3b8", font: { size: 12 } },
        },
      },
      cutout: "68%",
    },
  });
}

function renderDensityChart(transitions) {
  const ctx = document.getElementById("densityChart");
  if (!ctx || !window.Chart) return;

  const labels = [];
  const countData = [];

  if (transitions && transitions.length > 0) {
    transitions.forEach((t) => {
      labels.push(`Frame ${t.frame || 0}`);
      countData.push(t.vehicle_count || 0);
    });
  } else {
    // Empty state labels
    for (let i = 1; i <= 6; i++) {
      labels.push(`T-${i}`);
      countData.push(0);
    }
  }

  if (densityChartInstance) {
    densityChartInstance.data.labels = labels;
    densityChartInstance.data.datasets[0].data = countData;
    densityChartInstance.update();
    return;
  }

  densityChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Vehicle Volume",
          data: countData,
          borderColor: "#06b6d4",
          backgroundColor: "rgba(6, 182, 212, 0.1)",
          fill: true,
          tension: 0.35,
          borderWidth: 2,
          pointRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: { color: "#1e293b" },
          ticks: { color: "#64748b" },
        },
        x: {
          grid: { color: "#1e293b" },
          ticks: { color: "#64748b" },
        },
      },
    },
  });
}

function renderActivityChart(classCounts) {
  const ctx = document.getElementById("activityChart");
  if (!ctx || !window.Chart) return;

  const counts = classCounts || { car: 0, motorcycle: 0, bus: 0, truck: 0 };
  const labels = ["Cars", "Motorcycles", "Buses", "Trucks"];
  const values = [counts.car || 0, counts.motorcycle || 0, counts.bus || 0, counts.truck || 0];

  if (activityChartInstance) {
    activityChartInstance.data.datasets[0].data = values;
    activityChartInstance.update();
    return;
  }

  activityChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Detected Vehicles",
          data: values,
          backgroundColor: ["#3b82f6", "#06b6d4", "#10b981", "#f59e0b"],
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: { color: "#1e293b" },
          ticks: { color: "#64748b" },
        },
        x: {
          grid: { display: false },
          ticks: { color: "#64748b" },
        },
      },
    },
  });
}

function renderRecentAlerts(events) {
  const listEl = document.getElementById("recent-alerts-list");
  if (!listEl) return;

  if (!events || events.length === 0) {
    listEl.innerHTML = `
      <div class="empty-state" style="padding: 1.5rem 0;">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
        <p>No active or recent congestion alerts.</p>
      </div>
    `;
    return;
  }

  listEl.innerHTML = events
    .map(
      (ev) => `
    <div class="activity-item">
      <div class="activity-icon" style="background-color: rgba(239, 68, 68, 0.15); color: #ef4444;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
          <line x1="12" y1="9" x2="12" y2="13"></line>
          <line x1="12" y1="17" x2="12.01" y2="17"></line>
        </svg>
      </div>
      <div class="activity-content">
        <div class="activity-title">Sustained Congestion (Event #${ev.event_id || 1})</div>
        <div class="activity-meta">Duration: ${ev.duration_sec?.toFixed(1) || 0}s (${ev.duration_frames || 0} frames)</div>
      </div>
      <span class="badge badge-high">${ev.peak_density_level || "HIGH"}</span>
    </div>
  `
    )
    .join("");
}

function renderRecentActivity(activity) {
  const feedEl = document.getElementById("recent-activity-feed");
  if (!feedEl) return;

  if (!activity || activity.length === 0) {
    feedEl.innerHTML = `<p class="card-subtitle">No recent activities recorded.</p>`;
    return;
  }

  feedEl.innerHTML = activity
    .map(
      (act) => `
    <div class="activity-item">
      <div class="activity-icon">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
        </svg>
      </div>
      <div class="activity-content">
        <div class="activity-title">${escapeHtml(act.action)}</div>
        <div class="activity-meta">${escapeHtml(act.file)} &bull; ${escapeHtml(act.time)}</div>
      </div>
      <span class="badge badge-low">${escapeHtml(act.status)}</span>
    </div>
  `
    )
    .join("");
}

function setElText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

