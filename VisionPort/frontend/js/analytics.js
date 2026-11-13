/**
 * VisionPort Traffic Analytics Controller (Tasks 9-11)
 * Displays counting breakdowns, density level transitions, and congestion monitoring
 */

let analyticsTrendChart = null;

document.addEventListener("DOMContentLoaded", () => {
  loadAnalyticsData();
});

async function loadAnalyticsData() {
  try {
    const data = await api.getAnalytics();
    const counting = data.counting || {};
    const density = data.density || {};
    const congestion = data.congestion || {};

    // Counting
    setElText("analytic-total", counting.total_vehicles || 0);
    setElText("analytic-cars", counting.counts?.car || 0);
    setElText("analytic-motorcycles", counting.counts?.motorcycle || 0);
    setElText("analytic-buses", counting.counts?.bus || 0);
    setElText("analytic-trucks", counting.counts?.truck || 0);

    // Density
    const level = density.density_level || "LOW";
    const densBadge = document.getElementById("analytic-density-tier");
    if (densBadge) {
      densBadge.textContent = level;
      densBadge.className = `badge badge-${level.toLowerCase()}`;
    }

    setElText("analytic-occupancy", `${density.occupancy_percent || 0.0}%`);
    setElText("analytic-score", `${(density.density_score || 0.0).toFixed(3)} / 1.000`);

    // Congestion
    setElText("analytic-congestion-events", congestion.total_congestion_events || 0);
    setElText("analytic-congestion-status", congestion.currently_congested ? "CONGESTED" : "NORMAL");

    renderStateTransitions(density.state_transitions || []);
    renderCongestionLog(congestion.congestion_events || []);
    renderAnalyticsChart(density.state_transitions || []);
  } catch (err) {
    console.error("Error loading analytics:", err);
  }
}

function renderStateTransitions(transitions) {
  const tableBody = document.getElementById("transitions-table-body");
  if (!tableBody) return;

  if (!transitions || transitions.length === 0) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="5" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">
          No state transitions recorded.
        </td>
      </tr>
    `;
    return;
  }

  tableBody.innerHTML = transitions
    .map(
      (t) => `
    <tr>
      <td>Frame ${t.frame}</td>
      <td>${t.timestamp_sec?.toFixed(1) || 0}s</td>
      <td><span class="badge badge-neutral">${t.from_level}</span></td>
      <td>&rarr;</td>
      <td><span class="badge badge-${t.to_level.toLowerCase()}">${t.to_level}</span> (${t.vehicle_count} veh)</td>
    </tr>
  `
    )
    .join("");
}

function renderCongestionLog(events) {
  const listEl = document.getElementById("congestion-events-container");
  if (!listEl) return;

  if (!events || events.length === 0) {
    listEl.innerHTML = `
      <div class="empty-state" style="padding: 2rem 0;">
        <p>No sustained congestion events detected.</p>
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
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
      </div>
      <div class="activity-content">
        <div class="activity-title">Congestion Event #${ev.event_id}</div>
        <div class="activity-meta">
          Started frame ${ev.start_frame} (${ev.start_time_sec?.toFixed(1)}s) &bull; 
          Duration: ${ev.duration_sec?.toFixed(1)}s (${ev.duration_frames} frames)
        </div>
      </div>
      <span class="badge badge-high">${ev.peak_density_level}</span>
    </div>
  `
    )
    .join("");
}

function renderAnalyticsChart(transitions) {
  const ctx = document.getElementById("analyticsTrendChart");
  if (!ctx || !window.Chart) return;

  const labels = transitions.map((t) => `F${t.frame}`);
  const data = transitions.map((t) => t.vehicle_count);

  if (analyticsTrendChart) {
    analyticsTrendChart.data.labels = labels;
    analyticsTrendChart.data.datasets[0].data = data;
    analyticsTrendChart.update();
    return;
  }

  analyticsTrendChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Vehicle Volume per State Change",
          data: data,
          backgroundColor: "#3b82f6",
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, grid: { color: "#1e293b" }, ticks: { color: "#94a3b8" } },
        x: { grid: { display: false }, ticks: { color: "#94a3b8" } },
      },
    },
  });
}

