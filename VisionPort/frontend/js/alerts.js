/**
 * VisionPort Congestion Alerts Controller
 * Displays active and resolved congestion alarms
 */

document.addEventListener("DOMContentLoaded", () => {
  loadAlerts();
});

async function loadAlerts() {
  try {
    const data = await api.getAlerts();
    const alerts = data.alerts || [];

    setElText("active-alert-count", data.total_active || 0);
    setElText("total-alert-count", alerts.length);

    renderAlertsTable(alerts);
  } catch (err) {
    console.error("Failed to load alerts:", err);
  }
}

function renderAlertsTable(alerts) {
  const tableBody = document.getElementById("alerts-table-body");
  if (!tableBody) return;

  if (!alerts || alerts.length === 0) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="6" style="text-align: center; color: var(--text-muted); padding: 2rem;">
          No active or logged congestion alarms.
        </td>
      </tr>
    `;
    return;
  }

  tableBody.innerHTML = alerts
    .map(
      (a) => `
    <tr>
      <td><strong>${a.id}</strong></td>
      <td><span class="badge ${a.severity === "HIGH" ? "badge-high" : "badge-medium"}">${a.severity}</span></td>
      <td>${a.type}</td>
      <td>${a.start_time}</td>
      <td>${a.duration}</td>
      <td><span class="badge ${a.status === "Active" ? "badge-high" : "badge-low"}">${a.status}</span></td>
    </tr>
  `
    )
    .join("");
}

