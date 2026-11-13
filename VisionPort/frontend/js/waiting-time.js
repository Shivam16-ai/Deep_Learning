/**
 * VisionPort Traffic Waiting Time Controller (Task 15)
 * Displays vehicle queue metrics and waiting time statistics
 */

document.addEventListener("DOMContentLoaded", () => {
  loadWaitingTime();
});

async function loadWaitingTime() {
  try {
    const data = await api.getWaitingTime();

    setElText("wait-avg-time", `${data.average_waiting_time_sec?.toFixed(1) || 0}s`);
    setElText("wait-max-time", `${data.max_waiting_time_sec?.toFixed(1) || 0}s`);
    setElText("wait-queue-count", data.vehicles_in_queue || 0);

    const zone = data.configured_queue_zone || {};
    setElText("wait-zone-range", `Y: [${zone.y_min * 100}% - ${zone.y_max * 100}%]`);

    renderQueueTable(data.queue_records || []);
  } catch (err) {
    console.error("Failed to load waiting time:", err);
  }
}

function renderQueueTable(records) {
  const tableBody = document.getElementById("queue-table-body");
  if (!tableBody) return;

  if (!records || records.length === 0) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="4" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">
          No vehicles currently queued in waiting zone.
        </td>
      </tr>
    `;
    return;
  }

  tableBody.innerHTML = records
    .map(
      (r) => `
    <tr>
      <td><strong>#${r.track_id}</strong></td>
      <td><span class="badge badge-cyan">${escapeHtml(r.category.toUpperCase())}</span></td>
      <td>${r.waiting_time_sec.toFixed(1)}s</td>
      <td><span class="badge badge-medium">${r.status}</span></td>
    </tr>
  `
    )
    .join("");
}

