/**
 * VisionPort Vehicle Tracking & Speed Controller (Tasks 12 & 13)
 * Displays ByteTrack persistent tracking IDs, speeds, trajectories, and reports
 */

document.addEventListener("DOMContentLoaded", () => {
  loadTrackingData();
});

async function loadTrackingData() {
  try {
    const data = await api.getVehicles();
    const tracking = data.tracking || {};
    const speed = data.speed || {};

    const uniqueCounts = tracking.unique_vehicle_counts || { total: 0, car: 0, motorcycle: 0, bus: 0, truck: 0 };

    setElText("track-total-unique", uniqueCounts.total);
    setElText("track-cars", uniqueCounts.car);
    setElText("track-motorcycles", uniqueCounts.motorcycle);
    setElText("track-buses", uniqueCounts.bus);
    setElText("track-trucks", uniqueCounts.truck);

    setElText("track-avg-speed", `${speed.average_speed || 0.0} ${speed.unit || "px/s"}`);
    setElText("track-calibrated", speed.calibrated ? "Calibrated (km/h)" : "Uncalibrated (px/s)");

    renderTracksTable(tracking.tracks || []);
  } catch (err) {
    console.warn("Could not load tracking data:", err);
  }
}

function renderTracksTable(tracks) {
  const tableBody = document.getElementById("tracks-table-body");
  if (!tableBody) return;

  if (!tracks || tracks.length === 0) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="6" style="text-align: center; color: var(--text-muted); padding: 2rem;">
          No active tracks recorded yet. Run video tracking or wait for live stream.
        </td>
      </tr>
    `;
    return;
  }

  tableBody.innerHTML = tracks
    .map(
      (t) => `
    <tr>
      <td><strong>#${t.track_id}</strong></td>
      <td>
        <span class="badge badge-cyan">${escapeHtml(t.class_name.toUpperCase())}</span>
      </td>
      <td>Frame ${t.first_frame}</td>
      <td>Frame ${t.last_frame}</td>
      <td>${t.frames_seen} frames</td>
      <td>${(t.last_conf * 100).toFixed(1)}%</td>
    </tr>
  `
    )
    .join("");
}

