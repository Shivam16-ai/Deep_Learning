/**
 * VisionPort Traffic Signal Controller (Task 14)
 * Visualizes Red/Yellow/Green traffic signal states and historical transitions
 */

document.addEventListener("DOMContentLoaded", () => {
  loadSignalStatus();
});

async function loadSignalStatus() {
  try {
    const data = await api.getSignals();
    updateSignalLamp(data.status);

    setElText("signal-status-text", data.status);
    setElText("signal-confidence", `${(data.confidence * 100).toFixed(1)}%`);

    renderSignalHistory(data.change_history || []);
  } catch (err) {
    console.error("Failed to load signals:", err);
  }
}

function updateSignalLamp(state) {
  const lampRed = document.getElementById("lamp-red");
  const lampYellow = document.getElementById("lamp-yellow");
  const lampGreen = document.getElementById("lamp-green");

  if (lampRed) lampRed.classList.remove("active");
  if (lampYellow) lampYellow.classList.remove("active");
  if (lampGreen) lampGreen.classList.remove("active");

  if (state === "RED" && lampRed) lampRed.classList.add("active");
  else if (state === "YELLOW" && lampYellow) lampYellow.classList.add("active");
  else if (state === "GREEN" && lampGreen) lampGreen.classList.add("active");
}

function renderSignalHistory(history) {
  const tableBody = document.getElementById("signal-history-table-body");
  if (!tableBody) return;

  if (!history || history.length === 0) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="4" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">
          No signal transitions recorded.
        </td>
      </tr>
    `;
    return;
  }

  tableBody.innerHTML = history
    .map(
      (h) => `
    <tr>
      <td>${h.timestamp}</td>
      <td><span class="badge ${h.from === "RED" ? "badge-high" : h.from === "YELLOW" ? "badge-medium" : "badge-low"}">${h.from}</span></td>
      <td>&rarr;</td>
      <td><span class="badge ${h.to === "RED" ? "badge-high" : h.to === "YELLOW" ? "badge-medium" : "badge-low"}">${h.to}</span></td>
    </tr>
  `
    )
    .join("");
}

