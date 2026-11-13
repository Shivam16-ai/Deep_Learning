/**
 * VisionPort Reports Controller
 * Lists generated reports and provides download links
 */

document.addEventListener("DOMContentLoaded", () => {
  loadReports();
});

async function loadReports() {
  try {
    const data = await api.getReports();
    renderReportsTable(data.reports || []);
  } catch (err) {
    console.error("Failed to load reports:", err);
  }
}

function renderReportsTable(reports) {
  const tableBody = document.getElementById("reports-table-body");
  if (!tableBody) return;

  if (!reports || reports.length === 0) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="5" style="text-align: center; color: var(--text-muted); padding: 2rem;">
          No reports generated yet. Execute detection or analytics tasks to generate outputs.
        </td>
      </tr>
    `;
    return;
  }

  tableBody.innerHTML = reports
    .map(
      (rep) => `
    <tr>
      <td><strong>${escapeHtml(rep.name)}</strong></td>
      <td>
        <span class="badge ${rep.format === "JSON" ? "badge-cyan" : rep.format === "CSV" ? "badge-low" : "badge-medium"}">
          ${rep.format}
        </span>
      </td>
      <td>${rep.size_kb} KB</td>
      <td>${rep.modified}</td>
      <td>
        <a href="${rep.download_url}" target="_blank" download class="btn btn-secondary" style="padding: 0.35rem 0.75rem; font-size: 0.75rem;">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="7 10 12 15 17 10"></polyline>
            <line x1="12" y1="15" x2="12" y2="3"></line>
          </svg>
          Download
        </a>
      </td>
    </tr>
  `
    )
    .join("");
}

