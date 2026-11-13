/**
 * VisionPort Main Application Controller
 * Handles global navigation, system status monitoring, clock, and toasts
 */

document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initSystemHealthMonitor();
  initMobileNav();
  highlightActiveNav();
});

/**
 * Live topbar clock
 */
function initClock() {
  const clockEl = document.getElementById("topbar-datetime");
  if (!clockEl) return;

  function update() {
    const now = new Date();
    clockEl.textContent = now.toLocaleString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  }
  update();
  setInterval(update, 1000);
}

/**
 * Periodically verifies backend health and updates connection status
 */
function initSystemHealthMonitor() {
  const badgeEl = document.getElementById("backend-status-badge");
  const textEl = document.getElementById("backend-status-text");

  async function check() {
    try {
      const health = await api.getHealth();
      if (health && health.status === "online") {
        if (badgeEl) {
          badgeEl.className = "status-badge online";
        }
        if (textEl) {
          textEl.textContent = "Backend Connected";
        }
      } else {
        markOffline();
      }
    } catch (err) {
      markOffline();
    }
  }

  function markOffline() {
    if (badgeEl) {
      badgeEl.className = "status-badge offline";
    }
    if (textEl) {
      textEl.textContent = "Backend Offline";
    }
  }

  check();
  setInterval(check, 10000); // Check every 10s
}

/**
 * Mobile responsive sidebar drawer
 */
function initMobileNav() {
  const toggleBtn = document.getElementById("mobile-toggle");
  const sidebar = document.querySelector(".sidebar");
  const overlay = document.querySelector(".sidebar-overlay");

  if (!toggleBtn || !sidebar) return;

  toggleBtn.addEventListener("click", () => {
    sidebar.classList.toggle("open");
    if (overlay) overlay.classList.toggle("active");
  });

  if (overlay) {
    overlay.addEventListener("click", () => {
      sidebar.classList.remove("open");
      overlay.classList.remove("active");
    });
  }
}

/**
 * Automatically highlights the current active link in the sidebar
 */
function highlightActiveNav() {
  const path = window.location.pathname;
  const navLinks = document.querySelectorAll(".sidebar-nav .nav-link");

  navLinks.forEach((link) => {
    const href = link.getAttribute("href");
    if (!href) return;

    if (
      (path.endsWith("/") || path.endsWith("index.html")) &&
      (href === "index.html" || href === "/index.html" || href === "/")
    ) {
      link.classList.add("active");
    } else if (path.includes(href) && href !== "index.html" && href !== "/") {
      link.classList.add("active");
    } else {
      link.classList.remove("active");
    }
  });
}

/**
 * Toast notifications
 */
function showToast(message, type = "info") {
  let container = document.querySelector(".toast-container");
  if (!container) {
    container = document.createElement("div");
    container.className = "toast-container";
    document.body.appendChild(container);
  }

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span class="toast-message">${escapeHtml(message)}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

