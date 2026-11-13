/**
 * VisionPort Vehicle Detection Controller (Tasks 6-8)
 * Uploads images, runs YOLOv8 vehicle detection, compares original and annotated output
 */

let selectedFile = null;
let selectedDatasetImage = null;

document.addEventListener("DOMContentLoaded", () => {
  initImageInputs();
  populateDatasetImages();
  initConfidenceSlider();
});

function initConfidenceSlider() {
  const slider = document.getElementById("conf-slider");
  const valueDisplay = document.getElementById("conf-value");
  if (!slider || !valueDisplay) return;

  slider.addEventListener("input", (e) => {
    valueDisplay.textContent = `${parseFloat(e.target.value).toFixed(2)}`;
  });
}

function initImageInputs() {
  const fileInput = document.getElementById("image-upload-input");
  const dropZone = document.getElementById("upload-dropzone");
  const detectBtn = document.getElementById("btn-run-detection");

  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
      }
    });
  }

  if (dropZone) {
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.style.borderColor = "var(--accent-blue)";
    });
    dropZone.addEventListener("dragleave", () => {
      dropZone.style.borderColor = "var(--border-default)";
    });
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.style.borderColor = "var(--border-default)";
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });
  }

  if (detectBtn) {
    detectBtn.addEventListener("click", runDetection);
  }
}

function handleFileSelect(file) {
  selectedFile = file;
  selectedDatasetImage = null;

  // Uncheck dataset dropdown if active
  const datasetSelect = document.getElementById("dataset-image-select");
  if (datasetSelect) datasetSelect.value = "";

  const filenameEl = document.getElementById("selected-filename");
  if (filenameEl) filenameEl.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;

  // Preview original image immediately
  const reader = new FileReader();
  reader.onload = (e) => {
    const origImg = document.getElementById("original-image-preview");
    if (origImg) {
      origImg.src = e.target.result;
      origImg.style.display = "block";
    }
    const origPlaceholder = document.getElementById("orig-placeholder");
    if (origPlaceholder) origPlaceholder.style.display = "none";
  };
  reader.readAsDataURL(file);

  const detectBtn = document.getElementById("btn-run-detection");
  if (detectBtn) detectBtn.disabled = false;
}

async function populateDatasetImages() {
  const selectEl = document.getElementById("dataset-image-select");
  if (!selectEl) return;

  try {
    const data = await api.getDatasetImages();
    if (data && data.images) {
      data.images.forEach((img) => {
        const opt = document.createElement("option");
        opt.value = img.filename;
        opt.textContent = `${img.filename} (${img.size_kb} KB)`;
        selectEl.appendChild(opt);
      });

      selectEl.addEventListener("change", (e) => {
        if (e.target.value) {
          selectedDatasetImage = e.target.value;
          selectedFile = null;

          const filenameEl = document.getElementById("selected-filename");
          if (filenameEl) filenameEl.textContent = `Selected: ${selectedDatasetImage}`;

          // Preview original dataset image
          const origImg = document.getElementById("original-image-preview");
          if (origImg) {
            origImg.src = `/dataset/images/${selectedDatasetImage}`;
            origImg.style.display = "block";
          }
          const origPlaceholder = document.getElementById("orig-placeholder");
          if (origPlaceholder) origPlaceholder.style.display = "none";

          const detectBtn = document.getElementById("btn-run-detection");
          if (detectBtn) detectBtn.disabled = false;
        }
      });
    }
  } catch (err) {
    console.warn("Could not load dataset images:", err);
  }
}

async function runDetection() {
  const detectBtn = document.getElementById("btn-run-detection");
  const loadingIndicator = document.getElementById("detection-loading");
  const confSlider = document.getElementById("conf-slider");
  const conf = confSlider ? parseFloat(confSlider.value) : 0.40;

  if (!selectedFile && !selectedDatasetImage) {
    showToast("Please upload an image or choose one from dataset", "warning");
    return;
  }

  try {
    if (detectBtn) detectBtn.disabled = true;
    if (loadingIndicator) loadingIndicator.style.display = "flex";

    const result = await api.detectImage({
      file: selectedFile,
      datasetImage: selectedDatasetImage,
      conf: conf,
    });

    if (result && result.success) {
      // Render annotated image
      const annotatedImg = document.getElementById("annotated-image-preview");
      if (annotatedImg) {
        annotatedImg.src = result.annotated_image_url;
        annotatedImg.style.display = "block";
      }
      const annotPlaceholder = document.getElementById("annot-placeholder");
      if (annotPlaceholder) annotPlaceholder.style.display = "none";

      // Render summary metrics
      setElText("det-total", result.total_vehicles);
      setElText("det-cars", result.counts?.car || 0);
      setElText("det-motorcycles", result.counts?.motorcycle || 0);
      setElText("det-buses", result.counts?.bus || 0);
      setElText("det-trucks", result.counts?.truck || 0);
      setElText("det-inference-time", `${result.inference_time_ms} ms`);

      // Populate detections table
      renderDetectionsTable(result.detections);

      showToast(`Detected ${result.total_vehicles} vehicles in ${result.inference_time_ms}ms`, "success");
    }
  } catch (err) {
    showToast(`Detection error: ${err.message}`, "danger");
  } finally {
    if (detectBtn) detectBtn.disabled = false;
    if (loadingIndicator) loadingIndicator.style.display = "none";
  }
}

function renderDetectionsTable(detections) {
  const tableBody = document.getElementById("detections-table-body");
  if (!tableBody) return;

  if (!detections || detections.length === 0) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="5" style="text-align: center; color: var(--text-muted); padding: 2rem;">
          No vehicles detected above confidence threshold.
        </td>
      </tr>
    `;
    return;
  }

  tableBody.innerHTML = detections
    .map(
      (d, i) => `
    <tr>
      <td>#${i + 1}</td>
      <td>
        <span class="badge badge-cyan">${escapeHtml(d.class_name.toUpperCase())}</span>
      </td>
      <td><strong>${(d.confidence * 100).toFixed(1)}%</strong></td>
      <td>[${d.bbox.join(", ")}]</td>
      <td>COCO ID: ${d.class_id}</td>
    </tr>
  `
    )
    .join("");
}

