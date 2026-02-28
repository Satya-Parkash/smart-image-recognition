/**
 * Smart Image Recognition System — Frontend Script
 * =================================================
 * Features:
 *  - Drag & Drop image upload
 *  - Click-to-browse file selection
 *  - Live image preview with metadata
 *  - AJAX POST to /predict (fetch API, FormData)
 *  - Animated confidence bar rendering
 *  - Loading spinner during inference
 *  - Error handling & display
 */

/* ── Element References ──────────────────────────────────────────────────── */
const dropZone      = document.getElementById("dropZone");
const fileInput     = document.getElementById("fileInput");
const previewWrap   = document.getElementById("previewWrap");
const previewImg    = document.getElementById("previewImg");
const previewMeta   = document.getElementById("previewMeta");
const btnClear      = document.getElementById("btnClear");
const btnAnalyse    = document.getElementById("btnAnalyse");
const errorBox      = document.getElementById("errorBox");

// Results panel elements
const emptyState    = document.getElementById("emptyState");
const spinnerWrap   = document.getElementById("spinnerWrap");
const resultsContent = document.getElementById("resultsContent");
const topResult     = document.getElementById("topResult");
const predictionsList = document.getElementById("predictionsList");

/* ── State ───────────────────────────────────────────────────────────────── */
let selectedFile = null;

/* ── Drag & Drop ─────────────────────────────────────────────────────────── */

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

["dragleave", "dragend"].forEach((evt) =>
  dropZone.addEventListener(evt, () => dropZone.classList.remove("drag-over"))
);

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

/* Click to open file dialog */
dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

/* ── File Handling ───────────────────────────────────────────────────────── */

/**
 * Validate and load a selected file into the preview.
 * @param {File} file
 */
function handleFile(file) {
  clearError();

  // Client-side validation
  const ALLOWED = ["image/png", "image/jpeg", "image/webp", "image/bmp", "image/gif"];
  if (!ALLOWED.includes(file.type)) {
    showError("Invalid file type. Please upload PNG, JPG, WEBP, BMP, or GIF.");
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    showError("File too large. Maximum size is 10 MB.");
    return;
  }

  selectedFile = file;

  // Build preview
  const objectURL = URL.createObjectURL(file);
  previewImg.src = objectURL;
  previewImg.onload = () => URL.revokeObjectURL(objectURL); // free memory

  // File metadata strip
  const sizeKB = (file.size / 1024).toFixed(1);
  previewMeta.textContent = `${file.name}  ·  ${sizeKB} KB  ·  ${file.type.split("/")[1].toUpperCase()}`;

  // Show preview, hide drop zone
  dropZone.style.display = "none";
  previewWrap.style.display = "block";

  // Enable analyse button
  btnAnalyse.disabled = false;

  // Reset results panel to empty state
  showPanel("empty");
}

/* ── Clear ───────────────────────────────────────────────────────────────── */

btnClear.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewWrap.style.display = "none";
  dropZone.style.display = "";
  btnAnalyse.disabled = true;
  clearError();
  showPanel("empty");
});

/* ── Analyse Button → AJAX Predict ──────────────────────────────────────── */

btnAnalyse.addEventListener("click", async () => {
  if (!selectedFile) return;

  clearError();
  showPanel("spinner");
  btnAnalyse.disabled = true;
  btnAnalyse.querySelector(".btn-text").textContent = "ANALYSING…";

  try {
    const formData = new FormData();
    formData.append("image", selectedFile);

    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    // Handle HTTP errors
    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
      throw new Error(err.error || `Server error ${response.status}`);
    }

    const data = await response.json();

    if (!data.success) {
      throw new Error(data.error || "Unknown server error.");
    }

    renderResults(data.predictions);
    showPanel("results");

  } catch (err) {
    showError(err.message);
    showPanel("empty");
  } finally {
    btnAnalyse.disabled = false;
    btnAnalyse.querySelector(".btn-text").textContent = "ANALYSE IMAGE";
  }
});

/* ── Results Rendering ───────────────────────────────────────────────────── */

/**
 * Populate the results panel with prediction data.
 * @param {Array<{label: string, confidence: number}>} predictions
 */
function renderResults(predictions) {
  if (!predictions || predictions.length === 0) {
    showError("No predictions returned.");
    showPanel("empty");
    return;
  }

  const top = predictions[0];

  // ── Top result banner ──────────────────────────────────────────────────
  topResult.innerHTML = `
    <div class="top-label">${escapeHtml(top.label)}</div>
    <div class="top-confidence">${top.confidence}% confidence</div>
  `;

  // ── All predictions with confidence bars ───────────────────────────────
  predictionsList.innerHTML = predictions
    .map((pred, i) => `
      <div class="prediction-item">
        <div class="pred-header">
          <span class="pred-rank">#${i + 1}</span>
          <span class="pred-label">${escapeHtml(pred.label)}</span>
          <span class="pred-pct">${pred.confidence}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill rank-${i + 1}" data-width="${pred.confidence}"></div>
        </div>
      </div>
    `)
    .join("");

  // Animate bars after a short delay (allow DOM paint first)
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.querySelectorAll(".bar-fill").forEach((bar) => {
        bar.style.width = bar.dataset.width + "%";
      });
    }, 80);
  });
}

/* ── Panel State Management ──────────────────────────────────────────────── */

/**
 * Switch results panel between: "empty" | "spinner" | "results"
 */
function showPanel(state) {
  emptyState.style.display     = state === "empty"   ? "flex" : "none";
  spinnerWrap.style.display    = state === "spinner" ? "flex" : "none";
  resultsContent.style.display = state === "results" ? "flex" : "none";
}

/* ── Error Helpers ───────────────────────────────────────────────────────── */

function showError(message) {
  errorBox.textContent = `⚠ ${message}`;
  errorBox.style.display = "block";
}

function clearError() {
  errorBox.style.display = "none";
  errorBox.textContent = "";
}

/* ── Security ────────────────────────────────────────────────────────────── */

/** Basic XSS prevention for dynamically inserted text. */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}