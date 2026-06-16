"use strict";

// ─── State ────────────────────────────────────────────────────────────────────
let selectedFile = null;
let resultBase64 = null;

// ─── DOM references ───────────────────────────────────────────────────────────
const uploadArea        = document.getElementById("uploadArea");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");
const fileInput         = document.getElementById("fileInput");
const browseLink        = document.getElementById("browseLink");
const clearBtn          = document.getElementById("clearBtn");
const previewImg        = document.getElementById("previewImage");
const promptInput       = document.getElementById("promptInput");
const processBtn        = document.getElementById("processBtn");
const progressPanel     = document.getElementById("progressPanel");
const resultsPanel      = document.getElementById("resultsPanel");
const errorPanel        = document.getElementById("errorPanel");
const errorMessage      = document.getElementById("errorMessage");
const dismissError      = document.getElementById("dismissError");
const originalImg       = document.getElementById("originalImg");
const resultImg         = document.getElementById("resultImg");
const downloadBtn       = document.getElementById("downloadBtn");
const retryBtn          = document.getElementById("retryBtn");
const aiSummary         = document.getElementById("aiSummary");

// ─── Upload handling ──────────────────────────────────────────────────────────
uploadArea.addEventListener("click", (e) => {
  if (e.target === clearBtn || clearBtn.contains(e.target)) return;
  fileInput.click();
});

browseLink.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("drag-over");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("drag-over");
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) loadFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

clearBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  clearImage();
});

function loadFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.classList.remove("hidden");
    uploadPlaceholder.classList.add("hidden");
    clearBtn.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
  refreshProcessBtn();
}

function clearImage() {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewImg.classList.add("hidden");
  clearBtn.classList.add("hidden");
  uploadPlaceholder.classList.remove("hidden");
  refreshProcessBtn();
}

// ─── Prompt input ─────────────────────────────────────────────────────────────
promptInput.addEventListener("input", refreshProcessBtn);

function refreshProcessBtn() {
  processBtn.disabled = !(selectedFile && promptInput.value.trim().length > 0);
}

// ─── Step helpers ─────────────────────────────────────────────────────────────
function setStep(id, state, detail = "") {
  const el = document.getElementById(id);
  el.classList.remove("active", "done", "error");
  if (state) el.classList.add(state);
  const detailEl = document.getElementById(id + "Detail");
  if (detailEl) detailEl.textContent = detail;
}

function allStepsError() {
  ["step1", "step2"].forEach((id) => setStep(id, "error"));
}

// ─── Process ──────────────────────────────────────────────────────────────────
processBtn.addEventListener("click", runPipeline);
promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runPipeline();
});

async function runPipeline() {
  if (!selectedFile || !promptInput.value.trim()) return;

  hidePanel(resultsPanel);
  hidePanel(errorPanel);
  showPanel(progressPanel);
  processBtn.disabled = true;
  resultBase64 = null;

  setStep("step1", "active");
  setStep("step2", null);

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("prompt", promptInput.value.trim());

  const t2 = setTimeout(() => {
    setStep("step1", "done", "Style prompt ready");
    setStep("step2", "active", "This may take 30–60 seconds...");
  }, 1500);

  try {
    const response = await fetch("/api/process", { method: "POST", body: formData });

    clearTimeout(t2);

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || `Server error ${response.status}`);
    }

    setStep("step1", "done", `Style: ${data.parsed.style_name}`);
    setStep("step2", "done", "Transformation complete");

    await delay(500);
    hidePanel(progressPanel);

    originalImg.src = URL.createObjectURL(selectedFile);
    resultBase64 = data.result_image;
    resultImg.src = `data:image/png;base64,${data.result_image}`;

    aiSummary.innerHTML = buildSummaryHTML(data);

    showPanel(resultsPanel);

  } catch (err) {
    clearTimeout(t2);
    allStepsError();
    await delay(400);
    hidePanel(progressPanel);
    showError(err.message || String(err));
  } finally {
    processBtn.disabled = !(selectedFile && promptInput.value.trim());
  }
}

// ─── AI Summary HTML ──────────────────────────────────────────────────────────
function buildSummaryHTML(data) {
  const p = data.parsed;
  const items = [
    ["Style",    escapeHTML(p.style_name)],
    ["Strength", escapeHTML(String(p.strength))],
  ];
  return items.map(([label, value]) => `
    <div class="ai-summary-item">
      <div class="label">${label}</div>
      <div class="value">${value}</div>
    </div>`).join("");
}

function escapeHTML(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ─── Download ─────────────────────────────────────────────────────────────────
downloadBtn.addEventListener("click", () => {
  if (!resultBase64) return;
  const a = document.createElement("a");
  a.href = `data:image/png;base64,${resultBase64}`;
  a.download = "transformed.png";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});

// ─── Retry ────────────────────────────────────────────────────────────────────
retryBtn.addEventListener("click", () => {
  hidePanel(resultsPanel);
});

// ─── Error ────────────────────────────────────────────────────────────────────
dismissError.addEventListener("click", () => hidePanel(errorPanel));

function showError(msg) {
  errorMessage.textContent = msg;
  showPanel(errorPanel);
}

// ─── Panel helpers ────────────────────────────────────────────────────────────
function showPanel(el) { el.classList.remove("hidden"); }
function hidePanel(el) { el.classList.add("hidden"); }

function delay(ms) { return new Promise((r) => setTimeout(r, ms)); }
