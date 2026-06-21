// ============================================
// Annadata — AI Crop Advisor — Frontend Logic
// ============================================

// ⚠️ IMPORTANT: After deploying the backend on Render, replace the URL below
// with your live Render URL, e.g. "https://annadata-backend.onrender.com/api"
// Keep it as localhost while testing on your own machine.
const API_BASE = "http://localhost:5000/api";

let metadata = null;
let selectedCrop = null;
let lastRecommendations = [];

// ---------- Init ----------

document.addEventListener("DOMContentLoaded", async () => {
  await loadMetadata();
  populateYieldDropdowns();
  populateManualCropSelect();
  bindEvents();
});

async function loadMetadata() {
  try {
    const res = await fetch(`${API_BASE}/metadata`);
    metadata = await res.json();

    document.getElementById("stat-accuracy").textContent =
      (metadata.recommendation.accuracy * 100).toFixed(1) + "%";
    document.getElementById("stat-r2").textContent =
      metadata.yield.r2.toFixed(3);
    document.getElementById("method-acc").textContent =
      (metadata.recommendation.accuracy * 100).toFixed(1) + "% test accuracy";
    document.getElementById("method-r2").textContent =
      "R² = " + metadata.yield.r2.toFixed(3) + " on test set";
  } catch (err) {
    console.error("Could not load metadata:", err);
  }
}

function populateYieldDropdowns() {
  if (!metadata) return;
  const opts = metadata.yield.categorical_options;

  fillSelect("Soil_Type", opts.Soil_Type);
  fillSelect("Region", opts.Region);
  fillSelect("Season", opts.Season);
  fillSelect("Irrigation_Type", opts.Irrigation_Type);
}

function fillSelect(id, values) {
  const select = document.getElementById(id);
  if (!select) return;
  select.innerHTML = "";
  values.forEach((v) => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    select.appendChild(opt);
  });
}

function populateManualCropSelect() {
  if (!metadata) return;
  const select = document.getElementById("manual-crop-select");
  metadata.yield_supported_crops.forEach((crop) => {
    const opt = document.createElement("option");
    opt.value = crop;
    opt.textContent = crop;
    select.appendChild(opt);
  });
}

// ---------- Event bindings ----------

function bindEvents() {
  document.getElementById("recommend-form").addEventListener("submit", handleRecommendSubmit);
  document.getElementById("yield-form").addEventListener("submit", handleYieldSubmit);
  document.getElementById("clear-crop-btn").addEventListener("click", clearSelectedCrop);
  document.getElementById("manual-crop-select").addEventListener("change", (e) => {
    if (e.target.value) selectCrop(e.target.value, true);
  });
}

// ---------- Stage 1: Recommendation ----------

async function handleRecommendSubmit(e) {
  e.preventDefault();
  const btn = document.getElementById("recommend-btn");
  const errorBanner = document.getElementById("recommend-error");
  errorBanner.hidden = true;

  const payload = {
    N: document.getElementById("N").value,
    P: document.getElementById("P").value,
    K: document.getElementById("K").value,
    pH: document.getElementById("pH").value,
    rainfall: document.getElementById("rainfall").value,
    temperature: document.getElementById("temperature").value,
  };

  setLoading(btn, true, "Recommend Crops");

  try {
    const res = await fetch(`${API_BASE}/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Something went wrong");
    }
    const data = await res.json();
    lastRecommendations = data.recommendations;
    renderRecommendations(data.recommendations);
  } catch (err) {
    errorBanner.textContent = "Couldn't get a recommendation: " + err.message + ". Is the backend running on localhost:5000?";
    errorBanner.hidden = false;
  } finally {
    setLoading(btn, false, "Recommend Crops");
  }
}

function renderRecommendations(recommendations) {
  const container = document.getElementById("crop-cards");
  container.innerHTML = "";

  recommendations.forEach((rec, i) => {
    const card = document.createElement("div");
    card.className = "crop-card";
    card.dataset.crop = rec.crop;

    const yieldNote = !isYieldSupported(rec.crop)
      ? `<p class="crop-no-yield-note">Yield model not yet trained for this crop</p>`
      : "";

    card.innerHTML = `
      <div class="crop-rank">#${i + 1} match</div>
      <div class="crop-name">${rec.crop}</div>
      <div class="crop-confidence-bar"><div class="crop-confidence-fill" style="width: ${rec.confidence}%"></div></div>
      <div class="crop-confidence-text">${rec.confidence}% confidence</div>
      ${yieldNote}
    `;

    card.addEventListener("click", () => selectCrop(rec.crop, isYieldSupported(rec.crop)));
    container.appendChild(card);
  });

  document.getElementById("recommend-results").hidden = false;
}

function isYieldSupported(crop) {
  if (!metadata) return false;
  const supported = metadata.yield_supported_crops.map((c) => c.toLowerCase());
  return supported.includes(crop.toLowerCase());
}

// ---------- Crop selection (bridges Stage 1 -> Stage 2) ----------

function selectCrop(cropName, yieldSupported) {
  // visually mark selected card
  document.querySelectorAll(".crop-card").forEach((card) => {
    card.classList.toggle("is-selected", card.dataset.crop === cropName);
  });

  if (!yieldSupported) {
    selectedCrop = null;
    showNoCropNotice();
    const errorBanner = document.getElementById("yield-error");
    errorBanner.textContent = `The yield model currently supports: ${metadata.yield_supported_crops.join(", ")}. "${cropName}" isn't covered yet — pick one of those below to forecast yield.`;
    errorBanner.hidden = false;
    document.getElementById("stage-yield").scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }

  selectedCrop = normalizeCropName(cropName);
  document.getElementById("yield-error").hidden = true;

  // Set the Crop_Type field via a hidden state — we store on the form submit
  document.getElementById("banner-crop-name").textContent = selectedCrop;
  document.getElementById("selected-crop-banner").hidden = false;
  document.getElementById("no-crop-notice").hidden = true;
  document.getElementById("yield-form").hidden = false;

  document.getElementById("stage-yield").scrollIntoView({ behavior: "smooth", block: "start" });
}

function normalizeCropName(crop) {
  // Yield dataset uses capitalized crop names e.g. "Maize"
  if (!metadata) return crop;
  const match = metadata.yield_supported_crops.find(
    (c) => c.toLowerCase() === crop.toLowerCase()
  );
  return match || crop;
}

function clearSelectedCrop() {
  selectedCrop = null;
  document.getElementById("selected-crop-banner").hidden = true;
  document.getElementById("yield-form").hidden = true;
  document.getElementById("yield-result").hidden = true;
  showNoCropNotice();
  document.querySelectorAll(".crop-card").forEach((card) => card.classList.remove("is-selected"));
}

function showNoCropNotice() {
  document.getElementById("no-crop-notice").hidden = false;
  document.getElementById("yield-form").hidden = true;
}

// ---------- Stage 2: Yield Prediction ----------

async function handleYieldSubmit(e) {
  e.preventDefault();
  const btn = document.getElementById("yield-btn");
  const errorBanner = document.getElementById("yield-error");
  errorBanner.hidden = true;

  if (!selectedCrop) {
    errorBanner.textContent = "Please select a crop first.";
    errorBanner.hidden = false;
    return;
  }

  const payload = {
    N: document.getElementById("y-N").value,
    P: document.getElementById("y-P").value,
    K: document.getElementById("y-K").value,
    Soil_pH: document.getElementById("Soil_pH").value,
    Soil_Moisture: document.getElementById("Soil_Moisture").value,
    Soil_Type: document.getElementById("Soil_Type").value,
    Organic_Carbon: document.getElementById("Organic_Carbon").value,
    Temperature: document.getElementById("Temperature").value,
    Humidity: document.getElementById("Humidity").value,
    Rainfall: document.getElementById("Rainfall").value,
    Sunlight_Hours: document.getElementById("Sunlight_Hours").value,
    Wind_Speed: document.getElementById("Wind_Speed").value,
    Region: document.getElementById("Region").value,
    Altitude: document.getElementById("Altitude").value,
    Season: document.getElementById("Season").value,
    Crop_Type: selectedCrop,
    Irrigation_Type: document.getElementById("Irrigation_Type").value,
    Fertilizer_Used: document.getElementById("Fertilizer_Used").value,
    Pesticide_Used: document.getElementById("Pesticide_Used").value,
  };

  setLoading(btn, true, "Forecast Yield");

  try {
    const res = await fetch(`${API_BASE}/predict-yield`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Something went wrong");
    }
    const data = await res.json();
    renderYieldResult(data);
  } catch (err) {
    errorBanner.textContent = "Couldn't forecast yield: " + err.message;
    errorBanner.hidden = false;
  } finally {
    setLoading(btn, false, "Forecast Yield");
  }
}

function renderYieldResult(data) {
  document.getElementById("yield-number").textContent = data.predicted_yield_tons_per_hectare;

  const riskEl = document.getElementById("yield-risk");
  riskEl.textContent = data.risk_assessment;
  riskEl.className = "yield-risk";
  if (data.risk_assessment.toLowerCase().includes("high")) riskEl.classList.add("risk-high");
  else if (data.risk_assessment.toLowerCase().includes("moderate")) riskEl.classList.add("risk-moderate");
  else riskEl.classList.add("risk-low");

  document.getElementById("yield-result").hidden = false;
}

// ---------- Helpers ----------

function setLoading(btn, isLoading, defaultLabel) {
  btn.disabled = isLoading;
  const span = btn.querySelector("span");
  span.textContent = isLoading ? "Calculating…" : defaultLabel;
}
