// ==================== STATE MANAGEMENT ====================
let videoElem = document.getElementById("webcam-video");
let canvasElem = document.getElementById("output-canvas");
let ctx = canvasElem.getContext("2d");

let currentFacingMode = "environment"; // Mặc định camera sau của điện thoại
let isStreaming = false;
let currentConf = 35;
let autoEnhance = false;
let autoSpeak = false; // Tự động đọc khi quét được vật mới
let lastSpokenPrimary = "";
let currentModel = "hybrid"; // "hybrid", "finetuned", "small" hoặc "large"
let detectedHistory = new Map(); // {primary_key: {primary, vi, en, zh, ko, ru}}
let lastTranslation = "";

// ==================== CẤU HÌNH HỌC TỪ VỰNG ĐA NGÔN NGỮ ====================
let currentLang = localStorage.getItem("damon_selected_lang") || "vi";

const LANG_LEARNING_INFO = {
  vi: { name: "Tiếng Việt", flag: "🇻🇳", label: "Tiếng Việt 🇻🇳" },
  en: { name: "Tiếng Anh", flag: "🇬🇧", label: "Tiếng Anh 🇬🇧" },
  zh: { name: "Tiếng Trung", flag: "🇨🇳", label: "Tiếng Trung 🇨🇳" },
  ko: { name: "Tiếng Hàn", flag: "🇰🇷", label: "Tiếng Hàn 🇰🇷" },
  ru: { name: "Tiếng Nga", flag: "🇷🇺", label: "Tiếng Nga 🇷🇺" }
};

// Lấy từ vựng theo ngôn ngữ mục tiêu đang học
function getTargetWord(detOrItem) {
  if (!detOrItem) return "";
  return detOrItem[currentLang] || detOrItem.vi || detOrItem.primary || detOrItem.key || "";
}

// Lấy nghĩa Tiếng Việt chuẩn để đối chiếu học từ
function getViMeaning(detOrItem) {
  if (!detOrItem) return "";
  return detOrItem.vi || detOrItem.primary || detOrItem.key || "";
}

// Cập nhật lại các vùng hiển thị từ vựng khi đổi ngôn ngữ học
function updateLearningViews() {
  updateHistoryUI(true);
  renderQuickChips();
  const searchInput = document.getElementById("feedback-search-input");
  renderClassResults(searchInput ? searchInput.value : "");

  // Cập nhật ngay banner từ vựng chính nếu đang có vật thể được nhận diện
  if (currentBestPrimary) {
    const item = detectedHistory.get(currentBestPrimary);
    if (item) {
      currentBestTarget = getTargetWord(item);
      currentBestVi = getViMeaning(item);
      const confPct = item.conf ? Math.round(item.conf * 100) : null;
      const confSuffix = confPct ? ` - ${confPct}%` : "";
      if (currentLang === "vi") {
        document.getElementById("primary-translation-text").innerText = `${currentBestVi} (${currentBestPrimary.toUpperCase()})${confSuffix}`;
      } else {
        document.getElementById("primary-translation-text").innerText = `${currentBestTarget} • ${currentBestVi}${confSuffix}`;
      }
    }
  }
}

let lastFrameTime = performance.now();
let fps = 60;
let isPredicting = false;

// ==================== CẤU HÌNH CAMERA ĐIỆN THOẠI ====================
async function startMobileCamera() {
  try {
    if (videoElem.srcObject) {
      videoElem.srcObject.getTracks().forEach(track => track.stop());
    }

    const constraints = {
      video: {
        facingMode: { ideal: currentFacingMode },
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 60 }
      },
      audio: false
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoElem.srcObject = stream;

    videoElem.onloadedmetadata = () => {
      videoElem.play();
      canvasElem.width = videoElem.videoWidth || 640;
      canvasElem.height = videoElem.videoHeight || 480;
      isStreaming = true;
      requestAnimationFrame(renderLoop);
    };

  } catch (err) {
    console.error("Không thể mở camera:", err);
    document.getElementById("primary-translation-text").innerText = "⚠️ Lỗi mở camera. Hãy cấp quyền truy cập camera!";
  }
}

// ==================== DỰ ĐOÁN KHUNG HÌNH (GỬI LÊN GPU BACKEND) ====================
async function captureAndPredict() {
  if (!isStreaming || isPredicting) return;

  isPredicting = true;

  const offCanvas = document.createElement("canvas");
  offCanvas.width = canvasElem.width;
  offCanvas.height = canvasElem.height;
  const offCtx = offCanvas.getContext("2d");
  offCtx.drawImage(videoElem, 0, 0, offCanvas.width, offCanvas.height);

  offCanvas.toBlob(async (blob) => {
    if (!blob) {
      isPredicting = false;
      return;
    }

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");
    formData.append("conf", (currentConf / 100).toFixed(2));
    formData.append("enhance", autoEnhance ? "true" : "false");
    formData.append("model", currentModel);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        handlePredictionResult(data);
      }
    } catch (e) {
      console.warn("Lỗi gửi khung hình:", e);
    } finally {
      isPredicting = false;
    }
  }, "image/jpeg", 0.75);
}

// ==================== RENDER KHUNG HÌNH & BOUNDING BOX ====================
let latestDetections = [];
let currentBestPrimary = "";
let currentBestTarget = "";
let currentBestVi = "";

function handlePredictionResult(data) {
  latestDetections = data.detections || [];

  if (data.gpu_name) {
    document.getElementById("gpu-name-text").innerText = `GPU: ${data.gpu_name}`;
  }

  let bestConf = 0;
  let bestText = "";
  currentBestPrimary = "";
  currentBestTarget = "";
  currentBestVi = "";

  latestDetections.forEach(det => {
    const targetWord = getTargetWord(det);
    const viWord = getViMeaning(det);

    detectedHistory.set(det.primary, {
      primary: det.primary,
      conf: det.conf,
      vi: det.vi || det.primary,
      en: det.en || det.primary,
      zh: det.zh || det.primary,
      ko: det.ko || det.primary,
      ru: det.ru || det.primary,
      targetWord: targetWord,
      viWord: viWord
    });

    if (det.conf > bestConf) {
      bestConf = det.conf;
      currentBestPrimary = det.primary;
      currentBestTarget = targetWord;
      currentBestVi = viWord;

      const confPct = Math.round(det.conf * 100);
      if (currentLang === "vi") {
        bestText = `${viWord} (${det.primary.toUpperCase()}) - ${confPct}%`;
      } else {
        bestText = `${targetWord} • ${viWord} - ${confPct}%`;
      }
    }
  });

  if (bestText) {
    lastTranslation = bestText;
    document.getElementById("primary-translation-text").innerText = bestText;

    // Nếu bật Auto Speak và gặp đồ vật mới
    if (autoSpeak && currentBestPrimary && currentBestPrimary !== lastSpokenPrimary) {
      lastSpokenPrimary = currentBestPrimary;
      speakTargetWord(currentBestPrimary, currentBestTarget);
    }
  } else {
    document.getElementById("primary-translation-text").innerText = "Đang quét đồ vật...";
  }

  updateHistoryUI();
}

function renderLoop(now) {
  if (!isStreaming) return;

  const delta = now - lastFrameTime;
  if (delta > 0) {
    fps = 0.9 * fps + 0.1 * (1000 / delta);
    document.getElementById("fps-text").innerText = `${Math.round(fps)} FPS`;
  }
  lastFrameTime = now;

  ctx.drawImage(videoElem, 0, 0, canvasElem.width, canvasElem.height);

  latestDetections.forEach(det => {
    const [x1, y1, x2, y2] = det.box;
    const confPct = Math.round(det.conf * 100);
    const targetWord = getTargetWord(det);
    const viWord = getViMeaning(det);

    let label = "";
    if (currentLang === "vi") {
      label = `${viWord} (${det.primary}) ${confPct}%`;
    } else {
      label = `${targetWord} (${viWord}) ${confPct}%`;
    }

    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    ctx.font = "bold 15px 'Outfit', 'Microsoft YaHei', 'Malgun Gothic', 'Segoe UI', sans-serif";
    const textWidth = ctx.measureText(label).width;
    const bgHeight = 26;
    const bgY = Math.max(0, y1 - bgHeight);

    ctx.fillStyle = "#059669";
    ctx.fillRect(x1, bgY, textWidth + 12, bgHeight);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x1 + 6, bgY + 18);
  });

  captureAndPredict();
  requestAnimationFrame(renderLoop);
}

// ==================== CẬP NHẬT GIAO DIỆN LỊCH SỬ ====================
let lastHistoryCount = -1;

function updateHistoryUI(force = false) {
  // Chỉ cập nhật DOM khi có đồ vật MỚI được thêm vào hoặc đổi ngôn ngữ học
  if (!force && detectedHistory.size === lastHistoryCount) return;
  lastHistoryCount = detectedHistory.size;

  const container = document.getElementById("history-container");
  const countEl = document.getElementById("history-count");
  if (countEl) countEl.innerText = detectedHistory.size;

  if (detectedHistory.size === 0) {
    container.innerHTML = `<div style="color: var(--text-muted); font-size: 0.85rem;" id="empty-history-msg">Chưa có đồ vật nào trong lịch sử quét...</div>`;
    return;
  }

  let html = "";
  detectedHistory.forEach((item, primaryKey) => {
    const targetWord = (typeof item === "object") ? getTargetWord(item) : item;
    const viWord = (typeof item === "object") ? getViMeaning(item) : item;

    if (currentLang === "vi") {
      html += `
        <div class="history-item">
          <span class="history-item-dot"></span>
          <span>${primaryKey}: <strong>${viWord}</strong></span>
          <button onclick="speakTargetWord('${primaryKey}', '${viWord.replace(/'/g, "\\'")}')" style="background:none; border:none; color: var(--primary); cursor:pointer; margin-left: 4px;" title="Nghe phát âm">🔊</button>
        </div>
      `;
    } else {
      html += `
        <div class="history-item">
          <span class="history-item-dot"></span>
          <span><strong style="color: var(--secondary); font-size: 0.95rem;">${targetWord}</strong> <span style="color: var(--text-muted); font-size: 0.82rem;">(${viWord})</span></span>
          <button onclick="speakTargetWord('${primaryKey}', '${targetWord.replace(/'/g, "\\'")}')" style="background:none; border:none; color: var(--primary); cursor:pointer; margin-left: 4px;" title="Nghe phát âm tiếng ${LANG_LEARNING_INFO[currentLang]?.name || ''}">🔊</button>
        </div>
      `;
    }
  });
  container.innerHTML = html;
}

// ==================== SPEECH SYNTHESIS ĐA NGÔN NGỮ (VIỆT, ANH, TRUNG, HÀN, NGA) ====================
let currentAudioPlayer = null;

function cleanWordForSpeech(text) {
  if (!text) return "";
  // Lấy từ chính đầu tiên trước dấu gạch chéo hoặc phẩy
  return text.split("/")[0].split(",")[0].trim();
}

function speakTargetWord(primaryKey, word) {
  const rawText = word || primaryKey;
  if (!rawText) return;
  const textToSpeak = cleanWordForSpeech(rawText);

  if (currentAudioPlayer) {
    try {
      currentAudioPlayer.pause();
      currentAudioPlayer.currentTime = 0;
    } catch (_) {}
    currentAudioPlayer = null;
  }

  // Gọi Edge Neural TTS từ server theo đúng ngôn ngữ đang học (chỉ đọc duy nhất từ vựng mục tiêu)
  const ttsUrl = `/api/tts?text=${encodeURIComponent(textToSpeak)}&lang=${encodeURIComponent(currentLang)}`;

  const audio = new Audio(ttsUrl);
  currentAudioPlayer = audio;

  audio.onerror = () => {
    console.warn("Lỗi tải audio server, fallback sang giọng trình duyệt:", textToSpeak);
    fallbackBrowserSpeak(textToSpeak);
  };

  audio.play().catch(err => {
    console.warn("Dùng giọng đọc dự phòng của trình duyệt:", err);
    fallbackBrowserSpeak(textToSpeak);
  });
}

function fallbackBrowserSpeak(text) {
  if (!('speechSynthesis' in window) || !text) return;
  window.speechSynthesis.cancel();

  const langCodeMap = {
    vi: "vi-VN",
    en: "en-US",
    zh: "zh-CN",
    ko: "ko-KR",
    ru: "ru-RU"
  };

  const targetCode = langCodeMap[currentLang] || "vi-VN";
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = targetCode;
  utter.rate = 0.9;

  const voices = window.speechSynthesis.getVoices();
  const matchVoice = voices.find(v => {
    const vlang = (v.lang || "").replace("_", "-").toLowerCase();
    return vlang.startsWith(targetCode.toLowerCase().slice(0, 2));
  });
  if (matchVoice) utter.voice = matchVoice;

  window.speechSynthesis.speak(utter);
}

// ==================== BỘ LẮNG NGHE SỰ KIỆN GIAO DIỆN ====================
document.addEventListener("DOMContentLoaded", () => {
  startMobileCamera();

  document.getElementById("btn-switch-camera").addEventListener("click", () => {
    currentFacingMode = (currentFacingMode === "environment") ? "user" : "environment";
    startMobileCamera();
  });

  const lightBtn = document.getElementById("btn-toggle-light");
  lightBtn.addEventListener("click", () => {
    autoEnhance = !autoEnhance;
    lightBtn.style.background = autoEnhance ? "var(--primary)" : "rgba(15, 23, 42, 0.75)";
  });

  const confSlider = document.getElementById("conf-slider");
  confSlider.addEventListener("input", (e) => {
    currentConf = parseInt(e.target.value);
    document.getElementById("conf-val-display").innerText = `${currentConf}%`;
  });

  const btnHybrid = document.getElementById("btn-model-hybrid");
  const btnSmall = document.getElementById("btn-model-small");
  const btnLarge = document.getElementById("btn-model-large");
  const btnFinetuned = document.getElementById("btn-model-finetuned");

  function setModelActive(modelKey, activeBtn) {
    currentModel = modelKey;
    [btnHybrid, btnFinetuned, btnSmall, btnLarge].forEach(btn => {
      if (btn) btn.classList.remove("active");
    });
    if (activeBtn) activeBtn.classList.add("active");
    showToast(`Đã chuyển sang: ${activeBtn ? activeBtn.innerText.trim() : modelKey}`);
  }

  if (btnHybrid) {
    btnHybrid.addEventListener("click", () => setModelActive("hybrid", btnHybrid));
  }
  if (btnFinetuned) {
    btnFinetuned.addEventListener("click", () => setModelActive("finetuned", btnFinetuned));
  }
  if (btnSmall) {
    btnSmall.addEventListener("click", () => setModelActive("small", btnSmall));
  }
  if (btnLarge) {
    btnLarge.addEventListener("click", () => setModelActive("large", btnLarge));
  }

  // Bộ chọn ngôn ngữ học từ vựng
  const langSelector = document.getElementById("lang-selector");
  if (langSelector) {
    langSelector.value = currentLang;
    langSelector.addEventListener("change", (e) => {
      currentLang = e.target.value;
      localStorage.setItem("damon_selected_lang", currentLang);
      updateLearningViews();
      const info = LANG_LEARNING_INFO[currentLang];
      showToast(`🎯 Chế độ học từ vựng: ${info ? info.label : currentLang}`);
    });
  }

  // Khởi tạo hiển thị từ vựng ban đầu
  updateLearningViews();

  // Nút Auto đọc khi quét vật mới
  const btnAutoSpeak = document.getElementById("btn-toggle-auto-speak");
  btnAutoSpeak.addEventListener("click", () => {
    autoSpeak = !autoSpeak;
    if (autoSpeak) {
      btnAutoSpeak.innerText = "🎙️ Auto Đọc: BẬT";
      btnAutoSpeak.style.background = "rgba(16, 185, 129, 0.2)";
      btnAutoSpeak.style.borderColor = "var(--primary)";
      btnAutoSpeak.style.color = "var(--primary)";
    } else {
      btnAutoSpeak.innerText = "🎙️ Auto Đọc: TẮT";
      btnAutoSpeak.style.background = "rgba(139, 92, 246, 0.15)";
      btnAutoSpeak.style.borderColor = "var(--accent)";
      btnAutoSpeak.style.color = "var(--accent)";
    }
  });

  // Nút Nghe đọc thủ công (Theo ngôn ngữ đang học)
  document.getElementById("btn-speak").addEventListener("click", () => {
    if (currentBestPrimary) {
      let wordToSpeak = currentBestTarget;
      if (latestDetections && latestDetections.length > 0) {
        const match = latestDetections.find(d => d.primary === currentBestPrimary);
        if (match) {
          wordToSpeak = getTargetWord(match);
        }
      }
      speakTargetWord(currentBestPrimary, wordToSpeak);
    } else {
      showToast("Đang quét đồ vật... Hãy hướng camera về vật thể!");
    }
  });

  document.getElementById("btn-clear-history").addEventListener("click", () => {
    detectedHistory.clear();
    lastHistoryCount = -1;
    updateHistoryUI();
  });

  // ==================== XỬ LÝ SỬA NHÃN & ACTIVE LEARNING ====================
  loadClasses();

  // Mở modal sửa nhãn
  const btnReportWrong = document.getElementById("btn-report-wrong");
  if (btnReportWrong) {
    btnReportWrong.addEventListener("click", openFeedbackModal);
  }

  const btnOpenFeedback = document.getElementById("btn-open-feedback");
  if (btnOpenFeedback) {
    btnOpenFeedback.addEventListener("click", openFeedbackModal);
  }

  // Đóng modal sửa nhãn
  document.getElementById("btn-close-feedback-modal").addEventListener("click", closeFeedbackModal);
  document.getElementById("btn-cancel-feedback").addEventListener("click", closeFeedbackModal);

  // Search input & Clear search trong Modal Sửa Nhãn
  const searchInput = document.getElementById("feedback-search-input");
  const clearSearchBtn = document.getElementById("btn-clear-class-search");

  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      const q = e.target.value;
      if (clearSearchBtn) {
        clearSearchBtn.style.display = q ? "flex" : "none";
      }
      renderClassResults(q);
    });
  }

  if (clearSearchBtn) {
    clearSearchBtn.addEventListener("click", () => {
      if (searchInput) {
        searchInput.value = "";
        searchInput.focus();
      }
      clearSearchBtn.style.display = "none";
      renderClassResults("");
    });
  }

  // Nút Hủy / Đổi lại nhãn đã chọn
  const deselectBtn = document.getElementById("btn-deselect-class");
  if (deselectBtn) {
    deselectBtn.addEventListener("click", deselectClass);
  }

  // Nút Thêm nhãn mới thủ công
  const btnApplyCustom = document.getElementById("btn-apply-custom-class");
  const customInput = document.getElementById("feedback-custom-input");
  if (btnApplyCustom && customInput) {
    btnApplyCustom.addEventListener("click", () => {
      const val = customInput.value.trim();
      if (!val) {
        alert("Vui lòng nhập tên đồ vật mới!");
        return;
      }
      const newKey = removeVietnameseTones(val).toLowerCase().replace(/[^a-z0-9]+/g, "_");
      selectClass(newKey, val);
    });

    customInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        btnApplyCustom.click();
      }
    });
  }

  // Lưu mẫu feedback
  document.getElementById("btn-save-feedback").addEventListener("click", submitFeedback);

  // Modal hướng dẫn train
  const guideBackdrop = document.getElementById("modal-guide-backdrop");
  document.getElementById("btn-open-guide").addEventListener("click", () => {
    guideBackdrop.style.display = "flex";
  });
  document.getElementById("btn-close-guide-modal").addEventListener("click", () => {
    guideBackdrop.style.display = "none";
  });
  document.getElementById("btn-close-guide-action").addEventListener("click", () => {
    guideBackdrop.style.display = "none";
  });

  // Nút copy lệnh train
  document.getElementById("btn-copy-train-cmd").addEventListener("click", () => {
    const cmd = "python train_finetune.py --epochs 25";
    navigator.clipboard.writeText(cmd).then(() => {
      const copyBtn = document.getElementById("btn-copy-train-cmd");
      const oldText = copyBtn.innerText;
      copyBtn.innerText = "✅ Đã copy!";
      setTimeout(() => { copyBtn.innerText = oldText; }, 2000);
    });
  });
});

// ==================== TÍNH NĂNG ACTIVE LEARNING / SỬA NHÃN ====================
let availableClasses = [];
let feedbackBlob = null;
let feedbackBBox = null;
let selectedClassKey = "";
let selectedClassVi = "";

const CLASS_ICONS = {
  smartphone: "📱",
  laptop: "💻",
  mouse: "🖱️",
  keyboard: "⌨️",
  headset: "🎧",
  bottle: "🧴",
  cup: "☕",
  book: "📖",
  pen: "✏️",
  glasses: "👓",
  watch: "⌚",
  clock: "⏰",
  remote: "📟",
  wallet: "👛",
  key: "🔑",
  charger: "🔌",
  chair: "🪑",
  backpack: "🎒",
  scissors: "✂️",
  calculator: "🔢",
  speaker: "🔊",
  umbrella: "☂️",
  car: "🚗",
  motorcycle: "🏍️",
  bicycle: "🚲",
  apple: "🍎",
  banana: "🍌",
  spoon: "🥄",
  money: "💵",
  helmet: "🪖",
  tissue: "🧻",
  card: "💳",
  power_outlet: "🔌"
};

const QUICK_CLASS_KEYS = ["pen", "scissors", "glasses", "card", "smartphone", "bottle", "wallet", "key", "charger", "laptop", "cup", "book"];

// Chuẩn hóa loại bỏ dấu tiếng Việt để tìm kiếm thông minh (không phân biệt bút / but)
function removeVietnameseTones(str) {
  if (!str) return "";
  str = str.toLowerCase();
  str = str.replace(/à|á|ạ|ả|ã|â|ầ|ấ|ậ|ẩ|ẫ|ă|ằ|ắ|ặ|ẳ|ẵ/g, "a");
  str = str.replace(/è|é|ẹ|ẻ|ẽ|ê|ề|ế|ệ|ể|ễ/g, "e");
  str = str.replace(/ì|í|ị|ỉ|ĩ/g, "i");
  str = str.replace(/ò|ó|ọ|ỏ|õ|ô|ồ|ố|ộ|ổ|ỗ|ơ|ờ|ớ|ợ|ở|ỡ/g, "o");
  str = str.replace(/ù|ú|ụ|ủ|ũ|ư|ừ|ứ|ự|ử|ữ/g, "u");
  str = str.replace(/ỳ|ý|ỵ|ỷ|ỹ/g, "y");
  str = str.replace(/đ/g, "d");
  str = str.replace(/\u0300|\u0301|\u0303|\u0309|\u0323/g, "");
  str = str.replace(/\u02C6|\u0306|\u031B/g, "");
  return str.trim();
}

async function loadClasses() {
  try {
    const res = await fetch("/api/classes");
    const data = await res.json();
    availableClasses = data.classes || [];

    const badge = document.getElementById("feedback-count-badge");
    if (badge && data.sample_count !== undefined) {
      badge.innerText = `${data.sample_count} mẫu`;
    }

    renderQuickChips();
    renderClassResults("");
  } catch (err) {
    console.error("Không thể tải danh sách classes:", err);
  }
}

// Render các chip chọn nhanh
function renderQuickChips() {
  const container = document.getElementById("quick-chips-list");
  if (!container) return;

  let html = "";
  QUICK_CLASS_KEYS.forEach(key => {
    const item = availableClasses.find(c => c.key === key);
    if (item) {
      const icon = CLASS_ICONS[key] || "📦";
      const targetWord = getTargetWord(item);
      const viWord = getViMeaning(item);
      const chipDisplay = (currentLang === "vi") ? viWord : `${targetWord} (${viWord})`;
      const isActive = (selectedClassKey === key) ? "active" : "";
      html += `
        <button type="button" class="quick-chip ${isActive}" onclick="selectClass('${item.key}', '${viWord.replace(/'/g, "\\'")}', '${targetWord.replace(/'/g, "\\'")}')">
          <span>${icon}</span>
          <span>${chipDisplay}</span>
        </button>
      `;
    }
  });

  container.innerHTML = html;
}

// Render lưới kết quả tìm kiếm đồ vật đa ngôn ngữ
function renderClassResults(query) {
  const grid = document.getElementById("class-results-grid");
  if (!grid) return;

  const rawQuery = (query || "").toLowerCase().trim();
  const cleanQuery = removeVietnameseTones(rawQuery);

  const filtered = availableClasses.filter(c => {
    if (!rawQuery) return true;
    const cleanVi = removeVietnameseTones(c.vi || "");
    const cleanKey = (c.key || "").toLowerCase();
    const cleanEn = (c.en || "").toLowerCase();
    const cleanZh = (c.zh || "").toLowerCase();
    const cleanKo = (c.ko || "").toLowerCase();
    const cleanRu = (c.ru || "").toLowerCase();

    return (
      cleanVi.includes(cleanQuery) ||
      cleanKey.includes(rawQuery) ||
      cleanEn.includes(rawQuery) ||
      cleanZh.includes(rawQuery) ||
      cleanKo.includes(rawQuery) ||
      cleanRu.includes(rawQuery)
    );
  });

  let html = "";

  if (filtered.length === 0) {
    html += `
      <div style="grid-column: 1 / -1; text-align: center; padding: 16px; color: var(--text-muted); font-size: 0.85rem;">
        Không tìm thấy đồ vật nào khớp với "<strong>${escapeHtml(query)}</strong>".
      </div>
    `;
  } else {
    filtered.forEach(item => {
      const icon = CLASS_ICONS[item.key] || "📦";
      const isSelected = (selectedClassKey === item.key) ? "selected" : "";
      const targetWord = getTargetWord(item);
      const viWord = getViMeaning(item);

      let cardTitle = viWord;
      let cardSub = item.key;
      if (currentLang !== "vi") {
        cardTitle = targetWord;
        cardSub = `${viWord} • ${item.key}`;
      }

      html += `
        <div class="class-item-card ${isSelected}" onclick="selectClass('${item.key}', '${viWord.replace(/'/g, "\\'")}', '${targetWord.replace(/'/g, "\\'")}')">
          <span class="class-item-icon">${icon}</span>
          <div class="class-item-info">
            <span class="class-item-vi">${cardTitle}</span>
            <span class="class-item-en">${cardSub}</span>
          </div>
        </div>
      `;
    });
  }

  // Luôn có nút tạo mới nếu người dùng gõ từ khóa chưa có trong từ điển
  if (query && query.trim().length > 0) {
    html += `
      <div class="class-custom-card" onclick="openCustomClassInput('${escapeHtml(query.trim())}')">
        <span>➕ Thêm nhãn mới: "<strong>${escapeHtml(query.trim())}</strong>"</span>
      </div>
    `;
  }

  grid.innerHTML = html;
}

function escapeHtml(text) {
  if (!text) return "";
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function selectClass(key, vi, target) {
  selectedClassKey = key;
  selectedClassVi = vi;

  const hiddenKey = document.getElementById("feedback-selected-key");
  const hiddenVi = document.getElementById("feedback-selected-vi");
  if (hiddenKey) hiddenKey.value = key;
  if (hiddenVi) hiddenVi.value = vi;

  const pill = document.getElementById("selected-class-pill");
  const nameEl = document.getElementById("selected-class-name");
  const icon = CLASS_ICONS[key] || "🏷️";
  const displayLabel = (target && currentLang !== "vi") ? `${target} (${vi})` : vi;
  if (nameEl) nameEl.innerText = `${icon} ${displayLabel} [${key.toUpperCase()}]`;
  if (pill) pill.style.display = "flex";

  const customGroup = document.getElementById("custom-class-group");
  if (customGroup) customGroup.style.display = "none";

  renderQuickChips();
  const searchInput = document.getElementById("feedback-search-input");
  renderClassResults(searchInput ? searchInput.value : "");
}

function deselectClass() {
  selectedClassKey = "";
  selectedClassVi = "";

  const hiddenKey = document.getElementById("feedback-selected-key");
  const hiddenVi = document.getElementById("feedback-selected-vi");
  if (hiddenKey) hiddenKey.value = "";
  if (hiddenVi) hiddenVi.value = "";

  const pill = document.getElementById("selected-class-pill");
  if (pill) pill.style.display = "none";

  renderQuickChips();
  const searchInput = document.getElementById("feedback-search-input");
  renderClassResults(searchInput ? searchInput.value : "");
  if (searchInput) searchInput.focus();
}

function openCustomClassInput(prefill) {
  const customGroup = document.getElementById("custom-class-group");
  const customInput = document.getElementById("feedback-custom-input");
  if (customGroup && customInput) {
    customGroup.style.display = "flex";
    customInput.value = prefill || "";
    customInput.focus();
  }
}

// ==================== TÍNH NĂNG KÉO THẢ BOUNDING BOX TRÊN ẢNH ====================
let feedbackCanvas = null;
let feedbackCtx = null;
let feedbackImgElement = null;
let boxState = { x1: 0, y1: 0, x2: 0, y2: 0 };
let initialBoxState = null;
let dragMode = null;
let dragStart = { x: 0, y: 0 };
let dragBoxStart = null;
let boxEventsAttached = false;
const HANDLE_RADIUS = 14;

function initBoxEditor(imgObj, detectedBox) {
  feedbackCanvas = document.getElementById("feedback-canvas");
  if (!feedbackCanvas) return;
  feedbackCtx = feedbackCanvas.getContext("2d");
  feedbackImgElement = imgObj;

  const imgW = imgObj.naturalWidth || imgObj.width || 640;
  const imgH = imgObj.naturalHeight || imgObj.height || 480;

  // Lấy kích thước thực của khung chứa
  const previewBox = document.querySelector(".modal-preview-box");
  const boxW = (previewBox && previewBox.clientWidth > 0) ? previewBox.clientWidth : 340;
  const boxH = (previewBox && previewBox.clientHeight > 0) ? previewBox.clientHeight : 250;

  // Tính kích thước hiển thị CSS đúng tuyệt đối theo tỉ lệ ảnh gốc (không méo 1 pixel)
  const imgAspect = imgW / imgH;
  const boxAspect = boxW / boxH;

  let displayW, displayH;
  if (imgAspect > boxAspect) {
    displayW = boxW;
    displayH = Math.round(boxW / imgAspect);
  } else {
    displayH = boxH;
    displayW = Math.round(boxH * imgAspect);
  }

  feedbackCanvas.style.width = displayW + "px";
  feedbackCanvas.style.height = displayH + "px";

  // Độ phân giải thực của canvas bằng ảnh gốc
  feedbackCanvas.width = imgW;
  feedbackCanvas.height = imgH;

  const w = imgW;
  const h = imgH;

  if (detectedBox && detectedBox.length === 4) {
    let [bx1, by1, bx2, by2] = detectedBox;
    boxState = {
      x1: Math.max(0, Math.min(w - 20, Math.min(bx1, bx2))),
      y1: Math.max(0, Math.min(h - 20, Math.min(by1, by2))),
      x2: Math.min(w, Math.max(20, Math.max(bx1, bx2))),
      y2: Math.min(h, Math.max(20, Math.max(by1, by2)))
    };
  } else {
    // Mặc định khung ở trung tâm 60%
    boxState = {
      x1: Math.round(w * 0.2),
      y1: Math.round(h * 0.2),
      x2: Math.round(w * 0.8),
      y2: Math.round(h * 0.8)
    };
  }

  initialBoxState = { ...boxState };
  setupBoxCanvasEvents();
  drawFeedbackCanvas();
}

function drawFeedbackCanvas() {
  if (!feedbackCanvas || !feedbackCtx || !feedbackImgElement) return;

  const w = feedbackCanvas.width;
  const h = feedbackCanvas.height;

  feedbackCtx.clearRect(0, 0, w, h);

  // 1. Vẽ ảnh chụp gốc
  feedbackCtx.drawImage(feedbackImgElement, 0, 0, w, h);

  const x1 = Math.min(boxState.x1, boxState.x2);
  const y1 = Math.min(boxState.y1, boxState.y2);
  const x2 = Math.max(boxState.x1, boxState.x2);
  const y2 = Math.max(boxState.y1, boxState.y2);
  const bw = Math.max(1, x2 - x1);
  const bh = Math.max(1, y2 - y1);

  // 2. Làm mờ vùng bên ngoài bounding box (Vùng không chọn)
  feedbackCtx.fillStyle = "rgba(0, 0, 0, 0.45)";
  feedbackCtx.fillRect(0, 0, w, y1);
  feedbackCtx.fillRect(0, y2, w, h - y2);
  feedbackCtx.fillRect(0, y1, x1, bh);
  feedbackCtx.fillRect(x2, y1, w - x2, bh);

  // 3. Vẽ viền hộp dạ quang Cyber Neon
  feedbackCtx.strokeStyle = "#10b981";
  feedbackCtx.lineWidth = 3;
  feedbackCtx.shadowColor = "rgba(16, 185, 129, 0.85)";
  feedbackCtx.shadowBlur = 10;
  feedbackCtx.strokeRect(x1, y1, bw, bh);

  feedbackCtx.shadowBlur = 0;
  feedbackCtx.strokeStyle = "rgba(255, 255, 255, 0.6)";
  feedbackCtx.lineWidth = 1;
  feedbackCtx.setLineDash([6, 6]);
  feedbackCtx.strokeRect(x1 + 3, y1 + 3, Math.max(1, bw - 6), Math.max(1, bh - 6));
  feedbackCtx.setLineDash([]);

  // 4. Vẽ 4 góc chốt co giãn (Handles)
  const handles = [
    { x: x1, y: y1 },
    { x: x2, y: y1 },
    { x: x1, y: y2 },
    { x: x2, y: y2 }
  ];

  handles.forEach(pos => {
    feedbackCtx.beginPath();
    feedbackCtx.arc(pos.x, pos.y, 8, 0, Math.PI * 2);
    feedbackCtx.fillStyle = "#ffffff";
    feedbackCtx.fill();
    feedbackCtx.strokeStyle = "#10b981";
    feedbackCtx.lineWidth = 3;
    feedbackCtx.stroke();
  });

  // 5. Thẻ hiển thị kích thước vùng chọn
  const labelText = `🎯 VÙNG DẠY: ${Math.round(bw)} × ${Math.round(bh)}px`;
  feedbackCtx.font = "bold 13px 'Outfit', sans-serif";
  const textW = feedbackCtx.measureText(labelText).width;
  const tagH = 22;
  const tagY = (y1 >= tagH + 6) ? (y1 - tagH - 4) : (y1 + 4);

  feedbackCtx.fillStyle = "rgba(16, 185, 129, 0.95)";
  feedbackCtx.fillRect(x1, tagY, textW + 12, tagH);
  feedbackCtx.fillStyle = "#000000";
  feedbackCtx.fillText(labelText, x1 + 6, tagY + 15);

  // Đồng bộ tọa độ gửi lên Backend
  feedbackBBox = [Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)];
}

function getCanvasCoords(e) {
  const rect = feedbackCanvas.getBoundingClientRect();
  const scaleX = feedbackCanvas.width / rect.width;
  const scaleY = feedbackCanvas.height / rect.height;
  return {
    x: Math.max(0, Math.min(feedbackCanvas.width, (e.clientX - rect.left) * scaleX)),
    y: Math.max(0, Math.min(feedbackCanvas.height, (e.clientY - rect.top) * scaleY))
  };
}

function getHitMode(x, y) {
  const x1 = Math.min(boxState.x1, boxState.x2);
  const y1 = Math.min(boxState.y1, boxState.y2);
  const x2 = Math.max(boxState.x1, boxState.x2);
  const y2 = Math.max(boxState.y1, boxState.y2);

  const dist = (px, py) => Math.hypot(x - px, y - py);
  const hitRadius = HANDLE_RADIUS * 1.6;

  if (dist(x1, y1) <= hitRadius) return "resize-tl";
  if (dist(x2, y1) <= hitRadius) return "resize-tr";
  if (dist(x1, y2) <= hitRadius) return "resize-bl";
  if (dist(x2, y2) <= hitRadius) return "resize-br";

  if (x >= x1 && x <= x2 && y >= y1 && y <= y2) return "move";

  return "draw";
}

function setupBoxCanvasEvents() {
  if (boxEventsAttached) return;
  boxEventsAttached = true;

  feedbackCanvas.addEventListener("pointerdown", (e) => {
    feedbackCanvas.setPointerCapture(e.pointerId);
    const pos = getCanvasCoords(e);
    dragStart = pos;
    dragMode = getHitMode(pos.x, pos.y);

    dragBoxStart = {
      x1: Math.min(boxState.x1, boxState.x2),
      y1: Math.min(boxState.y1, boxState.y2),
      x2: Math.max(boxState.x1, boxState.x2),
      y2: Math.max(boxState.y1, boxState.y2)
    };

    if (dragMode === "draw") {
      boxState = { x1: pos.x, y1: pos.y, x2: pos.x, y2: pos.y };
      drawFeedbackCanvas();
    }
  });

  feedbackCanvas.addEventListener("pointermove", (e) => {
    const pos = getCanvasCoords(e);

    if (!dragMode) {
      const mode = getHitMode(pos.x, pos.y);
      if (mode === "resize-tl" || mode === "resize-br") {
        feedbackCanvas.style.cursor = "nwse-resize";
      } else if (mode === "resize-tr" || mode === "resize-bl") {
        feedbackCanvas.style.cursor = "nesw-resize";
      } else if (mode === "move") {
        feedbackCanvas.style.cursor = "move";
      } else {
        feedbackCanvas.style.cursor = "crosshair";
      }
      return;
    }

    const w = feedbackCanvas.width;
    const h = feedbackCanvas.height;

    if (dragMode === "draw") {
      boxState.x2 = pos.x;
      boxState.y2 = pos.y;
    } else if (dragMode === "move") {
      const dx = pos.x - dragStart.x;
      const dy = pos.y - dragStart.y;
      const bw = dragBoxStart.x2 - dragBoxStart.x1;
      const bh = dragBoxStart.y2 - dragBoxStart.y1;
      const nx1 = Math.max(0, Math.min(w - bw, dragBoxStart.x1 + dx));
      const ny1 = Math.max(0, Math.min(h - bh, dragBoxStart.y1 + dy));
      boxState = {
        x1: nx1,
        y1: ny1,
        x2: nx1 + bw,
        y2: ny1 + bh
      };
    } else if (dragMode === "resize-tl") {
      boxState.x1 = Math.max(0, Math.min(dragBoxStart.x2 - 20, pos.x));
      boxState.y1 = Math.max(0, Math.min(dragBoxStart.y2 - 20, pos.y));
      boxState.x2 = dragBoxStart.x2;
      boxState.y2 = dragBoxStart.y2;
    } else if (dragMode === "resize-tr") {
      boxState.x2 = Math.min(w, Math.max(dragBoxStart.x1 + 20, pos.x));
      boxState.y1 = Math.max(0, Math.min(dragBoxStart.y2 - 20, pos.y));
      boxState.x1 = dragBoxStart.x1;
      boxState.y2 = dragBoxStart.y2;
    } else if (dragMode === "resize-bl") {
      boxState.x1 = Math.max(0, Math.min(dragBoxStart.x2 - 20, pos.x));
      boxState.y2 = Math.min(h, Math.max(dragBoxStart.y1 + 20, pos.y));
      boxState.x2 = dragBoxStart.x2;
      boxState.y1 = dragBoxStart.y1;
    } else if (dragMode === "resize-br") {
      boxState.x2 = Math.min(w, Math.max(dragBoxStart.x1 + 20, pos.x));
      boxState.y2 = Math.min(h, Math.max(dragBoxStart.y1 + 20, pos.y));
      boxState.x1 = dragBoxStart.x1;
      boxState.y1 = dragBoxStart.y1;
    }

    drawFeedbackCanvas();
  });

  const endDrag = (e) => {
    if (!dragMode) return;
    try {
      feedbackCanvas.releasePointerCapture(e.pointerId);
    } catch (_) {}
    dragMode = null;

    const nx1 = Math.min(boxState.x1, boxState.x2);
    const ny1 = Math.min(boxState.y1, boxState.y2);
    const nx2 = Math.max(boxState.x1, boxState.x2);
    const ny2 = Math.max(boxState.y1, boxState.y2);

    if (nx2 - nx1 < 20 || ny2 - ny1 < 20) {
      if (initialBoxState) {
        boxState = { ...initialBoxState };
      } else {
        boxState = { x1: 50, y1: 50, x2: 250, y2: 250 };
      }
    } else {
      boxState = { x1: nx1, y1: ny1, x2: nx2, y2: ny2 };
    }

    drawFeedbackCanvas();
  };

  feedbackCanvas.addEventListener("pointerup", endDrag);
  feedbackCanvas.addEventListener("pointercancel", endDrag);

  // Nút khôi phục khung ban đầu
  const btnResetBox = document.getElementById("btn-reset-box");
  if (btnResetBox) {
    btnResetBox.addEventListener("click", () => {
      if (initialBoxState) {
        boxState = { ...initialBoxState };
        drawFeedbackCanvas();
        showToast("🔄 Đã khôi phục khung gợi ý ban đầu");
      }
    });
  }

  // Nút chọn toàn bộ khung ảnh
  const btnFullBox = document.getElementById("btn-full-box");
  if (btnFullBox) {
    btnFullBox.addEventListener("click", () => {
      boxState = {
        x1: 4,
        y1: 4,
        x2: feedbackCanvas.width - 4,
        y2: feedbackCanvas.height - 4
      };
      drawFeedbackCanvas();
      showToast("🔲 Đã mở rộng khung trọn vẹn bức ảnh");
    });
  }
}

function captureFrameForFeedback() {
  const offscreen = document.createElement("canvas");
  offscreen.width = videoElem.videoWidth || 640;
  offscreen.height = videoElem.videoHeight || 480;
  const octx = offscreen.getContext("2d");
  octx.drawImage(videoElem, 0, 0, offscreen.width, offscreen.height);

  return new Promise(resolve => {
    offscreen.toBlob(blob => {
      resolve({ blob, dataUrl: offscreen.toDataURL("image/jpeg", 0.9) });
    }, "image/jpeg", 0.9);
  });
}

async function openFeedbackModal() {
  if (!isStreaming) {
    alert("Camera chưa bật hoặc chưa sẵn sàng!");
    return;
  }

  const captured = await captureFrameForFeedback();
  feedbackBlob = captured.blob;

  const previewImg = document.getElementById("feedback-preview-img");
  previewImg.src = captured.dataUrl;

  const predLabel = document.getElementById("modal-predicted-label");
  let currentPred = "Chưa nhận diện được";
  if (currentBestTarget && currentBestVi) {
    currentPred = (currentLang === "vi") 
      ? `${currentBestVi} (${currentBestPrimary.toUpperCase()})` 
      : `${currentBestTarget} • ${currentBestVi}`;
  } else if (lastTranslation) {
    currentPred = lastTranslation;
  }
  predLabel.innerText = currentPred;

  feedbackBBox = null;
  if (latestDetections && latestDetections.length > 0) {
    const match = latestDetections.find(d => d.primary === currentBestPrimary) || latestDetections[0];
    if (match && match.box) {
      feedbackBBox = match.box;
    }
  }

  // Khởi tạo Canvas tương tác khoanh vùng
  const imgObj = new Image();
  imgObj.onload = () => {
    initBoxEditor(imgObj, feedbackBBox);
  };
  imgObj.src = captured.dataUrl;
  if (imgObj.complete && imgObj.naturalWidth > 0) {
    initBoxEditor(imgObj, feedbackBBox);
  }

  // Reset tìm kiếm & lựa chọn
  const searchInput = document.getElementById("feedback-search-input");
  const clearBtn = document.getElementById("btn-clear-class-search");
  if (searchInput) searchInput.value = "";
  if (clearBtn) clearBtn.style.display = "none";

  deselectClass();

  const customGroup = document.getElementById("custom-class-group");
  const customInput = document.getElementById("feedback-custom-input");
  if (customInput) customInput.value = "";
  if (customGroup) customGroup.style.display = "none";

  renderQuickChips();
  renderClassResults("");

  document.getElementById("modal-feedback-backdrop").style.display = "flex";
}

function closeFeedbackModal() {
  document.getElementById("modal-feedback-backdrop").style.display = "none";
}

function showToast(msg) {
  const toast = document.getElementById("toast-notification");
  if (!toast) return;
  toast.innerText = msg;
  toast.style.display = "block";
  setTimeout(() => {
    toast.style.display = "none";
  }, 3500);
}

async function submitFeedback() {
  const saveBtn = document.getElementById("btn-save-feedback");

  // Kiểm tra nếu người dùng đã gõ ô custom mà chưa bấm xác nhận
  const customGroup = document.getElementById("custom-class-group");
  const customInput = document.getElementById("feedback-custom-input");
  if (customGroup && customGroup.style.display !== "none" && customInput && customInput.value.trim()) {
    const val = customInput.value.trim();
    const newKey = removeVietnameseTones(val).toLowerCase().replace(/[^a-z0-9]+/g, "_");
    selectClass(newKey, val);
  }

  if (!selectedClassKey) {
    alert("Vui lòng tìm và bấm chọn đồ vật đúng thực tế!");
    return;
  }

  if (!feedbackBlob) {
    alert("Không tìm thấy ảnh chụp!");
    return;
  }

  saveBtn.disabled = true;
  saveBtn.innerText = "⏳ Đang lưu...";

  try {
    const formData = new FormData();
    formData.append("file", feedbackBlob, "frame.jpg");
    formData.append("predicted", document.getElementById("modal-predicted-label").innerText);
    formData.append("correct_key", selectedClassKey);
    formData.append("correct_vi", selectedClassVi || selectedClassKey);

    // Đồng bộ tọa độ khung box người dùng đã kéo chọn
    if (boxState && boxState.x2 > boxState.x1 && boxState.y2 > boxState.y1) {
      feedbackBBox = [
        Math.round(Math.min(boxState.x1, boxState.x2)),
        Math.round(Math.min(boxState.y1, boxState.y2)),
        Math.round(Math.max(boxState.x1, boxState.x2)),
        Math.round(Math.max(boxState.y1, boxState.y2))
      ];
    }

    if (feedbackBBox) {
      formData.append("bbox", JSON.stringify(feedbackBBox));
    }

    const res = await fetch("/api/feedback", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (res.ok) {
      closeFeedbackModal();
      showToast(`✅ Đã lưu mẫu #${data.total_samples}! Chạy 'python train_finetune.py' để mô hình ghi nhớ.`);
      const badge = document.getElementById("feedback-count-badge");
      if (badge) badge.innerText = `${data.total_samples} mẫu`;
      loadClasses();
    } else {
      alert("Lỗi lưu mẫu: " + (data.error || "Không rõ nguyên nhân"));
    }
  } catch (err) {
    console.error("Lỗi khi gửi feedback:", err);
    alert("Không thể kết nối đến máy chủ để lưu mẫu!");
  } finally {
    saveBtn.disabled = false;
    saveBtn.innerText = "💾 Lưu Mẫu Huấn Luyện";
  }
}

