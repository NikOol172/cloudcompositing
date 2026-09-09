// RunPod Studio Frontend JavaScript

let allMedia = [];
let allJobs = [];
let allPrompts = [];
let activeFilter = 'all';
let activePromptFilter = 'all';
let targetPromptInputId = null;
let currentPickerCallback = null;
let currentPickerType = 'all';
let pollingInterval = null;

let currentLang = 'en';
let systemInfo = { has_gpu: false, gpu_name: null, vram_mb: null };

const TRANSLATIONS = {
  en: {
    // Brand & Nav
    nav_studio: "AI Generation Studio",
    nav_gallery: "Media & Gallery",
    nav_prompts: "Prompt History",
    nav_jobs: "Queue & Jobs",
    nav_settings: "Settings & API",
    btn_mobile_access: "📱 Mobile Pod Access",
    btn_mobile_badge: "📱 Pod Mobile",
    mobile_modal_title: "📱 Mobile RunPod Access",
    mobile_modal_desc: "Scan this QR code with your smartphone camera to access and control your RunPod Studio directly from your mobile device:",
    mobile_url_label: "RunPod Mobile Web URL",
    btn_copy_url: "📋 Copy URL",
    btn_open_tab: "↗️ Open in New Tab",
    mobile_tip: "💡 Tip: Open this URL in Safari (iOS) or Chrome (Android) and add it to your home screen to use it like a native mobile app!",
    btn_close: "Close",
    gpu_detecting: "Detecting hardware...",
    gpu_cloud_only: "☁️ Cloud Mode (No GPU)",
    gpu_detected: "⚡ Pod GPU:",
    
    // Studio Header
    studio_title: "AI Creative Studio",
    studio_subtitle: "Generate, animate, transform, and customize your visual assets in high definition.",
    pipe_txt2vid: "🎬 Text-to-Video",
    pipe_img2vid: "🖼️ Image-to-Video",
    pipe_img2img: "🎨 Image-to-Image",
    pipe_faceswap: "🎭 Face Swap",
    pipe_vid2vid: "🎞️ Video-to-Video",
    pipe_txt2img: "✨ Text-to-Image",
    pipe_tts: "🎙️ Text-to-Speech",
    pipe_lora: "🎓 LoRA Training",
    pipe_enhance: "🧠 Prompt Enhancer",

    // Common form elements
    label_prompt: "Scene Prompt",
    btn_history: "📜 History",
    btn_enhance: "🧠 Optimize",
    video_engine: "Video Engine",
    initial_img_engine: "Initial Image Engine",
    gen_engine: "Generation Engine",
    engine_cloud: "🚀 RunPod Cloud (Flux 1 Schnell)",
    engine_local: "⚡ Pod GPU / Local",
    resolution: "Resolution",
    res_hd: "720p (High Quality HD)",
    res_fast: "480p (Fast Render)",
    duration: "Duration (seconds)",
    aspect_ratio: "Aspect Ratio",
    ratio_portrait: "Portrait 9:16 (Mobile / Reels)",
    ratio_landscape: "Landscape 16:9 (Cinema)",
    ratio_square: "Square 1:1 (Instagram)",
    chk_enhance_qwen: "Enable automatic prompt enhancement (Qwen3-32B)",
    output_filename: "Output Filename",
    btn_launch_t2v: "🚀 Generate Text-to-Video",

    // Image to Video
    label_source_media: "Source Image",
    drop_source_media: "🖼️ Drag and drop an image here or click to choose from gallery",
    btn_gallery: "Gallery",
    btn_upload: "Upload",
    btn_crop: "✂️ Crop",
    label_i2v_prompt: "Motion Animation Prompt",
    btn_launch_i2v: "🚀 Generate Image-to-Video",

    // Image to Image
    label_denoise: "Transformation Strength (Denoise):",
    label_local_model: "Local Model (HuggingFace / Diffusers)",
    btn_launch_i2i: "🎨 Generate Image-to-Image",

    // Face Swap
    label_target_media: "Target Video or Image",
    drop_target_media: "🎬 Drag a video or image here",
    label_source_face: "Source Face Photo",
    drop_source_face: "👤 Drag a face photo here",
    label_face_restorer: "Face Restoration Model",
    label_face_quality: "Processing Quality",
    btn_launch_faceswap: "🎭 Launch Face Swap",

    // Video to Video
    label_v2v_source: "Source Video",
    drop_v2v_source: "🎞️ Drag a video here (.mp4, .webm)",
    label_v2v_prompt: "Transformation / Restyling Prompt",
    btn_launch_v2v: "🎞️ Generate Video-to-Video",

    // Text to Image
    label_t2i_prompt: "Image Description Prompt",
    label_t2i_steps: "Inference Steps:",
    label_t2i_guidance: "Guidance Scale (CFG):",
    label_t2i_seed: "Seed (0 = Random):",
    btn_launch_t2i: "✨ Generate Text-to-Image",

    // Text to Speech
    label_tts_text: "Speech Text / Dialogue Script",
    label_tts_engine: "Voice Engine",
    label_tts_voice: "Voice Character / Profile",
    label_tts_speed: "Speech Rate / Speed:",
    label_tts_ref_wav: "Voice Cloning Reference Audio (.wav)",
    drop_tts_ref_wav: "🎙️ Select or upload a reference WAV voice sample for zero-shot cloning",
    btn_launch_tts: "🎙️ Synthesize Speech Audio",

    // LoRA Training
    label_lora_dataset: "Dataset / Training Folder",
    label_lora_prompt: "Trigger / Instance Prompt",
    label_lora_model: "Base Diffusion Model",
    label_lora_steps: "Training Steps:",
    lora_steps_help: "500 steps ≈ 10-15 min on standard GPU for 10 images.",
    label_lora_rank: "LoRA Rank (Capacity / Dimension)",
    lora_hw_title: "Hardware GPU Acceleration",
    lora_hw_desc: "VAE latents pre-cached in RAM to minimize GPU overhead.",
    lora_vram_title: "Adaptive VRAM Optimization",
    lora_vram_desc: "FP16 precision + Gradient Accumulation to prevent OOM.",
    btn_launch_lora: "🚀 Launch LoRA Training",

    // Auto-caption
    label_caption_engine: "Auto-Captioning Vision Model",
    label_caption_prefix: "Prefix / Concept Trigger",
    btn_launch_caption: "🏷️ Auto-Caption Dataset",

    // Prompt Enhancer
    label_enhance_prompt: "Base Idea / Rough Prompt to Enhance",
    label_enhance_style: "Creative Enhancement Style",
    btn_launch_enhance: "🧠 Enhance Prompt with Qwen3",

    // Live monitor
    live_title: "Live Execution Monitor",
    live_idle_title: "Ready for generation",
    live_idle_desc: "Configure parameters on the left and start an AI task to follow live logs and outputs.",
    live_time_elapsed: "Time Elapsed",
    live_status: "Status",

    // Gallery
    gallery_title: "Media Manager & Production Gallery",
    gallery_subtitle: "Access, preview, download, and reuse all your generated creations.",
    filter_all: "All",
    filter_videos: "Videos",
    filter_images: "Images",
    filter_audio: "Audio",
    btn_refresh_media: "🔄 Refresh",
    drop_upload_media: "Drag and drop your images or videos here to import them into your studio",
    empty_gallery: "No media generated yet. Start with the AI Studio!",

    // Prompts
    prompts_title: "Prompt Library & History",
    prompts_subtitle: "Save, search, reuse, and favorite your best prompts.",
    search_prompts: "Search prompts...",
    filter_all_prompts: "All Prompts",
    filter_fav_prompts: "Favorites ⭐",
    empty_prompts: "No saved prompts yet.",

    // Jobs
    jobs_title: "Queue & Job Activity",
    jobs_subtitle: "Track real-time RunPod tasks, execution times, and engine outputs.",
    btn_clear_jobs: "🗑️ Clear History",

    // Settings
    settings_title: "Studio Configuration & Endpoints",
    settings_subtitle: "Configure your RunPod API key, Hugging Face Token, and serverless endpoints.",
    label_hf_token: "Hugging Face Access Token (HF_TOKEN)",
    settings_hf_token_help: "Required for gated models like Lightricks/LTX-Video (~11 GB). Create a Read token on huggingface.co/settings/tokens and accept the model license terms.",
    models_manager_title: "Pod GPU Models & Hugging Face Weights",
    models_manager_desc: "Some cutting-edge models (such as LTX-Video 2.5 DiT) run directly on your Pod's GPU rather than serverless endpoints, and require their model weights downloaded.",
    btn_save_settings: "💾 Save Settings",
    settings_saved: "Settings saved successfully!",
    settings_local_model_help: "Default model loaded on Pod GPU in local mode."
  }
};


function toggleLanguage() {
  currentLang = 'en';
  localStorage.setItem('runpod_studio_lang', 'en');
  applyLanguage('en');
}

function applyLanguage(lang) {
  const dict = TRANSLATIONS[lang] || TRANSLATIONS.en;
  
  const labelEl = document.getElementById('lang-label');
  if (labelEl) labelEl.textContent = lang === 'en' ? '🌐 EN' : '🌐 FR';
  const mobileLabelEl = document.getElementById('mobile-lang-label');
  if (mobileLabelEl) mobileLabelEl.textContent = lang.toUpperCase();

  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (dict[key]) {
      el.textContent = dict[key];
    }
  });

  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const key = el.getAttribute('data-i18n-placeholder');
    if (dict[key]) {
      el.setAttribute('placeholder', dict[key]);
    }
  });

  applySystemInfo();
  renderModelsStatus();
  updateLtxBanners();
}

async function fetchSystemInfo() {
  try {
    const res = await fetch('/api/system/info');
    if (res.ok) {
      systemInfo = await res.json();
      applySystemInfo();
    }
  } catch (err) {
    console.warn('System info check failed', err);
  }
}

function applySystemInfo() {
  const gpuBadgeText = document.getElementById('sidebar-gpu-text');
  const gpuBadge = document.getElementById('sidebar-gpu-badge');
  const t2vLocalOpt = document.getElementById('t2v-local-opt');
  const i2iLocalOpt = document.getElementById('i2i-local-opt');
  const t2iLocalOpt = document.getElementById('t2i-local-opt');
  const loraHwTitle = document.getElementById('lora-hw-title');

  if (systemInfo && systemInfo.has_gpu) {
    const gpuLabel = systemInfo.gpu_name ? systemInfo.gpu_name : 'GPU Active';
    if (gpuBadgeText) gpuBadgeText.textContent = `⚡ ${gpuLabel}`;
    if (gpuBadge) gpuBadge.className = 'gpu-status-badge';

    const localText = `⚡ Pod GPU (${gpuLabel})`;
    if (t2vLocalOpt) { t2vLocalOpt.textContent = localText; t2vLocalOpt.disabled = false; }
    if (i2iLocalOpt) { i2iLocalOpt.textContent = localText; i2iLocalOpt.disabled = false; }
    if (t2iLocalOpt) { t2iLocalOpt.textContent = localText; t2iLocalOpt.disabled = false; }
    if (loraHwTitle) loraHwTitle.textContent = `${gpuLabel} Acceleration`;
  } else {
    const noGpuText = '☁️ Cloud Mode (No GPU)';
    if (gpuBadgeText) gpuBadgeText.textContent = noGpuText;
    if (gpuBadge) gpuBadge.className = 'gpu-status-badge cloud-only';

    const disabledText = '⚡ Pod GPU (Unavailable on CPU Pod)';
    if (t2vLocalOpt) { t2vLocalOpt.textContent = disabledText; t2vLocalOpt.disabled = true; }
    if (i2iLocalOpt) { i2iLocalOpt.textContent = disabledText; i2iLocalOpt.disabled = true; }
    if (t2iLocalOpt) { t2iLocalOpt.textContent = disabledText; t2iLocalOpt.disabled = true; }

    const t2vMode = document.getElementById('t2v-mode');
    if (t2vMode && t2vMode.value === 'local') t2vMode.value = 'cloud';
    const i2iMode = document.getElementById('i2i-mode');
    if (i2iMode && i2iMode.value === 'local') { i2iMode.value = 'cloud'; if (typeof toggleI2iMode === 'function') toggleI2iMode(); }
    const t2iMode = document.getElementById('t2i-mode');
    if (t2iMode && t2iMode.value === 'local') { t2iMode.value = 'cloud'; if (typeof toggleT2iMode === 'function') toggleT2iMode(); }
  }
}

function initApp() {
  initNavigation();
  initPipelineSwitcher();
  initDropzone();
  initForms();
  loadMedia();
  loadLoras();
  loadLoraDatasets();
  initLoraTraining();
  loadJobs();
  loadPrompts();
  loadSettings();
  fetchBalance(false);
  fetchSystemInfo();
  applyLanguage(currentLang);

  document.getElementById('t2v-video-model')?.addEventListener('change', updateLtxBanners);
  document.getElementById('i2v-video-model')?.addEventListener('change', updateLtxBanners);
  checkModelsStatus();

  if (!pollingInterval) {
    pollingInterval = setInterval(() => {
      pollJobs();
    }, 2500);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}

// ----------------------------------------------------
// Navigation & Tabs
// ----------------------------------------------------
function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const tabName = item.getAttribute('data-tab');
      switchTab(tabName);
    });
  });
}

function switchTab(tabName) {
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));

  const navItem = document.querySelector(`.nav-item[data-tab="${tabName}"]`);
  const tabPane = document.getElementById(`tab-${tabName}`);

  if (navItem) navItem.classList.add('active');
  if (tabPane) tabPane.classList.add('active');

  if (tabName === 'gallery') {
    loadMedia();
  } else if (tabName === 'prompts') {
    loadPrompts();
  } else if (tabName === 'jobs') {
    loadJobs();
  } else if (tabName === 'settings') {
    loadSettings();
    fetchBalance(false);
  }
}

// ----------------------------------------------------
// Pipeline Switcher
// ----------------------------------------------------
function initPipelineSwitcher() {
  const pills = document.querySelectorAll('.pill-btn');
  pills.forEach(pill => {
    pill.addEventListener('click', () => {
      const pipeline = pill.getAttribute('data-pipeline');
      switchPipeline(pipeline);
    });
  });
}

function switchPipeline(pipeline) {
  document.querySelectorAll('.pill-btn').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.pipeline-form').forEach(el => el.classList.remove('active'));

  const pill = document.querySelector(`.pill-btn[data-pipeline="${pipeline}"]`);
  const form = document.getElementById(`form-${pipeline}`);

  if (pill) pill.classList.add('active');
  if (form) form.classList.add('active');
}

// ----------------------------------------------------
// Media Management & Gallery
// ----------------------------------------------------
async function loadMedia() {
  try {
    const res = await fetch('/api/media');
    if (!res.ok) throw new Error('Failed to load media');
    allMedia = await res.json();
    document.getElementById('media-count').innerText = allMedia.length;
    renderMediaGrid();
  } catch (err) {
    console.error(err);
  }
}

function renderMediaGrid() {
  const container = document.getElementById('media-grid-container');
  container.innerHTML = '';

  const filtered = allMedia.filter(m => {
    if (activeFilter === 'videos') return m.is_video;
    if (activeFilter === 'audio') return m.is_audio;
    if (activeFilter === 'images') return !m.is_video && !m.is_audio;
    return true;
  });

  if (filtered.length === 0) {
    container.innerHTML = `
      <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-dim);">
        <p style="font-size: 1.1rem; margin-bottom: 8px;">No media found</p>
        <small>Import or generate your first creations from the studio.</small>
      </div>
    `;
    return;
  }

  filtered.forEach(item => {
    const card = document.createElement('div');
    card.className = 'media-card' + (item.is_audio ? ' media-card-audio' : '');

    let thumbHtml;
    let typeLabel = 'Image';
    if (item.is_video) {
      thumbHtml = `<video src="/api/media/${encodeURIComponent(item.name)}" muted preload="metadata"></video>
         <span class="media-badge">🎬 VIDEO</span>
         <div class="play-overlay-icon">▶</div>`;
      typeLabel = 'MP4';
    } else if (item.is_audio) {
      thumbHtml = `<div class="audio-card-visual">
         <div class="audio-wave-bars"><span></span><span></span><span></span><span></span><span></span></div>
         <span class="audio-center-icon">🎙️</span>
       </div>
       <span class="media-badge badge-audio">🎙️ AUDIO</span>`;
      typeLabel = 'Audio';
    } else {
      thumbHtml = `<img src="/api/media/${encodeURIComponent(item.name)}" loading="lazy" alt="${item.name}">
         <span class="media-badge">🖼️ IMAGE</span>`;
      typeLabel = 'Image';
    }

    card.innerHTML = `
      <div class="media-thumbnail-wrapper" onclick="openMediaModal('${item.name}')">
        ${thumbHtml}
      </div>
      <div class="media-info">
        <div class="media-name" title="${item.name}">${item.name}</div>
        <div class="media-meta">
          <span>${formatBytes(item.size_bytes)}</span>
          <span>${typeLabel}</span>
        </div>
        <div class="media-actions">
          <button class="btn-secondary btn-sm" onclick="reuseMedia('${item.name}', ${item.is_video}, ${item.is_audio || false})">⚡ Reuse</button>
          <button class="btn-secondary btn-sm" onclick="downloadMedia('${item.name}')">⬇️</button>
          <button class="btn-danger btn-sm" onclick="deleteMedia('${item.name}')">🗑️</button>
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

// Gallery Filters
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeFilter = btn.getAttribute('data-filter');
    renderMediaGrid();
  });
});

// Drag & Drop Upload
function initDropzone() {
  const dropzone = document.getElementById('gallery-dropzone');
  if (dropzone) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
    });

    dropzone.addEventListener('drop', e => {
      const dt = e.dataTransfer;
      const files = dt.files;
      uploadFiles(files);
    });

    dropzone.addEventListener('click', () => {
      document.getElementById('file-upload-input').click();
    });
  }

  // Setup drag & drop for Studio preview selector boxes
  setupBoxDropzone('fs-source-preview', 'fs-source-path', 'image');
  setupBoxDropzone('fs-target-preview', 'fs-target-path', 'all');
  setupBoxDropzone('i2v-source-preview', 'i2v-image-path', 'image');
  setupBoxDropzone('v2v-source-preview', 'v2v-video-path', 'video');
}

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

function setupBoxDropzone(boxId, targetInputId, allowedType) {
  const box = document.getElementById(boxId);
  if (!box) return;

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    box.addEventListener(eventName, preventDefaults, false);
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    box.addEventListener(eventName, () => box.classList.add('dragover'), false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    box.addEventListener(eventName, () => box.classList.remove('dragover'), false);
  });

  box.addEventListener('drop', e => {
    const file = e.dataTransfer?.files?.[0];
    if (file) {
      uploadAndSelectFile(file, targetInputId, boxId);
    }
  });
}

function handleFileUpload(event) {
  const files = event.target.files;
  uploadFiles(files);
}

// Direct Upload Trigger for Studio Buttons
function triggerDirectUpload(targetInputId, previewContainerId, allowedType = 'all') {
  const input = document.createElement('input');
  input.type = 'file';
  input.style.display = 'none';

  if (allowedType === 'image') {
    input.accept = 'image/png,image/jpeg,image/webp,image/jpg';
  } else if (allowedType === 'video') {
    input.accept = 'video/mp4,video/webm,video/quicktime,video/x-matroska';
  } else {
    input.accept = 'image/*,video/*';
  }

  input.onchange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await uploadAndSelectFile(file, targetInputId, previewContainerId);
    input.remove();
  };

  document.body.appendChild(input);
  input.click();
}

function renderMediaPreviewHtml(filename, isVideo) {
  if (isVideo) {
    const encoded = encodeURIComponent(filename);
    const escaped = escapeHtml(filename);
    return `<video src="/api/media/${encoded}" autoplay loop muted playsinline onerror="this.outerHTML='<div class=\\'video-codec-box\\'>🎬 <strong>${escaped}</strong><br><small>Video file ready for RunPod</small></div>'"></video>`;
  } else {
    return `<img src="/api/media/${encodeURIComponent(filename)}" alt="${filename}">`;
  }
}

async function uploadAndSelectFile(file, targetInputId, previewContainerId) {
  showToast(`Uploading ${file.name}...`, 'info');
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('Upload failed');
    const data = await res.json();
    const uploadedName = data.filename || file.name;

    const inputEl = document.getElementById(targetInputId);
    if (inputEl) inputEl.value = uploadedName;

    const previewEl = document.getElementById(previewContainerId);
    if (previewEl) {
      const isVideo = file.type.startsWith('video/') || /\.(mp4|webm|mov|mkv)$/i.test(uploadedName);
      previewEl.innerHTML = renderMediaPreviewHtml(uploadedName, isVideo);
    }

    showToast(`✓ ${uploadedName} uploaded and selected`, 'success');
    await loadMedia();
  } catch (err) {
    showToast(`Erreur : ${err.message}`, 'error');
  }
}

async function uploadFiles(files) {
  if (!files || files.length === 0) return;

  for (let file of files) {
    const formData = new FormData();
    formData.append('file', file);

    showToast(`Uploading ${file.name}...`, 'info');
    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('Upload failed');
      showToast(`${file.name} imported successfully`, 'success');
    } catch (err) {
      showToast(`Erreur : ${err.message}`, 'error');
    }
  }

  loadMedia();
}

async function deleteMedia(filename) {
  if (!confirm(`Permanently delete '${filename}'?`)) return;

  try {
    const res = await fetch(`/api/media/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to delete file');
    showToast(`File ${filename} deleted`, 'success');
    loadMedia();
  } catch (err) {
    showToast(`Erreur : ${err.message}`, 'error');
  }
}

function downloadMedia(filename) {
  const a = document.createElement('a');
  a.href = `/api/media/${encodeURIComponent(filename)}`;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// ----------------------------------------------------
// Media Picker & Re-use
// ----------------------------------------------------
// Media Picker for Forms
// ----------------------------------------------------
let currentPickerAllowedType = 'all';

async function openMediaPicker(targetInputId, previewContainerId, allowedType = 'all') {
  currentPickerAllowedType = allowedType;
  currentPickerCallback = (filename, isVideo) => {
    const inputEl = document.getElementById(targetInputId);
    if (inputEl) inputEl.value = filename;
    const container = document.getElementById(previewContainerId);
    if (container) {
      container.innerHTML = renderMediaPreviewHtml(filename, isVideo);
    }
    closePickerModal();
  };

  const titleEl = document.getElementById('picker-modal-title');
  if (titleEl) {
    if (allowedType === 'image') titleEl.innerText = '🖼️ Choose Source Image';
    else if (allowedType === 'video') titleEl.innerText = '🎬 Choose Source Video';
    else if (allowedType === 'audio') titleEl.innerText = '🎙️ Choose Audio Clip (XTTS Reference)';
    else titleEl.innerText = '📁 Choose Source Media';
  }

  const searchInput = document.getElementById('picker-search-input');
  if (searchInput) searchInput.value = '';

  // Refresh file list from disk
  await loadMedia();
  renderPickerGrid('');

  document.getElementById('picker-modal').classList.add('active');
}

function filterPickerMedia() {
  const query = (document.getElementById('picker-search-input')?.value || '').toLowerCase().trim();
  renderPickerGrid(query);
}

function renderPickerGrid(searchQuery = '') {
  const container = document.getElementById('picker-grid-container');
  if (!container) return;
  container.innerHTML = '';

  let filtered = allMedia.filter(m => {
    if (currentPickerAllowedType === 'video') return m.is_video;
    if (currentPickerAllowedType === 'image') return !m.is_video && !m.is_audio;
    if (currentPickerAllowedType === 'audio') return m.is_audio;
    return true;
  });

  if (searchQuery) {
    filtered = filtered.filter(m => m.name.toLowerCase().includes(searchQuery));
  }

  if (filtered.length === 0) {
    container.innerHTML = `
      <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-dim);">
        <p style="font-size: 1.1rem; margin-bottom: 8px;">No media found</p>
        <small>Generated or imported files will appear here.</small>
      </div>
    `;
    return;
  }

  filtered.forEach(item => {
    const card = document.createElement('div');
    card.className = 'media-card';
    card.style.cursor = 'pointer';
    card.onclick = () => currentPickerCallback(item.name, item.is_video);

    let thumbHtml;
    if (item.is_video) {
      thumbHtml = `<video src="/api/media/${encodeURIComponent(item.name)}" muted playsinline onerror="this.outerHTML='<div class=\\'video-codec-box\\'>🎬</div>'"></video><span class="media-badge">VIDEO</span>`;
    } else if (item.is_audio) {
      thumbHtml = `<div class="audio-card-visual" style="height: 100px;"><span style="font-size: 2.5rem;">🎙️</span></div><span class="media-badge badge-audio">AUDIO</span>`;
    } else {
      thumbHtml = `<img src="/api/media/${encodeURIComponent(item.name)}" alt="${item.name}"><span class="media-badge">IMAGE</span>`;
    }

    card.innerHTML = `
      <div class="media-thumbnail-wrapper">${thumbHtml}</div>
      <div class="media-info">
        <div class="media-name" title="${item.name}">${item.name}</div>
        <div class="media-meta"><span>${formatBytes(item.size_bytes)}</span></div>
      </div>
    `;
    container.appendChild(card);
  });
}

function closePickerModal() {
  document.getElementById('picker-modal').classList.remove('active');
}

function useAsFaceSource(filename) {
  switchTab('studio');
  switchPipeline('faceswap');
  const input = document.getElementById('fs-source-path');
  if (input) input.value = filename;
  const preview = document.getElementById('fs-source-preview');
  if (preview) preview.innerHTML = renderMediaPreviewHtml(filename, false);
}

function useAsFaceTarget(filename, isVideo) {
  switchTab('studio');
  switchPipeline('faceswap');
  const input = document.getElementById('fs-target-path');
  if (input) input.value = filename;
  const preview = document.getElementById('fs-target-preview');
  if (preview) preview.innerHTML = renderMediaPreviewHtml(filename, isVideo);
}

function useAsImageToVideo(filename) {
  switchTab('studio');
  switchPipeline('img2vid');
  const input = document.getElementById('i2v-image-path');
  if (input) input.value = filename;
  const preview = document.getElementById('i2v-source-preview');
  if (preview) preview.innerHTML = renderMediaPreviewHtml(filename, false);
}

function useAsVideoToVideo(filename) {
  switchTab('studio');
  switchPipeline('vid2vid');
  const input = document.getElementById('v2v-video-path');
  if (input) input.value = filename;
  const preview = document.getElementById('v2v-source-preview');
  if (preview) preview.innerHTML = renderMediaPreviewHtml(filename, true);
}

// ----------------------------------------------------
// Media Modal View & Player
// ----------------------------------------------------
function openMediaModal(filename) {
  const item = allMedia.find(m => m.name === filename);
  if (!item) return;

  const modal = document.getElementById('media-modal');
  document.getElementById('modal-title').innerText = item.name;

  const body = document.getElementById('modal-body');
  if (item.is_video) {
    body.innerHTML = `<video src="/api/media/${encodeURIComponent(item.name)}" controls autoplay style="max-height: 65vh; width: 100%;"></video>`;
  } else if (item.is_audio) {
    body.innerHTML = `
      <div class="audio-modal-player">
        <div class="audio-modal-icon">🎙️</div>
        <h3 class="audio-modal-name">${escapeHtml(item.name)}</h3>
        <audio src="/api/media/${encodeURIComponent(item.name)}" controls autoplay style="width: 100%; max-width: 480px; margin-top: 15px;"></audio>
      </div>`;
  } else {
    body.innerHTML = `<img src="/api/media/${encodeURIComponent(item.name)}" alt="${item.name}" style="max-height: 65vh; object-fit: contain;">`;
  }

  const footer = document.getElementById('modal-footer');
  if (item.is_video) {
    footer.innerHTML = `
      <button class="btn-secondary" onclick="openCropModal('${item.name}', 'vid2vid-source'); closeMediaModal();">✂️ Crop</button>
      <button class="btn-secondary" onclick="useAsVideoToVideo('${item.name}'); closeMediaModal();">🎞️ Transform (Vid2Vid)</button>
      <button class="btn-secondary" onclick="useAsFaceTarget('${item.name}', true); closeMediaModal();">🎭 Face Swap Target</button>
      <button class="btn-primary" onclick="downloadMedia('${item.name}')">⬇️ Download</button>
    `;
  } else if (item.is_audio) {
    footer.innerHTML = `
      <button class="btn-secondary" onclick="useAsTtsReference('${item.name}'); closeMediaModal();">🎙️ Reference Voice (XTTS)</button>
      <button class="btn-primary" onclick="downloadMedia('${item.name}')">⬇️ Download</button>
    `;
  } else {
    footer.innerHTML = `
      <button class="btn-secondary" onclick="openCropModal('${item.name}', 'faceswap-source'); closeMediaModal();">✂️ Crop</button>
      <button class="btn-secondary" onclick="useAsImageToImage('${item.name}'); closeMediaModal();">🎨 Image-to-Image</button>
      <button class="btn-secondary" onclick="useAsFaceSource('${item.name}'); closeMediaModal();">🎭 Source Face</button>
      <button class="btn-secondary" onclick="useAsFaceTarget('${item.name}', false); closeMediaModal();">🎯 Face Swap Target</button>
      <button class="btn-secondary" onclick="useAsImageToVideo('${item.name}'); closeMediaModal();">🖼️ Animate (Img2Vid)</button>
      <button class="btn-primary" onclick="downloadMedia('${item.name}')">⬇️ Download</button>
    `;
  }

  modal.classList.add('active');
}

function closeMediaModal() {
  const modal = document.getElementById('media-modal');
  modal.classList.remove('active');
  const body = document.getElementById('modal-body');
  body.innerHTML = '';
}

function useAsImageToImage(filename) {
  switchTab('studio');
  switchPipeline('img2img');
  const inputEl = document.getElementById('i2i-image-path');
  const previewEl = document.getElementById('i2i-source-preview');
  if (inputEl && previewEl) {
    inputEl.value = filename;
    previewEl.innerHTML = `<img src="/api/media/${encodeURIComponent(filename)}" alt="Source">`;
  }
  showToast(`Image selected for Image-to-Image: ${filename}`, 'success');
}

// ----------------------------------------------------
// Interactive Crop & Resize Tool
// ----------------------------------------------------
let cropState = {
  activeFilename: '',
  isVideo: false,
  naturalW: 512,
  naturalH: 512,
  mediaLeft: 0,
  mediaTop: 0,
  mediaWidth: 300,
  mediaHeight: 300,
  boxLeft: 0,
  boxTop: 0,
  boxWidth: 200,
  boxHeight: 200,
  ratio: '1:1',
  isDragging: false,
  isResizing: false,
  resizeDir: '',
  startX: 0,
  startY: 0,
  startBoxLeft: 0,
  startBoxTop: 0,
  startBoxW: 0,
  startBoxH: 0,
};

function cropCurrentField(inputId) {
  const inputEl = document.getElementById(inputId);
  const filename = inputEl ? inputEl.value.trim() : '';
  if (!filename) {
    showToast('Please select or upload media first', 'info');
    return;
  }

  let defaultAction = 'faceswap-target';
  if (inputId === 'fs-source-path') defaultAction = 'faceswap-source';
  else if (inputId === 'fs-target-path') defaultAction = 'faceswap-target';
  else if (inputId === 'i2v-image-path') defaultAction = 'img2vid-source';
  else if (inputId === 'i2i-image-path') defaultAction = 'img2img-source';
  else if (inputId === 'v2v-video-path') defaultAction = 'vid2vid-source';

  openCropModal(filename, defaultAction);
}

function openCropModal(filename, defaultAction = 'faceswap-target') {
  cropState.activeFilename = filename;
  const isVideo = /\.(mp4|webm|mov|mkv)$/i.test(filename);
  cropState.isVideo = isVideo;

  const actionSelect = document.getElementById('crop-target-action');
  if (actionSelect) actionSelect.value = defaultAction;

  const titleEl = document.getElementById('crop-modal-title');
  if (titleEl) titleEl.innerText = `✂️ Crop: ${filename}`;

  const wrapper = document.getElementById('crop-media-wrapper');
  wrapper.innerHTML = '';

  const videoControls = document.getElementById('crop-video-controls');
  if (videoControls) videoControls.style.display = isVideo ? 'flex' : 'none';

  if (isVideo) {
    const video = document.createElement('video');
    video.src = `/api/media/${encodeURIComponent(filename)}`;
    video.playsInline = true;
    video.muted = true;
    video.autoplay = false;
    video.controls = false;

    const onReady = () => {
      cropState.naturalW = video.videoWidth || 1080;
      cropState.naturalH = video.videoHeight || 1920;
      initCropGeometry(video);
      updateCropVideoTimeline(video);
    };

    video.onloadedmetadata = onReady;
    video.onloadeddata = onReady;
    video.oncanplay = onReady;
    video.ontimeupdate = () => updateCropVideoTimeline(video);
    wrapper.appendChild(video);
  } else {
    const img = document.createElement('img');
    img.src = `/api/media/${encodeURIComponent(filename)}`;
    img.onload = () => {
      cropState.naturalW = img.naturalWidth || 1024;
      cropState.naturalH = img.naturalHeight || 1024;
      initCropGeometry(img);
    };
    wrapper.appendChild(img);
  }

  setCropRatio('1:1');
  document.getElementById('crop-modal').classList.add('active');
  setupCropListeners();
}

function closeCropModal() {
  const modal = document.getElementById('crop-modal');
  modal.classList.remove('active');
  const video = document.querySelector('#crop-media-wrapper video');
  if (video) video.pause();
  document.getElementById('crop-media-wrapper').innerHTML = '';
}

function initCropGeometry(mediaEl) {
  const container = document.getElementById('crop-viewport-container');
  if (!container || !mediaEl) return;

  requestAnimationFrame(() => {
    const contRect = container.getBoundingClientRect();
    const mediaRect = mediaEl.getBoundingClientRect();

    if (mediaRect.width === 0 || mediaRect.height === 0) {
      setTimeout(() => initCropGeometry(mediaEl), 100);
      return;
    }

    cropState.mediaLeft = mediaRect.left - contRect.left;
    cropState.mediaTop = mediaRect.top - contRect.top;
    cropState.mediaWidth = mediaRect.width;
    cropState.mediaHeight = mediaRect.height;

    // Calculate initial size (1:1 square centered)
    const initialSize = Math.min(cropState.mediaWidth, cropState.mediaHeight) * 0.85;
    cropState.boxWidth = initialSize;
    cropState.boxHeight = initialSize;
    cropState.boxLeft = cropState.mediaLeft + (cropState.mediaWidth - initialSize) / 2;
    // Position towards top where face is located
    const topOffset = (cropState.mediaHeight - initialSize) * 0.25;
    cropState.boxTop = cropState.mediaTop + Math.max(0, topOffset);

    applyCropBoxStyles();
    updateCropCoords();
  });
}

function applyCropBoxStyles() {
  const cropBox = document.getElementById('crop-box');
  if (!cropBox) return;
  cropBox.style.left = `${cropState.boxLeft}px`;
  cropBox.style.top = `${cropState.boxTop}px`;
  cropBox.style.width = `${cropState.boxWidth}px`;
  cropBox.style.height = `${cropState.boxHeight}px`;
}

function updateCropCoords() {
  if (!cropState.mediaWidth || !cropState.mediaHeight) return;

  const scaleX = cropState.naturalW / cropState.mediaWidth;
  const scaleY = cropState.naturalH / cropState.mediaHeight;

  const x = Math.max(0, Math.round((cropState.boxLeft - cropState.mediaLeft) * scaleX));
  const y = Math.max(0, Math.round((cropState.boxTop - cropState.mediaTop) * scaleY));
  const w = Math.min(cropState.naturalW - x, Math.round(cropState.boxWidth * scaleX));
  const h = Math.min(cropState.naturalH - y, Math.round(cropState.boxHeight * scaleY));

  document.getElementById('crop-val-x').innerText = x;
  document.getElementById('crop-val-y').innerText = y;
  document.getElementById('crop-val-w').innerText = w;
  document.getElementById('crop-val-h').innerText = h;
}

function setCropRatio(ratio) {
  cropState.ratio = ratio;
  document.querySelectorAll('.crop-preset-pills .pill-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-ratio') === ratio);
  });

  if (ratio === '1:1') {
    const size = Math.min(cropState.boxWidth, cropState.boxHeight);
    cropState.boxWidth = size;
    cropState.boxHeight = size;
  } else if (ratio === '9:16') {
    cropState.boxHeight = cropState.boxWidth * (16 / 9);
  } else if (ratio === '16:9') {
    cropState.boxHeight = cropState.boxWidth * (9 / 16);
  }

  // Check boundaries
  const maxW = cropState.mediaLeft + cropState.mediaWidth - cropState.boxLeft;
  const maxH = cropState.mediaTop + cropState.mediaHeight - cropState.boxTop;
  if (cropState.boxWidth > maxW) cropState.boxWidth = maxW;
  if (cropState.boxHeight > maxH) cropState.boxHeight = maxH;

  applyCropBoxStyles();
  updateCropCoords();
}

let cropListenersAttached = false;
function setupCropListeners() {
  if (cropListenersAttached) return;
  cropListenersAttached = true;

  const container = document.getElementById('crop-viewport-container');
  const cropBox = document.getElementById('crop-box');

  const onPointerDown = (e) => {
    if (!document.getElementById('crop-modal').classList.contains('active')) return;
    const target = e.target;
    if (target.classList.contains('crop-handle')) {
      e.preventDefault();
      cropState.isResizing = true;
      cropState.resizeDir = target.getAttribute('data-dir');
      cropState.startX = e.clientX;
      cropState.startY = e.clientY;
      cropState.startBoxLeft = cropState.boxLeft;
      cropState.startBoxTop = cropState.boxTop;
      cropState.startBoxW = cropState.boxWidth;
      cropState.startBoxH = cropState.boxHeight;
    } else if (target === cropBox || cropBox.contains(target)) {
      e.preventDefault();
      cropState.isDragging = true;
      cropState.startX = e.clientX;
      cropState.startY = e.clientY;
      cropState.startBoxLeft = cropState.boxLeft;
      cropState.startBoxTop = cropState.boxTop;
    }
  };

  const onPointerMove = (e) => {
    if (cropState.isDragging) {
      const dx = e.clientX - cropState.startX;
      const dy = e.clientY - cropState.startY;

      let newLeft = cropState.startBoxLeft + dx;
      let newTop = cropState.startBoxTop + dy;

      const minLeft = cropState.mediaLeft;
      const maxLeft = cropState.mediaLeft + cropState.mediaWidth - cropState.boxWidth;
      const minTop = cropState.mediaTop;
      const maxTop = cropState.mediaTop + cropState.mediaHeight - cropState.boxHeight;

      cropState.boxLeft = Math.max(minLeft, Math.min(maxLeft, newLeft));
      cropState.boxTop = Math.max(minTop, Math.min(maxTop, newTop));

      applyCropBoxStyles();
      updateCropCoords();
    } else if (cropState.isResizing) {
      const dx = e.clientX - cropState.startX;
      const dy = e.clientY - cropState.startY;
      const dir = cropState.resizeDir;

      let newW = cropState.startBoxW;
      let newH = cropState.startBoxH;
      let newLeft = cropState.startBoxLeft;
      let newTop = cropState.startBoxTop;

      if (dir === 'se') {
        newW = cropState.startBoxW + dx;
        if (cropState.ratio === '1:1') newH = newW;
        else if (cropState.ratio === '9:16') newH = newW * (16 / 9);
        else if (cropState.ratio === '16:9') newH = newW * (9 / 16);
        else newH = cropState.startBoxH + dy;
      } else if (dir === 'sw') {
        newW = cropState.startBoxW - dx;
        newLeft = cropState.startBoxLeft + dx;
        if (cropState.ratio === '1:1') newH = newW;
        else if (cropState.ratio === '9:16') newH = newW * (16 / 9);
        else if (cropState.ratio === '16:9') newH = newW * (9 / 16);
        else newH = cropState.startBoxH + dy;
      } else if (dir === 'ne') {
        newW = cropState.startBoxW + dx;
        if (cropState.ratio === '1:1') {
          newH = newW;
          newTop = cropState.startBoxTop - (newH - cropState.startBoxH);
        } else {
          newH = cropState.startBoxH - dy;
          newTop = cropState.startBoxTop + dy;
        }
      } else if (dir === 'nw') {
        newW = cropState.startBoxW - dx;
        newLeft = cropState.startBoxLeft + dx;
        if (cropState.ratio === '1:1') {
          newH = newW;
          newTop = cropState.startBoxTop - (newH - cropState.startBoxH);
        } else {
          newH = cropState.startBoxH - dy;
          newTop = cropState.startBoxTop + dy;
        }
      }

      if (newW >= 40 && newH >= 40) {
        if (newLeft >= cropState.mediaLeft && newLeft + newW <= cropState.mediaLeft + cropState.mediaWidth + 2) {
          cropState.boxLeft = newLeft;
          cropState.boxWidth = newW;
        }
        if (newTop >= cropState.mediaTop && newTop + newH <= cropState.mediaTop + cropState.mediaHeight + 2) {
          cropState.boxTop = newTop;
          cropState.boxHeight = newH;
        }
      }

      applyCropBoxStyles();
      updateCropCoords();
    }
  };

  const onPointerUp = () => {
    cropState.isDragging = false;
    cropState.isResizing = false;
  };

  window.addEventListener('mousedown', onPointerDown);
  window.addEventListener('mousemove', onPointerMove);
  window.addEventListener('mouseup', onPointerUp);
  window.addEventListener('touchstart', (e) => {
    if (e.touches.length === 1) onPointerDown(e.touches[0]);
  }, { passive: false });
  window.addEventListener('touchmove', (e) => {
    if (e.touches.length === 1) onPointerMove(e.touches[0]);
  }, { passive: false });
  window.addEventListener('touchend', onPointerUp);
}

function updateCropVideoTimeline(video) {
  const seek = document.getElementById('crop-video-seek');
  const label = document.getElementById('crop-time-label');
  if (seek && video.duration) {
    seek.value = (video.currentTime / video.duration) * 100;
  }
  if (label) {
    label.innerText = `${formatTime(video.currentTime)} / ${formatTime(video.duration || 0)}`;
  }
}

function seekCropVideo(percent) {
  const video = document.querySelector('#crop-media-wrapper video');
  if (video && video.duration) {
    video.currentTime = (percent / 100) * video.duration;
  }
}

function toggleCropPlay() {
  const video = document.querySelector('#crop-media-wrapper video');
  const btn = document.getElementById('btn-crop-play-pause');
  if (!video) return;
  if (video.paused) {
    video.play();
    if (btn) btn.innerText = '⏸ Pause';
  } else {
    video.pause();
    if (btn) btn.innerText = '▶ Lecture';
  }
}

async function executeCrop() {
  if (!cropState.activeFilename) return;

  const btn = document.getElementById('btn-submit-crop');
  const btnText = document.getElementById('btn-crop-text');
  btn.disabled = true;
  btnText.innerText = '⏳ Processing FFmpeg...';

  const scaleX = cropState.naturalW / cropState.mediaWidth;
  const scaleY = cropState.naturalH / cropState.mediaHeight;

  const x = Math.max(0, Math.round((cropState.boxLeft - cropState.mediaLeft) * scaleX));
  const y = Math.max(0, Math.round((cropState.boxTop - cropState.mediaTop) * scaleY));
  const width = Math.min(cropState.naturalW - x, Math.round(cropState.boxWidth * scaleX));
  const height = Math.min(cropState.naturalH - y, Math.round(cropState.boxHeight * scaleY));

  const resVal = document.getElementById('crop-target-resolution').value;
  let target_width = null;
  let target_height = null;

  if (resVal !== 'original') {
    const [tw, th] = resVal.split('x').map(Number);
    target_width = tw;
    target_height = th;
  }

  const payload = {
    source: cropState.activeFilename,
    x,
    y,
    width,
    height,
    target_width,
    target_height,
  };

  try {
    const res = await fetch('/api/crop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error('Error during crop operation');
    const data = await res.json();
    const newFilename = data.filename;

    showToast(`✓ Crop successful: ${newFilename}`, 'success');
    await loadMedia();

    // Appliquer l'action choisie
    const action = document.getElementById('crop-target-action').value;
    if (action === 'faceswap-target') {
      useAsFaceTarget(newFilename, data.is_video);
    } else if (action === 'faceswap-source') {
      useAsFaceSource(newFilename);
    } else if (action === 'img2img-source') {
      useAsImageToImage(newFilename);
    } else if (action === 'img2vid-source') {
      useAsImageToVideo(newFilename);
    } else if (action === 'vid2vid-source') {
      useAsVideoToVideo(newFilename);
    }

    closeCropModal();
  } catch (err) {
    showToast(`Erreur : ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btnText.innerText = '✂️ Apply & Crop';
  }
}

function toggleI2iMode() {
  const mode = document.getElementById('i2i-mode').value;
  const localGroup = document.getElementById('i2i-local-model-group');
  const submitText = document.getElementById('i2i-submit-text');
  const stepsInput = document.getElementById('i2i-steps');
  const outputInput = document.getElementById('i2i-output');

  if (mode === 'local') {
    if (localGroup) localGroup.style.display = 'block';
    if (submitText) submitText.textContent = '💻 Generate Locally (GPU Diffusers)';
    if (stepsInput && stepsInput.value === '4') stepsInput.value = '4';
    if (outputInput && outputInput.value === 'pickleball_player.png') outputInput.value = 'local_img2img.png';
  } else {
    if (localGroup) localGroup.style.display = 'none';
    if (submitText) submitText.textContent = '🎨 Generate Image-to-Image';
    if (outputInput && outputInput.value === 'local_img2img.png') outputInput.value = 'pickleball_player.png';
  }
}

function toggleT2iMode() {
  const mode = document.getElementById('t2i-mode').value;
  const localGroup = document.getElementById('t2i-local-model-group');
  const submitText = document.getElementById('t2i-submit-text');
  const stepsInput = document.getElementById('t2i-steps');
  const outputInput = document.getElementById('t2i-output');

  if (mode === 'local') {
    if (localGroup) localGroup.style.display = 'block';
    if (submitText) submitText.textContent = '💻 Generate Locally (GPU Diffusers)';
    if (stepsInput && stepsInput.value === '4') stepsInput.value = '2';
    if (outputInput && outputInput.value === 'flux_image.png') outputInput.value = 'local_sdxl.png';
  } else {
    if (localGroup) localGroup.style.display = 'none';
    if (submitText) submitText.textContent = '✨ Generate Image (Flux)';
    if (stepsInput && stepsInput.value === '2') stepsInput.value = '4';
    if (outputInput && outputInput.value === 'local_sdxl.png') outputInput.value = 'flux_image.png';
  }
}

// ----------------------------------------------------
// Form Submissions & Pipelines
// ----------------------------------------------------
function initForms() {
  // Text-to-Video
  document.getElementById('form-txt2vid').addEventListener('submit', async e => {
    e.preventDefault();
    const prompt = document.getElementById('t2v-prompt').value.trim();
    if (!prompt) return showToast('Please enter a prompt', 'error');

    const [width, height] = document.getElementById('t2v-format').value.split('x').map(Number);
    const isLocal = document.getElementById('t2v-mode') ? document.getElementById('t2v-mode').value === 'local' : false;
    const videoModel = document.getElementById('t2v-video-model') ? document.getElementById('t2v-video-model').value : 'wan-2-5';
    const payload = {
      prompt,
      model: videoModel,
      resolution: document.getElementById('t2v-resolution').value,
      duration: Number(document.getElementById('t2v-duration').value),
      enhance_prompt: document.getElementById('t2v-enhance-prompt').checked,
      width,
      height,
      output: document.getElementById('t2v-output-video').value.trim() || 'studio_video.mp4',
      local: isLocal,
    };

    submitJob('txt2vid', payload, e.target);
  });

  // Image-to-Video
  document.getElementById('form-img2vid').addEventListener('submit', async e => {
    e.preventDefault();
    const image = document.getElementById('i2v-image-path').value.trim();
    if (!image) return showToast('Please select a source image', 'error');

    const videoModel = document.getElementById('i2v-video-model') ? document.getElementById('i2v-video-model').value : 'wan-2-5';
    const payload = {
      image,
      model: videoModel,
      prompt: document.getElementById('i2v-prompt').value.trim(),
      resolution: document.getElementById('i2v-resolution').value,
      duration: Number(document.getElementById('i2v-duration').value),
      output: document.getElementById('i2v-output').value.trim() || 'animation_image.mp4',
    };

    submitJob('img2vid', payload, e.target);
  });

  // Image-to-Image
  document.getElementById('form-img2img').addEventListener('submit', async e => {
    e.preventDefault();
    const image = document.getElementById('i2i-image-path').value.trim();
    if (!image) return showToast('Please select a source image', 'error');

    const prompt = document.getElementById('i2i-prompt').value.trim();
    if (!prompt) return showToast('Please enter a transformation prompt', 'error');

    const formatVal = document.getElementById('i2i-format').value;
    let width = undefined;
    let height = undefined;
    if (formatVal !== 'auto') {
      [width, height] = formatVal.split('x').map(Number);
    }
    const isLocal = document.getElementById('i2i-mode') ? document.getElementById('i2i-mode').value === 'local' : false;
    const localModel = document.getElementById('i2i-local-model') ? document.getElementById('i2i-local-model').value : undefined;
    const isMaskEnabled = document.getElementById('i2i-enable-mask')?.checked;
    const maskData = isMaskEnabled ? (document.getElementById('i2i-mask-data')?.value.trim() || undefined) : undefined;

    // LoRA options
    const loraVal = document.getElementById('i2i-lora-select')?.value || undefined;
    const loraScale = loraVal ? Number(document.getElementById('i2i-lora-scale')?.value || 0.8) : undefined;

    // ControlNet options
    const cnEnabled = document.getElementById('i2i-controlnet-enable')?.checked;
    const cnType = cnEnabled ? document.getElementById('i2i-controlnet-type')?.value : undefined;
    const cnImage = cnEnabled ? (document.getElementById('i2i-controlnet-path')?.value.trim() || undefined) : undefined;
    const cnScale = cnEnabled ? Number(document.getElementById('i2i-controlnet-scale')?.value || 0.8) : undefined;

    const payload = {
      image,
      mask: maskData,
      prompt,
      strength: Number(document.getElementById('i2i-strength').value),
      width,
      height,
      steps: Number(document.getElementById('i2i-steps').value),
      output: document.getElementById('i2i-output').value.trim() || 'output_img2img.png',
      local: isLocal,
      local_model: isLocal ? localModel : undefined,
      lora: loraVal,
      lora_scale: loraScale,
      controlnet_image: cnImage,
      controlnet_type: cnType,
      controlnet_scale: cnScale,
    };

    submitJob('img2img', payload, e.target);
  });

  // Face Swap
  document.getElementById('form-faceswap').addEventListener('submit', async e => {
    e.preventDefault();
    const source = document.getElementById('fs-source-path').value.trim();
    const target = document.getElementById('fs-target-path').value.trim();
    if (!source || !target) return showToast('Please select source face AND target media', 'error');

    const payload = {
      source,
      target,
      restore_face: document.getElementById('fs-restore-face').checked,
      face_index: Number(document.getElementById('fs-face-index').value),
      output: document.getElementById('fs-output').value.trim() || 'output_faceswap.mp4',
    };

    submitJob('faceswap', payload, e.target);
  });

  // Video-to-Video
  document.getElementById('form-vid2vid').addEventListener('submit', async e => {
    e.preventDefault();
    const video = document.getElementById('v2v-video-path').value.trim();
    if (!video) return showToast('Please select a source video', 'error');

    const videoModel = document.getElementById('v2v-video-model') ? document.getElementById('v2v-video-model').value : 'wan-2-5';
    const payload = {
      video,
      model: videoModel,
      prompt: document.getElementById('v2v-prompt').value.trim(),
      strength: Number(document.getElementById('v2v-strength').value),
      resolution: document.getElementById('v2v-resolution').value,
      output: document.getElementById('v2v-output').value.trim() || 'output_vid2vid.mp4',
    };

    submitJob('vid2vid', payload, e.target);
  });

  // Text-to-Image
  document.getElementById('form-txt2img').addEventListener('submit', async e => {
    e.preventDefault();
    const prompt = document.getElementById('t2i-prompt').value.trim();
    if (!prompt) return showToast('Please enter a prompt', 'error');

    const [width, height] = document.getElementById('t2i-format').value.split('x').map(Number);
    const isLocal = document.getElementById('t2i-mode') ? document.getElementById('t2i-mode').value === 'local' : false;
    const localModel = document.getElementById('t2i-local-model') ? document.getElementById('t2i-local-model').value : undefined;

    // LoRA options
    const loraVal = document.getElementById('t2i-lora-select')?.value || undefined;
    const loraScale = loraVal ? Number(document.getElementById('t2i-lora-scale')?.value || 0.8) : undefined;

    // ControlNet options
    const cnEnabled = document.getElementById('t2i-controlnet-enable')?.checked;
    const cnType = cnEnabled ? document.getElementById('t2i-controlnet-type')?.value : undefined;
    const cnImage = cnEnabled ? (document.getElementById('t2i-controlnet-path')?.value.trim() || undefined) : undefined;
    const cnScale = cnEnabled ? Number(document.getElementById('t2i-controlnet-scale')?.value || 0.8) : undefined;

    const payload = {
      prompt,
      width,
      height,
      steps: Number(document.getElementById('t2i-steps').value),
      output: document.getElementById('t2i-output').value.trim() || (isLocal ? 'local_image.png' : 'flux_image.png'),
      local: isLocal,
      local_model: isLocal ? localModel : undefined,
      lora: loraVal,
      lora_scale: loraScale,
      controlnet_image: cnImage,
      controlnet_type: cnType,
      controlnet_scale: cnScale,
    };

    submitJob('txt2img', payload, e.target);
  });

  // Prompt Enhancer
  document.getElementById('form-enhance').addEventListener('submit', async e => {
    e.preventDefault();
    const prompt = document.getElementById('enh-prompt').value.trim();
    if (!prompt) return showToast('Please enter a prompt', 'error');

    const submitBtn = e.target.querySelector('button[type="submit"]');
    const origHtml = submitBtn ? submitBtn.innerHTML : '';
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.innerHTML = `<span class="btn-spinner"></span> <span>Optimizing prompt...</span>`;
    }

    const resultBox = document.getElementById('enh-result');
    resultBox.innerText = 'Optimizing prompt...';

    try {
      const res = await fetch('/api/generate/enhance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      if (data.enhanced_prompt) {
        resultBox.innerText = data.enhanced_prompt;
        showToast('Prompt optimized successfully!', 'success');
        loadPrompts();
      } else {
        resultBox.innerText = 'Error optimizing prompt.';
      }
    } catch (err) {
      resultBox.innerText = `Erreur : ${err.message}`;
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = origHtml;
      }
    }
  });

  // Text-to-Speech
  document.getElementById('form-tts')?.addEventListener('submit', handleTtsSubmit);

  // LoRA Training
  document.getElementById('form-lora-train')?.addEventListener('submit', handleLoraTrainSubmit);
}

// ----------------------------------------------------
// Text-to-Speech (TTS) Submission & Controls
// ----------------------------------------------------
async function handleTtsSubmit(e) {
  if (e) {
    e.preventDefault();
    e.stopPropagation();
  }
  const text = document.getElementById('tts-text')?.value.trim();
  if (!text) return showToast('Please enter speech text to synthesize', 'error');

  const engine = document.getElementById('tts-engine')?.value || 'kokoro';
  const language = document.getElementById('tts-language')?.value || 'fr';
  const voice = document.getElementById('tts-voice')?.value || 'af_bella';
  const speed = parseFloat(document.getElementById('tts-speed')?.value) || 1.0;
  const speakerWav = document.getElementById('tts-speaker-wav')?.value.trim() || undefined;

  const payload = {
    text,
    engine,
    language,
    voice,
    speed,
    speaker_wav: speakerWav,
  };

  const formOrBtn = document.getElementById('form-tts') || (e && (e.currentTarget || e.target));
  submitJob('tts', payload, formOrBtn);
  return false;
}
window.handleTtsSubmit = handleTtsSubmit;

// ----------------------------------------------------
// Text-to-Speech (TTS) Voice & Engine Controls
// ----------------------------------------------------
const TTS_VOICES = {
  kokoro: {
    en: [
      { id: 'af_bella', name: 'af_bella (US English - Warm)' },
      { id: 'af_sarah', name: 'af_sarah (US English - Calm)' },
      { id: 'af_heart', name: 'af_heart (US English - Expressive)' },
      { id: 'am_adam', name: 'am_adam (US English - Deep Male)' },
      { id: 'am_michael', name: 'am_michael (US English - Dynamic Male)' },
      { id: 'bf_emma', name: 'bf_emma (UK English - Distinguished)' },
      { id: 'bm_george', name: 'bm_george (UK English - Male)' },
    ],
    fr: [
      { id: 'ff_siwis', name: 'ff_siwis (French - Natural & Soft)' },
    ],
    es: [
      { id: 'ef_dora', name: 'ef_dora (Spanish - Female)' },
      { id: 'em_alex', name: 'em_alex (Spanish - Male)' },
    ],
    it: [
      { id: 'if_sara', name: 'if_sara (Italian - Female)' },
      { id: 'im_nicola', name: 'im_nicola (Italian - Male)' },
    ],
    pt: [
      { id: 'pf_dora', name: 'pf_dora (Portuguese - Female)' },
    ],
    ja: [
      { id: 'jf_alpha', name: 'jf_alpha (Japanese - Female)' },
    ],
    zh: [
      { id: 'zf_xiaobei', name: 'zf_xiaobei (Chinese - Female)' },
    ],
    de: [
      { id: 'af_bella', name: 'af_bella (German - Polyglot)' },
    ],
  },
  xtts: {
    all: [
      { id: 'Claribel Dervla', name: 'Claribel Dervla (Female)' },
      { id: 'Daisy Studious', name: 'Daisy Studious (Calm Female)' },
      { id: 'Gracie Wise', name: 'Gracie Wise (Expressive Female)' },
      { id: 'Tammie Ema', name: 'Tammie Ema (Clear Female)' },
      { id: 'Alison Dietlinde', name: 'Alison Dietlinde (Soft Female)' },
      { id: 'Ana Florence', name: 'Ana Florence (Warm Female)' },
      { id: 'Damien Black', name: 'Damien Black (Deep Male)' },
      { id: 'Baldur Sanjin', name: 'Baldur Sanjin (Steady Male)' },
      { id: 'Craig Gutsy', name: 'Craig Gutsy (Dynamic Male)' },
    ]
  }
};

function handleTtsEngineChange() {
  const engine = document.getElementById('tts-engine').value;
  const xttsGroup = document.getElementById('tts-xtts-ref-group');
  if (xttsGroup) {
    xttsGroup.style.display = (engine === 'xtts') ? 'block' : 'none';
  }
  updateTtsVoiceDropdown();
}

function handleTtsLanguageChange() {
  updateTtsVoiceDropdown();
}

function updateTtsVoiceDropdown() {
  const engine = document.getElementById('tts-engine')?.value || 'kokoro';
  const lang = document.getElementById('tts-language')?.value || 'en';
  const voiceSelect = document.getElementById('tts-voice');
  if (!voiceSelect) return;

  voiceSelect.innerHTML = '';
  let list = [];
  if (engine === 'kokoro') {
    list = (TTS_VOICES.kokoro[lang] || TTS_VOICES.kokoro.en || []);
  } else {
    list = TTS_VOICES.xtts.all;
  }

  list.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v.id;
    opt.innerText = v.name;
    voiceSelect.appendChild(opt);
  });
}

function useAsTtsReference(filename) {
  switchTab('studio');
  switchPipeline('tts');
  const engineSelect = document.getElementById('tts-engine');
  if (engineSelect) {
    engineSelect.value = 'xtts';
    handleTtsEngineChange();
  }
  const inputEl = document.getElementById('tts-speaker-wav');
  if (inputEl) {
    inputEl.value = filename;
  }
  showToast(`XTTS voice sample selected: ${filename}`, 'success');
}

function reuseMedia(filename, isVideo, isAudio = false) {
  if (isAudio) {
    useAsTtsReference(filename);
  } else if (isVideo) {
    useAsVideoToVideo(filename);
  } else {
    useAsImageToVideo(filename);
  }
}

// Inline prompt enhancer helper
async function enhanceCurrentPrompt(textareaId, btnEvent) {
  const el = document.getElementById(textareaId);
  const current = el.value.trim();
  if (!current) return showToast('Enter a description first', 'error');

  const btn = (btnEvent && btnEvent.currentTarget) || (event && event.currentTarget);
  let origHtml = '';
  if (btn) {
    origHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `⏳ Traitement...`;
  }

  showToast('Optimizing prompt...', 'info');
  try {
    const res = await fetch('/api/generate/enhance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: current }),
    });
    const data = await res.json();
    if (data.enhanced_prompt) {
      el.value = data.enhanced_prompt;
      showToast('Enhanced prompt applied!', 'success');
      loadPrompts();
    }
  } catch (err) {
    showToast(`Erreur : ${err.message}`, 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = origHtml;
    }
  }
}

// Submit job to backend queue with button state feedback
async function submitJob(type, payload, formOrBtn) {
  let btn = null;
  let originalHtml = '';
  if (formOrBtn) {
    btn = formOrBtn.tagName === 'BUTTON' ? formOrBtn : formOrBtn.querySelector('button[type="submit"]');
    if (btn) {
      originalHtml = btn.innerHTML;
      btn.disabled = true;
      btn.classList.add('btn-processing');
      btn.innerHTML = `<span class="btn-spinner"></span> <span>Commande en traitement...</span>`;
    }
  }

  showToast(`Launching ${type.toUpperCase()} job...`, 'info');
  appendTerminalLog(`🚀 Launching ${type.toUpperCase()} job`, 'info');

  updateLiveBadge('running', 'En cours');

  try {
    const res = await fetch(`/api/generate/${type}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Erreur de soumission');

    showToast(`Job ${data.id} started successfully!`, 'success');
    appendTerminalLog(`✓ Job initialized [ID: ${data.id}]`, 'info');
    pollJobs();
    loadPrompts();
  } catch (err) {
    showToast(`Erreur : ${err.message}`, 'error');
    appendTerminalLog(`❌ Failed: ${err.message}`, 'error');
    updateLiveBadge('error', 'Failed');
  } finally {
    if (btn) {
      setTimeout(() => {
        btn.disabled = false;
        btn.classList.remove('btn-processing');
        btn.innerHTML = originalHtml;
      }, 1500);
    }
  }
}

let lastCompletedJobCount = 0;

async function pollJobs() {
  try {
    const res = await fetch('/api/jobs');
    if (!res.ok) return;
    allJobs = await res.json();

    const activeJobs = allJobs.filter(j => j.status === 'RUNNING' || j.status === 'IN_PROGRESS' || j.status === 'IN_QUEUE');
    const badge = document.getElementById('active-jobs-count');
    if (activeJobs.length > 0) {
      badge.style.display = 'inline-block';
      badge.innerText = activeJobs.length;
    } else {
      badge.style.display = 'none';
    }

    const completedCount = allJobs.filter(j => j.status === 'COMPLETED').length;
    if (completedCount > lastCompletedJobCount) {
      lastCompletedJobCount = completedCount;
      loadMedia();
      loadLoras();
      fetchBalance(false);
    }

    renderJobsList();
    updateLivePanel(allJobs[0]);
  } catch (err) {
    console.error(err);
  }
}

async function loadJobs() {
  pollJobs();
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

let lastRenderedPreviewKey = '';

function updateLivePanel(latestJob) {
  const badge = document.getElementById('live-job-status-badge');
  const previewBox = document.getElementById('live-preview-box');

  // Prioritize finding a running job in queue
  const runningJob = allJobs.find(j => j.status === 'RUNNING' || j.status === 'IN_PROGRESS' || j.status === 'IN_QUEUE');
  const activeOrLatest = runningJob || latestJob;

  if (!activeOrLatest) {
    lastRenderedPreviewKey = '';
    badge.className = 'badge';
    badge.innerText = 'Idle';
    previewBox.innerHTML = `
      <div class="preview-placeholder">
        <div class="pulsing-circle"></div>
        <p>No jobs currently running.</p>
        <small>Launch a generation from the left panel to see real-time progress and results here.</small>
      </div>
    `;
    return;
  }

  // Update logs without interrupting media playback
  if (activeOrLatest.logs && activeOrLatest.logs.length > 0) {
    const container = document.getElementById('live-logs-container');
    if (container) {
      container.innerHTML = activeOrLatest.logs.map(log => `<div class="log-line ${log.level}">${log.message}</div>`).join('');
      container.scrollTop = container.scrollHeight;
    }
  }

  if (activeOrLatest.status === 'RUNNING' || activeOrLatest.status === 'IN_PROGRESS' || activeOrLatest.status === 'IN_QUEUE') {
    lastRenderedPreviewKey = `${activeOrLatest.id}_running`;
    const rStatus = activeOrLatest.runpod_status || 'IN_PROGRESS';
    
    let subStatusText = '⚡ Processing...';
    let badgeLabel = `Running: ${activeOrLatest.pipeline_type.toUpperCase()}`;
    let badgeClass = 'badge running';

    if (rStatus === 'IN_QUEUE') {
      subStatusText = '⏳ In queue...';
      badgeLabel = '⏳ In Queue';
    } else if (rStatus === 'IN_PROGRESS') {
      subStatusText = '⚡ Processing on GPU...';
      badgeLabel = '⚡ Active Render';
    }

    badge.className = badgeClass;
    badge.innerText = badgeLabel;

    const pipelineLabels = {
      txt2vid: '🎬 Text-to-Video',
      img2vid: '🖼️ Image-to-Video Animation',
      img2img: '🎨 Image-to-Image Transformation',
      faceswap: '🎭 Face Swap (ReActor / InsightFace)',
      vid2vid: '🎞️ Video-to-Video Transformation',
      txt2img: '✨ Text-to-Image Generation (Flux)',
      enhance: '🧠 Prompt Optimization (Qwen3)',
      tts: '🎙️ Text-to-Speech Synthesis',
      lora_train: '🎓 LoRA Fine-Tuning Training'
    };

    const label = pipelineLabels[activeOrLatest.pipeline_type] || activeOrLatest.pipeline_type.toUpperCase();
    const promptText = activeOrLatest.prompt || activeOrLatest.source || 'Processing media...';
    const isPodTask = activeOrLatest.pipeline_type === 'tts' || activeOrLatest.pipeline_type === 'lora_train';
    const gpuTitle = (systemInfo && systemInfo.gpu_name) ? systemInfo.gpu_name : 'Pod GPU';
    const runpodJobId = activeOrLatest.runpod_job_id || (isPodTask ? gpuTitle : 'RunPod Cloud');
    const endpointName = activeOrLatest.runpod_endpoint || (isPodTask ? gpuTitle : 'RunPod Serverless');

    if (activeOrLatest.stage_info) {
      subStatusText = activeOrLatest.stage_info;
    }

    const elapsedLabel = 'Elapsed Time';
    const statusLabel = 'Status';
    const engineLabel = 'Engine:';

    previewBox.innerHTML = `
      <div class="live-running-card">
        <div class="spinner-ring"></div>
        <div class="live-running-title">${label}</div>
        <div class="live-running-sub">${subStatusText}</div>
        <div class="live-running-meta-grid">
          <div class="live-meta-pill"><span>${elapsedLabel}</span><strong>⏱️ ${activeOrLatest.elapsed_seconds || 0}s</strong></div>
          <div class="live-meta-pill"><span>${statusLabel}</span><strong class="status-${rStatus.toLowerCase()}">${rStatus}</strong></div>
          <div class="live-meta-pill" style="grid-column: 1 / -1;"><span>Job ID :</span> <code>${runpodJobId}</code></div>
          <div class="live-meta-pill" style="grid-column: 1 / -1;"><span>${engineLabel}</span> <code>${endpointName}</code></div>
        </div>
        <div class="live-running-prompt">"${escapeHtml(promptText)}"</div>
      </div>
    `;
  } else if (activeOrLatest.status === 'COMPLETED') {
    badge.className = 'badge success';
    badge.innerText = 'Completed Successfully';

    const completedKey = `${activeOrLatest.id}_completed_${activeOrLatest.result_file || ''}`;
    // CRITICAL: If media is ALREADY rendered, do not reset DOM every 2.5s
    if (lastRenderedPreviewKey === completedKey) {
      return;
    }
    lastRenderedPreviewKey = completedKey;

    if (activeOrLatest.result_file) {
      const isVideo = activeOrLatest.result_file.endsWith('.mp4') || activeOrLatest.result_file.endsWith('.webm');
      const isAudio = activeOrLatest.result_file.endsWith('.wav') || activeOrLatest.result_file.endsWith('.mp3') || activeOrLatest.result_file.endsWith('.ogg');
      const isLora = activeOrLatest.pipeline_type === 'lora_train' || activeOrLatest.result_file.endsWith('.safetensors');

      if (isLora) {
        const loraSuccessText = currentLang === 'fr' 
          ? 'LoRA model trained successfully! Available immediately in generation forms.'
          : 'LoRA model trained successfully! Available immediately in generation forms.';
        const loraBtnText = '🎨 Use in Text-to-Image';
        previewBox.innerHTML = `
          <div class="audio-live-preview-card" style="border-color: rgba(139, 92, 246, 0.4); background: rgba(139, 92, 246, 0.05);">
            <div class="audio-pulse-icon" style="background: rgba(139, 92, 246, 0.2); color: #8b5cf6;">🎓</div>
            <div class="audio-filename-display" style="font-size: 1.1rem; font-weight: 700; color: #fff;">${escapeHtml(activeOrLatest.result_file)}</div>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 8px 0 16px;">${loraSuccessText}</p>
            <div style="display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;">
              <button class="btn-primary btn-sm btn-glow" onclick="useTrainedLora('${escapeHtml(activeOrLatest.result_file)}')">${loraBtnText}</button>
            </div>
          </div>
        `;
      } else if (isVideo) {
        previewBox.innerHTML = `<video src="/api/media/${encodeURIComponent(activeOrLatest.result_file)}" controls autoplay loop></video>`;
      } else if (isAudio) {
        previewBox.innerHTML = `
          <div class="audio-live-preview-card">
            <div class="audio-pulse-icon">🎙️</div>
            <div class="audio-filename-display">${escapeHtml(activeOrLatest.result_file)}</div>
            <audio id="live-audio-player" src="/api/media/${encodeURIComponent(activeOrLatest.result_file)}" controls preload="auto" style="width: 100%; margin: 15px 0;"></audio>
            <div style="display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;">
              <button class="btn-secondary btn-sm" onclick="useAsTtsReference('${escapeHtml(activeOrLatest.result_file)}')">🎙️ Clone this voice (XTTS)</button>
              <button class="btn-primary btn-sm" onclick="downloadMedia('${escapeHtml(activeOrLatest.result_file)}')">⬇️ Download</button>
            </div>
          </div>
        `;
        // Play once when result loads
        setTimeout(() => {
          const a = document.getElementById('live-audio-player');
          if (a) a.play().catch(() => {});
        }, 150);
      } else {
        previewBox.innerHTML = `<img src="/api/media/${encodeURIComponent(activeOrLatest.result_file)}" alt="Result">`;
      }
    }
  } else if (activeOrLatest.status === 'FAILED') {
    lastRenderedPreviewKey = `${activeOrLatest.id}_failed`;
    badge.className = 'badge error';
    badge.innerText = 'Job Failed';
    previewBox.innerHTML = `
      <div class="preview-placeholder">
        <div style="font-size: 2.5rem; margin-bottom: 8px;">❌</div>
        <p style="color: var(--accent-red); font-weight: 600;">Generation failed</p>
        <small>Check the terminal logs below for error details.</small>
      </div>
    `;
  }
}

function renderJobsList() {
  const container = document.getElementById('jobs-list-container');
  if (!container) return;

  if (allJobs.length === 0) {
    container.innerHTML = `<div style="text-align: center; padding: 40px; color: var(--text-dim);">No jobs in history</div>`;
    return;
  }

  container.innerHTML = allJobs.map(job => {
    const statusClass = job.status === 'COMPLETED' ? 'success' : job.status === 'FAILED' ? 'error' : 'running';
    return `
      <div class="job-card">
        <div class="job-type-icon">⚡</div>
        <div class="job-details">
          <div class="job-title-row">
            <span class="job-title">${job.pipeline_type.toUpperCase()}</span>
            <span class="badge ${statusClass}">${job.status}</span>
          </div>
          <div class="job-prompt">${job.prompt || job.source || 'No prompt'}</div>
          <div class="job-meta">ID: ${job.id} • Duration: ${job.elapsed_seconds || 0}s • Output: ${job.result_file || 'N/A'}</div>
        </div>
        ${job.result_file ? `<button class="btn-secondary btn-sm" onclick="openMediaModal('${job.result_file}')">👁️ View</button>` : ''}
      </div>
    `;
  }).join('');
}

function clearCompletedJobs() {
  fetch('/api/jobs', { method: 'DELETE' }).then(() => pollJobs());
}

function updateLiveBadge(type, text) {
  const badge = document.getElementById('live-job-status-badge');
  badge.className = `badge ${type}`;
  badge.innerText = text;
}

function appendTerminalLog(msg, type = 'info') {
  const container = document.getElementById('live-logs-container');
  const line = document.createElement('div');
  line.className = `log-line ${type}`;
  line.innerText = msg;
  container.appendChild(line);
  container.scrollTop = container.scrollHeight;
}

// ----------------------------------------------------
// Settings Management
// ----------------------------------------------------
async function loadSettings() {
  try {
    const res = await fetch('/api/settings');
    if (!res.ok) return;
    const settings = await res.json();
    if (settings.api_key) document.getElementById('setting-api-key').value = settings.api_key;
    if (settings.hf_token && document.getElementById('setting-hf-token')) {
      document.getElementById('setting-hf-token').value = settings.hf_token;
    }
    if (settings.wan_endpoint) document.getElementById('setting-wan-endpoint').value = settings.wan_endpoint;
    if (settings.ltx_endpoint && document.getElementById('setting-ltx-endpoint')) document.getElementById('setting-ltx-endpoint').value = settings.ltx_endpoint;
    if (settings.minimax_endpoint && document.getElementById('setting-minimax-endpoint')) document.getElementById('setting-minimax-endpoint').value = settings.minimax_endpoint;
    if (settings.flux_endpoint) document.getElementById('setting-flux-endpoint').value = settings.flux_endpoint;
    if (settings.faceswap_endpoint) document.getElementById('setting-faceswap-endpoint').value = settings.faceswap_endpoint;
    if (settings.llm_endpoint) document.getElementById('setting-llm-endpoint').value = settings.llm_endpoint;
    if (settings.local_model) {
      const el = document.getElementById('setting-local-model');
      if (el) el.value = settings.local_model;
      const t2iModel = document.getElementById('t2i-local-model');
      if (t2iModel) t2iModel.value = settings.local_model;
    }
  } catch (err) {
    console.error(err);
  }
}

async function saveSettings(event) {
  event.preventDefault();
  const settings = {
    api_key: document.getElementById('setting-api-key').value.trim(),
    hf_token: (document.getElementById('setting-hf-token') ? document.getElementById('setting-hf-token').value.trim() : ''),
    wan_endpoint: document.getElementById('setting-wan-endpoint').value.trim(),
    ltx_endpoint: (document.getElementById('setting-ltx-endpoint') ? document.getElementById('setting-ltx-endpoint').value.trim() : 'ltx-video-2-5'),
    minimax_endpoint: (document.getElementById('setting-minimax-endpoint') ? document.getElementById('setting-minimax-endpoint').value.trim() : 'minimax-h3'),
    flux_endpoint: document.getElementById('setting-flux-endpoint').value.trim(),
    faceswap_endpoint: document.getElementById('setting-faceswap-endpoint').value.trim(),
    llm_endpoint: document.getElementById('setting-llm-endpoint').value.trim(),
    local_model: (document.getElementById('setting-local-model') ? document.getElementById('setting-local-model').value.trim() : 'stabilityai/sdxl-turbo'),
  };

  try {
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
    if (!res.ok) throw new Error('Failed to save settings');
    showToast('Settings saved successfully!', 'success');
    fetchBalance(true);
    checkModelsStatus();
  } catch (err) {
    showToast(`Erreur : ${err.message}`, 'error');
  }
}

// ----------------------------------------------------
// Pod GPU Model Management & Local Weights
// ----------------------------------------------------
let modelsStatusCache = [];

async function checkModelsStatus() {
  try {
    const res = await fetch('/api/models/status');
    if (!res.ok) return;
    const data = await res.json();
    modelsStatusCache = data.models || [];
    renderModelsStatus();
    updateLtxBanners();
  } catch (err) {
    console.error('Failed to check models status:', err);
  }
}

function renderModelsStatus() {
  const ltx = modelsStatusCache.find(m => m.id === 'ltx-video');
  if (!ltx) return;

  const badge = document.getElementById('model-badge-ltx');
  const pathEl = document.getElementById('model-path-ltx');
  const btnDl = document.getElementById('btn-download-ltx');

  if (badge && pathEl) {
    if (ltx.installed) {
      badge.className = 'status-badge installed';
      badge.innerHTML = `🟢 Installed (${ltx.size_gb.toFixed(1)} GB)`;
      pathEl.textContent = `📁 ${ltx.path}`;
      if (btnDl) {
        btnDl.innerHTML = `✅ Installed`;
        btnDl.className = 'btn-secondary';
        btnDl.disabled = true;
      }
    } else {
      badge.className = 'status-badge missing';
      badge.innerHTML = `🟡 Weights Missing`;
      pathEl.textContent = `📁 ${ltx.path} (Not found)`;
      if (btnDl) {
        btnDl.innerHTML = `⬇️ Download (~11 GB)`;
        btnDl.className = 'btn-primary';
        btnDl.disabled = false;
      }
    }
  }
}

function updateLtxBanners() {
  const ltx = modelsStatusCache.find(m => m.id === 'ltx-video');
  const t2vModel = document.getElementById('t2v-video-model')?.value;
  const i2vModel = document.getElementById('i2v-video-model')?.value;

  const renderBanner = (bannerEl, isLtxSelected) => {
    if (!bannerEl) return;
    if (!isLtxSelected) {
      bannerEl.style.display = 'none';
      return;
    }
    bannerEl.style.display = 'flex';
    if (ltx && ltx.installed) {
      bannerEl.className = 'model-status-card ready';
      bannerEl.innerHTML = `
        <div>
          <strong>🟢 LTX-Video 2.5 :</strong>
          <span>Ready on Pod GPU (${ltx.size_gb.toFixed(1)} GB local).</span>
        </div>
      `;
    } else {
      bannerEl.className = 'model-status-card missing';
      bannerEl.innerHTML = `
        <div style="flex:1;">
          <strong>🟡 LTX-Video 2.5 :</strong>
          <span>Model weights missing (~11 GB). Requires Hugging Face Token.</span>
        </div>
        <button type="button" class="btn-primary" onclick="downloadModel('ltx-video')" style="padding: 5px 12px; font-size: 0.78rem; white-space: nowrap;">
          ⬇️ Download (~11 GB)
        </button>
      `;
    }
  };

  renderBanner(document.getElementById('t2v-ltx-banner'), t2vModel === 'ltx-2-5');
  renderBanner(document.getElementById('i2v-ltx-banner'), i2vModel === 'ltx-2-5');
}

async function downloadModel(modelId = 'ltx-video') {
  const hfToken = document.getElementById('setting-hf-token')?.value.trim();
  if (!hfToken) {
    showToast('⚠️ Please configure your Hugging Face Token in Settings first.', 'warning');
    switchTab('settings');
    document.getElementById('setting-hf-token')?.focus();
    return;
  }

  showToast('Starting download of model weights in background...', 'info');

  try {
    const res = await fetch('/api/models/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId })
    });
    const data = await res.json();
    if (data.job_id) {
      showToast(`Download job started (${data.job_id}). Tracking in Jobs tab.`, 'success');
      switchTab('jobs');
      fetchJobs();
    }
  } catch (err) {
    showToast(`Erreur : ${err.message}`, 'error');
  }
}


// ----------------------------------------------------
// Account & Balance Management
// ----------------------------------------------------
async function fetchBalance(showToastOnManual = false) {
  const refreshBtns = document.querySelectorAll('.refresh-icon');
  refreshBtns.forEach(icon => icon.classList.add('spinning'));

  try {
    const res = await fetch('/api/balance');
    const data = await res.json();

    const sidebarBalance = document.getElementById('sidebar-balance-val');
    const sidebarEmail = document.getElementById('sidebar-account-email');
    const sidebarDot = document.getElementById('sidebar-status-dot');

    const settingsBalance = document.getElementById('settings-balance-val');
    const settingsEmail = document.getElementById('settings-account-email');
    const settingsId = document.getElementById('settings-account-id');
    const settingsStatus = document.getElementById('settings-connection-status');
    const settingsTime = document.getElementById('settings-balance-time');

    const rawBalance = (data.client_balance !== undefined && data.client_balance !== null)
      ? data.client_balance
      : data.clientBalance;

    if (data.success && rawBalance !== null && rawBalance !== undefined) {
      const formattedBalance = Number(rawBalance).toFixed(2);

      if (sidebarBalance) sidebarBalance.innerText = `${formattedBalance} $`;
      if (sidebarEmail) sidebarEmail.innerText = data.email || 'RunPod Connected';
      if (sidebarDot) {
        sidebarDot.className = 'status-dot online';
      }

      if (settingsBalance) settingsBalance.innerText = `${formattedBalance}`;
      if (settingsEmail) settingsEmail.innerText = data.email || 'RunPod Account Active';
      if (settingsId) settingsId.innerText = data.id || 'Connected';
      if (settingsStatus) {
        settingsStatus.innerHTML = '<span class="status-dot online"></span> Valid API Key';
      }
      if (settingsTime) {
        const now = new Date();
        settingsTime.innerText = `Updated at ${now.toLocaleTimeString()}`;
      }

      if (showToastOnManual) {
        showToast(`Balance updated: ${formattedBalance} $ USD`, 'success');
      }
    } else {
      const errMsg = data.error || 'Unable to retrieve balance';
      if (sidebarBalance) sidebarBalance.innerText = 'Err $';
      if (sidebarEmail) sidebarEmail.innerText = 'Invalid / unconfigured API key';
      if (sidebarDot) {
        sidebarDot.className = 'status-dot offline';
      }

      if (settingsBalance) settingsBalance.innerText = '--';
      if (settingsEmail) settingsEmail.innerText = errMsg;
      if (settingsStatus) {
        settingsStatus.innerHTML = '<span class="status-dot offline"></span> API Key Error';
      }

      if (showToastOnManual) {
        showToast(`Balance error: ${errMsg}`, 'error');
      }
    }
  } catch (err) {
    console.error('Erreur fetchBalance:', err);
    if (showToastOnManual) {
      showToast(`Network error: ${err.message}`, 'error');
    }
  } finally {
    setTimeout(() => {
      refreshBtns.forEach(icon => icon.classList.remove('spinning'));
    }, 400);
  }
}

// ----------------------------------------------------
// Toast Helper
// ----------------------------------------------------
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerText = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(50px)';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

function formatBytes(bytes, decimals = 1) {
  if (!+bytes) return '0 B';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

// ----------------------------------------------------
// Prompts History & Library Management
// ----------------------------------------------------
async function loadPrompts() {
  try {
    const res = await fetch('/api/prompts');
    if (!res.ok) return;
    allPrompts = await res.json();

    const countEl = document.getElementById('prompts-count');
    const countAllEl = document.getElementById('count-all-prompts');
    if (countEl) countEl.innerText = allPrompts.length;
    if (countAllEl) countAllEl.innerText = allPrompts.length;

    renderPromptsList();
  } catch (err) {
    console.error(err);
  }
}

function setPromptFilter(filter) {
  activePromptFilter = filter;
  document.querySelectorAll('[data-prompt-filter]').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-prompt-filter') === filter);
  });
  renderPromptsList();
}

function filterPrompts() {
  renderPromptsList();
}

function renderPromptsList() {
  const container = document.getElementById('prompts-grid-container');
  if (!container) return;

  const searchQuery = (document.getElementById('prompts-search-input')?.value || '').toLowerCase().trim();

  const filtered = allPrompts.filter(p => {
    const matchesSearch = !searchQuery || p.text.toLowerCase().includes(searchQuery);
    if (!matchesSearch) return false;

    if (activePromptFilter === 'favorites') return p.is_favorite;
    if (activePromptFilter === 'all') return true;
    return p.pipeline_type === activePromptFilter;
  });

  if (filtered.length === 0) {
    container.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; padding: 60px; color: var(--text-dim);">No prompts found</div>`;
    return;
  }

  container.innerHTML = filtered.map(p => {
    const dateStr = p.created_at ? new Date(p.created_at * 1000).toLocaleString('en-US', { dateStyle: 'short', timeStyle: 'short' }) : '';
    const starClass = p.is_favorite ? 'active' : '';

    return `
      <div class="prompt-card">
        <div class="prompt-card-header">
          <span class="prompt-tag">${p.pipeline_type}</span>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span class="prompt-date">${dateStr}</span>
            <button class="btn-star-favorite ${starClass}" title="Favorite" onclick="toggleFavoritePrompt('${p.id}')">⭐</button>
          </div>
        </div>
        <div class="prompt-card-body">${escapeHtml(p.text)}</div>
        <div class="prompt-card-footer">
          <div class="prompt-card-actions">
            <button class="btn-primary btn-sm" title="Use in Studio" onclick="usePromptInStudio('${escapeJsStr(p.text)}', '${p.pipeline_type}')">⚡ Use</button>
            <button class="btn-secondary btn-sm" title="Copy" onclick="copyPromptText('${escapeJsStr(p.text)}')">📋 Copy</button>
          </div>
          <button class="btn-danger btn-sm" title="Delete" onclick="deletePrompt('${p.id}')">🗑️</button>
        </div>
      </div>
    `;
  }).join('');
}

function escapeJsStr(str) {
  if (!str) return '';
  return str.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"').replace(/\n/g, '\\n');
}

async function toggleFavoritePrompt(id) {
  try {
    await fetch(`/api/prompts/${id}/favorite`, { method: 'POST' });
    await loadPrompts();
  } catch (err) {
    showToast(`Erreur : ${err.message}`, 'error');
  }
}

async function deletePrompt(id) {
  if (!confirm('Delete this prompt from history?')) return;
  try {
    await fetch(`/api/prompts/${id}`, { method: 'DELETE' });
    showToast('Prompt deleted', 'info');
    await loadPrompts();
  } catch (err) {
    showToast(`Erreur : ${err.message}`, 'error');
  }
}

function copyPromptText(text) {
  navigator.clipboard.writeText(text).then(() => {
    showToast('Prompt copied to clipboard!', 'success');
  }).catch(() => {
    showToast('Failed to copy', 'error');
  });
}

function usePromptInStudio(text, category) {
  switchTab('studio');
  let targetPipeline = 'txt2vid';
  let targetInput = 't2v-prompt';

  if (category === 'img2vid') {
    targetPipeline = 'img2vid';
    targetInput = 'i2v-prompt';
  } else if (category === 'img2img') {
    targetPipeline = 'img2img';
    targetInput = 'i2i-prompt';
  } else if (category === 'vid2vid') {
    targetPipeline = 'vid2vid';
    targetInput = 'v2v-prompt';
  } else if (category === 'txt2img') {
    targetPipeline = 'txt2img';
    targetInput = 't2i-prompt';
  } else if (category === 'enhance') {
    targetPipeline = 'enhance';
    targetInput = 'enh-prompt';
  } else if (category === 'tts') {
    targetPipeline = 'tts';
    targetInput = 'tts-text';
  }

  switchPipeline(targetPipeline);
  const el = document.getElementById(targetInput);
  if (el) {
    el.value = text;
    el.focus();
  }
  showToast(`Prompt inserted into ${targetPipeline.toUpperCase()} module!`, 'success');
}

// Prompt Quick Selector Modal for Textareas
function openPromptSelectorModal(targetInputId) {
  targetPromptInputId = targetInputId;
  const modal = document.getElementById('prompt-selector-modal');
  const searchInput = document.getElementById('prompt-modal-search');
  if (searchInput) searchInput.value = '';
  renderSelectorPrompts();
  modal.classList.add('active');
  if (searchInput) searchInput.focus();
}

function closePromptSelectorModal() {
  const modal = document.getElementById('prompt-selector-modal');
  modal.classList.remove('active');
  targetPromptInputId = null;
}

function filterSelectorPrompts() {
  renderSelectorPrompts();
}

function renderSelectorPrompts() {
  const container = document.getElementById('prompt-selector-list-container');
  if (!container) return;

  const searchQuery = (document.getElementById('prompt-modal-search')?.value || '').toLowerCase().trim();
  const filtered = allPrompts.filter(p => !searchQuery || p.text.toLowerCase().includes(searchQuery));

  if (filtered.length === 0) {
    container.innerHTML = `<div style="text-align:center; padding:30px; color:var(--text-dim);">No prompt dans l'historique</div>`;
    return;
  }

  container.innerHTML = filtered.map(p => {
    return `
      <div class="prompt-selector-item" onclick="selectPromptForInput('${escapeJsStr(p.text)}')">
        <div class="prompt-selector-item-text">
          <span class="prompt-tag" style="margin-right: 8px;">${p.pipeline_type}</span>
          ${escapeHtml(p.text)}
        </div>
        <button class="btn-primary btn-sm">Insert</button>
      </div>
    `;
  }).join('');
}

function selectPromptForInput(text) {
  if (targetPromptInputId) {
    const el = document.getElementById(targetPromptInputId);
    if (el) {
      el.value = text;
      el.focus();
    }
  }
  closePromptSelectorModal();
  showToast('Prompt inserted successfully!', 'success');
}

// Add New Prompt Modal
function openNewPromptModal() {
  document.getElementById('new-prompt-modal').classList.add('active');
  document.getElementById('new-prompt-text').value = '';
  document.getElementById('new-prompt-text').focus();
}

function closeNewPromptModal() {
  document.getElementById('new-prompt-modal').classList.remove('active');
}

async function submitNewPrompt(e) {
  e.preventDefault();
  const text = document.getElementById('new-prompt-text').value.trim();
  const category = document.getElementById('new-prompt-category').value;
  if (!text) return;

  try {
    const res = await fetch('/api/prompts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, pipeline_type: category }),
    });
    if (!res.ok) throw new Error('Error saving prompt');
    closeNewPromptModal();
    showToast('Prompt added to library!', 'success');
    await loadPrompts();
  } catch (err) {
    showToast(`Erreur : ${err.message}`, 'error');
  }
}

// ----------------------------------------------------
// Inpainting Mask Painter & State
// ----------------------------------------------------
let maskState = {
  isDrawing: false,
  tool: 'brush', // 'brush' or 'eraser'
  brushSize: 35,
  sourceImg: null,
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  history: [],
};

function toggleMaskSection(enabled) {
  const controls = document.getElementById('i2i-mask-controls');
  if (controls) controls.style.display = enabled ? 'block' : 'none';
  if (enabled) {
    const imgPath = document.getElementById('i2i-image-path')?.value.trim();
    if (!imgPath) {
      showToast('Select a source image first before drawing a mask', 'info');
    }
  }
}

function updateBrushSize(val) {
  maskState.brushSize = parseInt(val, 10) || 35;
  const label = document.getElementById('mask-brush-size-val');
  if (label) label.innerText = `${maskState.brushSize}px`;
  updateBrushCursorSize();
}

function updateBrushCursorSize() {
  const cursor = document.getElementById('mask-brush-cursor');
  if (cursor) {
    cursor.style.width = `${maskState.brushSize}px`;
    cursor.style.height = `${maskState.brushSize}px`;
  }
}

function setMaskTool(tool) {
  maskState.tool = tool;
  document.getElementById('btn-mask-brush')?.classList.toggle('active', tool === 'brush');
  document.getElementById('btn-mask-eraser')?.classList.toggle('active', tool === 'eraser');
  const cursor = document.getElementById('mask-brush-cursor');
  if (cursor) {
    cursor.style.borderColor = tool === 'eraser' ? '#38bdf8' : '#ec4899';
    cursor.style.background = tool === 'eraser' ? 'rgba(56, 189, 248, 0.25)' : 'rgba(236, 72, 153, 0.25)';
  }
}

function openMaskModal() {
  const imgPath = document.getElementById('i2i-image-path')?.value.trim();
  if (!imgPath) {
    return showToast('Please choose or upload a source image first', 'error');
  }

  const modal = document.getElementById('mask-modal');
  modal.classList.add('active');
  initMaskCanvases(imgPath);
}

function closeMaskModal(e) {
  if (e && e.target && e.target !== e.currentTarget && !e.target.classList.contains('modal-close')) return;
  const modal = document.getElementById('mask-modal');
  modal.classList.remove('active');
}

function initMaskCanvases(imgPath) {
  const container = document.getElementById('mask-canvas-container');
  const bgCanvas = document.getElementById('mask-bg-canvas');
  const drawCanvas = document.getElementById('mask-draw-canvas');
  const cursor = document.getElementById('mask-brush-cursor');
  if (!container || !bgCanvas || !drawCanvas) return;

  const bgCtx = bgCanvas.getContext('2d');
  const drawCtx = drawCanvas.getContext('2d');

  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = `/api/media/${encodeURIComponent(imgPath)}`;
  img.onload = () => {
    maskState.sourceImg = img;
    const contRect = container.getBoundingClientRect();
    const contW = contRect.width || 700;
    const contH = contRect.height || 480;

    const scale = Math.min((contW - 20) / img.width, (contH - 20) / img.height, 1);
    const dispW = Math.round(img.width * scale);
    const dispH = Math.round(img.height * scale);

    bgCanvas.width = img.width;
    bgCanvas.height = img.height;
    bgCanvas.style.width = `${dispW}px`;
    bgCanvas.style.height = `${dispH}px`;

    drawCanvas.width = img.width;
    drawCanvas.height = img.height;
    drawCanvas.style.width = `${dispW}px`;
    drawCanvas.style.height = `${dispH}px`;

    // Draw background image
    bgCtx.clearRect(0, 0, bgCanvas.width, bgCanvas.height);
    bgCtx.drawImage(img, 0, 0);

    // If we have an existing mask saved, load it
    const existingMaskData = document.getElementById('i2i-mask-data')?.value;
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    if (existingMaskData && existingMaskData.startsWith('data:image/')) {
      const maskImg = new Image();
      maskImg.onload = () => {
        drawCtx.drawImage(maskImg, 0, 0);
      };
      maskImg.src = existingMaskData;
    }

    setupMaskDrawingEvents(drawCanvas, container, cursor, img.width, img.height);
    updateBrushCursorSize();
  };
}

function setupMaskDrawingEvents(canvas, container, cursor, naturalW, naturalH) {
  const ctx = canvas.getContext('2d');
  let lastX = 0;
  let lastY = 0;

  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    const scaleX = naturalW / rect.width;
    const scaleY = naturalH / rect.height;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
      screenX: clientX,
      screenY: clientY,
    };
  }

  function startDraw(e) {
    e.preventDefault();
    maskState.isDrawing = true;
    const pos = getCanvasPos(e);
    lastX = pos.x;
    lastY = pos.y;
    drawPoint(pos.x, pos.y);
  }

  function drawPoint(x, y) {
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = maskState.brushSize;

    if (maskState.tool === 'eraser') {
      ctx.globalCompositeOperation = 'destination-out';
    } else {
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = 'rgba(236, 72, 153, 0.9)'; // Hot pink mask color
      ctx.fillStyle = 'rgba(236, 72, 153, 0.9)';
    }

    ctx.beginPath();
    ctx.arc(x, y, maskState.brushSize / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function drawMove(e) {
    const pos = getCanvasPos(e);

    // Update floating brush cursor
    if (cursor) {
      const contRect = container.getBoundingClientRect();
      cursor.style.display = 'block';
      cursor.style.left = `${pos.screenX - contRect.left}px`;
      cursor.style.top = `${pos.screenY - contRect.top}px`;
    }

    if (!maskState.isDrawing) return;
    e.preventDefault();

    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = maskState.brushSize;

    if (maskState.tool === 'eraser') {
      ctx.globalCompositeOperation = 'destination-out';
    } else {
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = 'rgba(236, 72, 153, 0.9)';
    }

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    ctx.restore();

    lastX = pos.x;
    lastY = pos.y;
  }

  function stopDraw() {
    maskState.isDrawing = false;
  }

  canvas.onmousedown = startDraw;
  window.onmousemove = drawMove;
  window.onmouseup = stopDraw;

  canvas.ontouchstart = startDraw;
  canvas.ontouchmove = drawMove;
  canvas.ontouchend = stopDraw;

  container.onmouseleave = () => {
    if (cursor) cursor.style.display = 'none';
  };
  container.onmouseenter = () => {
    if (cursor) cursor.style.display = 'block';
  };
}

function clearMaskCanvas() {
  const drawCanvas = document.getElementById('mask-draw-canvas');
  if (drawCanvas) {
    const ctx = drawCanvas.getContext('2d');
    ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  }
}

function invertMaskCanvas() {
  const drawCanvas = document.getElementById('mask-draw-canvas');
  if (!drawCanvas) return;
  const ctx = drawCanvas.getContext('2d');
  const imgData = ctx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
  const data = imgData.data;

  for (let i = 0; i < data.length; i += 4) {
    const alpha = data[i + 3];
    if (alpha > 20) {
      data[i + 3] = 0; // Unmask
    } else {
      data[i] = 236;
      data[i + 1] = 72;
      data[i + 2] = 153;
      data[i + 3] = 230; // Mask
    }
  }

  ctx.putImageData(imgData, 0, 0);
}

function saveMaskAndClose() {
  const drawCanvas = document.getElementById('mask-draw-canvas');
  if (!drawCanvas) return;

  // Create standard binary black-and-white mask (White = inpaint/regenerate, Black = preserve)
  const exportCanvas = document.createElement('canvas');
  exportCanvas.width = drawCanvas.width;
  exportCanvas.height = drawCanvas.height;
  const expCtx = exportCanvas.getContext('2d');

  // Fill black
  expCtx.fillStyle = '#000000';
  expCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);

  // Where drawCanvas has pixels, paint white
  const drawCtx = drawCanvas.getContext('2d');
  const imgData = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
  const data = imgData.data;

  let hasMaskedPixels = false;
  const expImgData = expCtx.getImageData(0, 0, exportCanvas.width, exportCanvas.height);
  const expData = expImgData.data;

  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] > 20) {
      hasMaskedPixels = true;
      expData[i] = 255;
      expData[i + 1] = 255;
      expData[i + 2] = 255;
      expData[i + 3] = 255;
    }
  }

  expCtx.putImageData(expImgData, 0, 0);

  if (!hasMaskedPixels) {
    showToast('No area was masked. Mask reset.', 'info');
    clearCurrentMask();
    closeMaskModal();
    return;
  }

  const maskDataUrl = exportCanvas.toDataURL('image/png');
  const inputEl = document.getElementById('i2i-mask-data');
  const previewEl = document.getElementById('i2i-mask-preview');

  if (inputEl) inputEl.value = maskDataUrl;
  if (previewEl) {
    previewEl.innerHTML = `<img src="${maskDataUrl}" alt="Inpainting Mask" style="background:#000;">`;
  }

  const checkbox = document.getElementById('i2i-enable-mask');
  if (checkbox) checkbox.checked = true;
  toggleMaskSection(true);

  closeMaskModal();
  showToast('✓ Mask saved successfully for Inpainting!', 'success');
}

function clearCurrentMask() {
  const inputEl = document.getElementById('i2i-mask-data');
  const previewEl = document.getElementById('i2i-mask-preview');
  if (inputEl) inputEl.value = '';
  if (previewEl) previewEl.innerHTML = `<span class="mask-placeholder-text">No mask drawn</span>`;
  showToast('Mask cleared', 'info');
}

// ----------------------------------------------------
// LoRA & ControlNet Helpers
// ----------------------------------------------------
function toggleCollapsible(bodyId, iconId) {
  const body = document.getElementById(bodyId);
  const icon = document.getElementById(iconId);
  if (body) body.classList.toggle('collapsed');
  if (icon) icon.classList.toggle('collapsed');
}

function toggleControlNetVisibility(prefix) {
  const enable = document.getElementById(`${prefix}-controlnet-enable`);
  const fields = document.getElementById(`${prefix}-controlnet-fields`);
  if (fields && enable) {
    fields.style.display = enable.checked ? 'block' : 'none';
  }
}

function clearControlNetRef(prefix) {
  const input = document.getElementById(`${prefix}-controlnet-path`);
  const preview = document.getElementById(`${prefix}-controlnet-preview`);
  if (input) input.value = '';
  if (preview) preview.innerHTML = '<div class="empty-state-selector"><span>🖼️ Custom guide image (or leave empty)</span></div>';
  showToast('ControlNet reference cleared', 'info');
}

async function loadLoras() {
  try {
    const res = await fetch('/api/loras');
    if (!res.ok) return;
    const loras = await res.json();
    const dropdowns = document.querySelectorAll('.lora-select-dropdown');
    dropdowns.forEach(dd => {
      const currentVal = dd.value;
      let html = '<option value="">None (Pure model)</option>';
      if (Array.isArray(loras) && loras.length > 0) {
        loras.forEach(l => {
          html += `<option value="${l.path}">${l.filename} (${l.size_mb} MB)</option>`;
        });
      }
      dd.innerHTML = html;
      if (currentVal) dd.value = currentVal;
    });
  } catch (err) {
    console.warn('Erreur chargement LoRAs:', err);
  }
}

// Global window bindings
window.toggleCollapsible = toggleCollapsible;
window.toggleControlNetVisibility = toggleControlNetVisibility;
window.clearControlNetRef = clearControlNetRef;
window.loadLoras = loadLoras;

// ----------------------------------------------------
// RunPod Mobile Modal
// ----------------------------------------------------
let currentMobileUrl = '';

async function openMobileModal() {
  const modal = document.getElementById('mobile-modal') || document.getElementById('wifi-modal');
  if (!modal) return;
  modal.classList.add('active');

  const linkEl = document.getElementById('mobile-url-link') || document.getElementById('wifi-url-link');
  const qrImg = document.getElementById('mobile-qr-img') || document.getElementById('wifi-qr-img');
  const openLink = document.getElementById('mobile-open-link');

  let resolvedUrl = '';

  // 1. Check if already connected via RunPod proxy or external domain
  if (window.location.hostname.includes('runpod.net') || (!window.location.hostname.includes('localhost') && !window.location.hostname.includes('127.0.0.1'))) {
    resolvedUrl = window.location.origin;
  }

  // 2. Query backend for RUNPOD_POD_ID proxy URL or LAN IP
  try {
    const res = await fetch('/api/network');
    if (res.ok) {
      const data = await res.json();
      if (data.is_runpod && data.mobile_url) {
        resolvedUrl = data.mobile_url;
      } else if (!resolvedUrl && data.mobile_url) {
        resolvedUrl = data.mobile_url;
      }
    }
  } catch (e) {
    console.warn('Network info error:', e);
  }

  if (!resolvedUrl) {
    resolvedUrl = window.location.origin;
  }

  currentMobileUrl = resolvedUrl;

  if (linkEl) {
    linkEl.href = resolvedUrl;
    linkEl.innerText = resolvedUrl;
  }
  if (openLink) {
    openLink.href = resolvedUrl;
  }
  if (qrImg) {
    qrImg.src = `https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=${encodeURIComponent(resolvedUrl)}`;
  }
}

function closeMobileModal() {
  const modal = document.getElementById('mobile-modal') || document.getElementById('wifi-modal');
  if (modal) modal.classList.remove('active');
}

function copyMobileUrl() {
  if (!currentMobileUrl) return;
  const copyBtnText = document.getElementById('btn-copy-url-text');
  navigator.clipboard.writeText(currentMobileUrl).then(() => {
    if (copyBtnText) copyBtnText.textContent = '✅ Copied!';
    setTimeout(() => {
      if (copyBtnText) copyBtnText.textContent = '📋 Copy URL';
    }, 2000);
  }).catch(() => {
    alert(currentMobileUrl);
  });
}

window.openMobileModal = openMobileModal;
window.closeMobileModal = closeMobileModal;
window.copyMobileUrl = copyMobileUrl;
// Backward compatibility
window.openWifiModal = openMobileModal;
window.closeWifiModal = closeMobileModal;

// ----------------------------------------------------
// LoRA Training Studio Functions
// ----------------------------------------------------
let cachedLoraDatasets = [];

function initLoraTraining() {
  const dropzone = document.getElementById('lora-dropzone');
  if (dropzone) {
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-over');
    });
    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('drag-over');
    });
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
      if (e.dataTransfer && e.dataTransfer.files.length > 0) {
        uploadLoraFilesList(e.dataTransfer.files);
      }
    });
  }
}

async function loadLoraDatasets(targetSelectName = null) {
  try {
    const res = await fetch('/api/lora/datasets');
    if (!res.ok) return;
    cachedLoraDatasets = await res.json();

    const select = document.getElementById('lora-dataset-select');
    if (!select) return;

    let html = '<option value="__new__">➕ New Dataset...</option>';
    cachedLoraDatasets.forEach(ds => {
      html += `<option value="${escapeHtml(ds.name)}">📁 ${escapeHtml(ds.name)} (${ds.image_count} images)</option>`;
    });
    select.innerHTML = html;

    if (targetSelectName && cachedLoraDatasets.some(ds => ds.name === targetSelectName)) {
      select.value = targetSelectName;
    } else if (cachedLoraDatasets.length > 0 && !targetSelectName) {
      select.value = cachedLoraDatasets[0].name;
    }

    handleLoraDatasetChange();
  } catch (err) {
    console.warn('Erreur chargement datasets LoRA:', err);
  }
}

function handleLoraDatasetChange() {
  const select = document.getElementById('lora-dataset-select');
  const newGroup = document.getElementById('lora-new-dataset-group');
  const outInput = document.getElementById('lora-output-name');
  if (!select) return;

  const val = select.value;
  if (val === '__new__') {
    if (newGroup) newGroup.style.display = 'block';
    renderDatasetImages(null);
    if (outInput && !outInput.value) outInput.value = 'mon_modele.safetensors';
  } else {
    if (newGroup) newGroup.style.display = 'none';
    const currentDs = cachedLoraDatasets.find(ds => ds.name === val);
    renderDatasetImages(currentDs);
    if (outInput) {
      const cleanName = val.replace(/[^a-zA-Z0-9_-]/g, '_');
      outInput.value = `${cleanName}.safetensors`;
    }
  }
}

function renderDatasetImages(dataset) {
  const container = document.getElementById('dataset-images-container');
  if (!container) return;

  if (!dataset || !dataset.images || dataset.images.length === 0) {
    container.innerHTML = `
      <div class="empty-dataset-hint">
        <span>🖼️</span>
        <p>No photos in this dataset yet. Drop your photos above to get started.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = dataset.images.map(img => `
    <div class="dataset-image-card">
      <img class="dataset-image-thumb" src="${img.path}" alt="${escapeHtml(img.name)}" loading="lazy">
      <div class="dataset-image-body">
        <span class="dataset-image-name" title="${escapeHtml(img.name)}">${escapeHtml(img.name)}</span>
        <textarea class="dataset-image-caption-input" placeholder="Training caption..." 
          onchange="saveDatasetCaption('${escapeHtml(dataset.name)}', '${escapeHtml(img.name)}', this.value)"
        >${escapeHtml(img.caption || '')}</textarea>
      </div>
    </div>
  `).join('');
}

async function handleLoraFilesUpload(event) {
  const files = event.target.files;
  if (files && files.length > 0) {
    await uploadLoraFilesList(files);
  }
  event.target.value = '';
}

async function uploadLoraFilesList(files) {
  const select = document.getElementById('lora-dataset-select');
  const nameInput = document.getElementById('lora-dataset-name');

  let datasetName = select?.value;
  if (datasetName === '__new__' || !datasetName) {
    datasetName = nameInput?.value.trim();
    if (!datasetName) {
      datasetName = `dataset_${Date.now().toString().slice(-6)}`;
      if (nameInput) nameInput.value = datasetName;
    }
  }

  showToast(`Uploading ${files.length} photos to '${datasetName}'...`, 'info');
  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append(`file_${i}`, files[i]);
  }

  try {
    const res = await fetch(`/api/lora/datasets/${encodeURIComponent(datasetName)}/upload`, {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed');

    showToast(`✓ ${data.uploaded_count} photos added to dataset '${datasetName}'!`, 'success');
    await loadLoraDatasets(datasetName);
  } catch (err) {
    showToast(`Erreur : ${err.message}`, 'error');
  }
}

async function triggerAutoCaption() {
  const select = document.getElementById('lora-dataset-select');
  const triggerInput = document.getElementById('lora-trigger');
  const catSelect = document.getElementById('lora-category');
  const btn = document.getElementById('btn-autocaption');
  const btnText = document.getElementById('btn-autocaption-text');

  const datasetName = select?.value;
  if (!datasetName || datasetName === '__new__') {
    return showToast('Please upload images to your dataset first.', 'warn');
  }

  const trigger = triggerInput?.value.trim() || 'sks person';
  const category = catSelect?.value || 'general';

  if (btn) btn.disabled = true;
  if (btnText) btnText.innerHTML = '🪄 Auto-captioning in progress...';

  try {
    const res = await fetch('/api/lora/autocaption', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_name: datasetName,
        trigger,
        category,
        overwrite: true,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Error during captioning');

    showToast(`✓ ${data.count} photos captioned with trigger word '${trigger}'!`, 'success');
    await loadLoraDatasets(datasetName);
  } catch (err) {
    showToast(`Erreur auto-caption : ${err.message}`, 'error');
  } finally {
    if (btn) btn.disabled = false;
    if (btnText) btnText.innerHTML = '🪄 Auto-Caption with AI';
  }
}

async function saveDatasetCaption(datasetName, imageName, caption) {
  try {
    await fetch('/api/lora/caption', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_name: datasetName,
        image_name: imageName,
        caption: caption.trim(),
      }),
    });
  } catch (err) {
    console.warn('Error saving caption:', err);
  }
}

async function handleLoraTrainSubmit(e) {
  if (e) {
    e.preventDefault();
    e.stopPropagation();
  }

  const select = document.getElementById('lora-dataset-select');
  const datasetName = select?.value;
  if (!datasetName || datasetName === '__new__') {
    return showToast('Please select a dataset containing photos to start training.', 'error');
  }

  const currentDs = cachedLoraDatasets.find(ds => ds.name === datasetName);
  if (!currentDs || currentDs.image_count === 0) {
    return showToast('This dataset contains no images. Please upload photos before training.', 'error');
  }

  const outputName = document.getElementById('lora-output-name')?.value.trim() || `${datasetName}.safetensors`;
  const trigger = document.getElementById('lora-trigger')?.value.trim() || 'sks person';
  const baseModel = document.getElementById('lora-base-model')?.value || 'runwayml/stable-diffusion-v1-5';
  const steps = parseInt(document.getElementById('lora-train-steps')?.value, 10) || 500;
  const rank = parseInt(document.getElementById('lora-rank')?.value, 10) || 8;
  const resolution = baseModel.includes('sdxl') ? 1024 : 512;

  const payload = {
    dataset_name: datasetName,
    output_name: outputName,
    instance_prompt: trigger,
    base_model: baseModel,
    train_steps: steps,
    lora_rank: rank,
    resolution,
    learning_rate: 0.0001,
  };

  const btn = document.getElementById('btn-submit-lora-train');
  const btnText = document.getElementById('lora-train-submit-text');
  if (btn) btn.disabled = true;
  if (btnText) btnText.innerHTML = '⚡ LoRA training in progress...';

  try {
    const res = await fetch('/api/lora/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Error starting training');

    showToast(`🚀 LoRA training started! (Job ${data.id})`, 'success');
    appendTerminalLog(`🚀 Starting LoRA training: ${outputName} (${steps} steps)`, 'info');
    pollJobs();
  } catch (err) {
    showToast(`Erreur : ${err.message}`, 'error');
  } finally {
    setTimeout(() => {
      if (btn) btn.disabled = false;
      if (btnText) btnText.innerHTML = '🚀 Launch LoRA Training';
    }, 2000);
  }
}

function useTrainedLora(safetensorsFile) {
  switchPipeline('txt2img');
  // Select in LoRA dropdown
  const dropdown = document.querySelector('#form-txt2img .lora-select-dropdown');
  if (dropdown) {
    for (let i = 0; i < dropdown.options.length; i++) {
      if (dropdown.options[i].text.includes(safetensorsFile) || dropdown.options[i].value.includes(safetensorsFile)) {
        dropdown.selectedIndex = i;
        break;
      }
    }
  }
  showToast(`LoRA '${safetensorsFile}' selected for Text-to-Image generation!`, 'success');
}

window.handleLoraDatasetChange = handleLoraDatasetChange;
window.handleLoraFilesUpload = handleLoraFilesUpload;
window.triggerAutoCaption = triggerAutoCaption;
window.saveDatasetCaption = saveDatasetCaption;
window.handleLoraTrainSubmit = handleLoraTrainSubmit;
window.useTrainedLora = useTrainedLora;


