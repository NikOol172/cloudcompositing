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

function initApp() {
  initNavigation();
  initPipelineSwitcher();
  initDropzone();
  initForms();
  loadMedia();
  loadJobs();
  loadPrompts();
  loadSettings();
  fetchBalance(false);

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
    if (!res.ok) throw new Error('Erreur lors du chargement des médias');
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
    if (activeFilter === 'images') return !m.is_video;
    return true;
  });

  if (filtered.length === 0) {
    container.innerHTML = `
      <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-dim);">
        <p style="font-size: 1.1rem; margin-bottom: 8px;">Aucun média trouvé</p>
        <small>Importez ou générez vos premières créations depuis le studio.</small>
      </div>
    `;
    return;
  }

  filtered.forEach(item => {
    const card = document.createElement('div');
    card.className = 'media-card';

    const thumbHtml = item.is_video
      ? `<video src="/api/media/${encodeURIComponent(item.name)}" muted preload="metadata"></video>
         <span class="media-badge">🎬 VIDÉO</span>
         <div class="play-overlay-icon">▶</div>`
      : `<img src="/api/media/${encodeURIComponent(item.name)}" loading="lazy" alt="${item.name}">
         <span class="media-badge">🖼️ IMAGE</span>`;

    card.innerHTML = `
      <div class="media-thumbnail-wrapper" onclick="openMediaModal('${item.name}')">
        ${thumbHtml}
      </div>
      <div class="media-info">
        <div class="media-name" title="${item.name}">${item.name}</div>
        <div class="media-meta">
          <span>${formatBytes(item.size_bytes)}</span>
          <span>${item.is_video ? 'MP4' : 'Image'}</span>
        </div>
        <div class="media-actions">
          <button class="btn-secondary btn-sm" onclick="reuseMedia('${item.name}', ${item.is_video})">⚡ Réutiliser</button>
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
    return `<video src="/api/media/${encoded}" autoplay loop muted playsinline onerror="this.outerHTML='<div class=\\'video-codec-box\\'>🎬 <strong>${escaped}</strong><br><small>Fichier vidéo prêt pour RunPod</small></div>'"></video>`;
  } else {
    return `<img src="/api/media/${encodeURIComponent(filename)}" alt="${filename}">`;
  }
}

async function uploadAndSelectFile(file, targetInputId, previewContainerId) {
  showToast(`Téléversement de ${file.name}...`, 'info');
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('Échec du téléversement');
    const data = await res.json();
    const uploadedName = data.filename || file.name;

    const inputEl = document.getElementById(targetInputId);
    if (inputEl) inputEl.value = uploadedName;

    const previewEl = document.getElementById(previewContainerId);
    if (previewEl) {
      const isVideo = file.type.startsWith('video/') || /\.(mp4|webm|mov|mkv)$/i.test(uploadedName);
      previewEl.innerHTML = renderMediaPreviewHtml(uploadedName, isVideo);
    }

    showToast(`✓ ${uploadedName} téléversé et sélectionné`, 'success');
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

    showToast(`Téléversement de ${file.name}...`, 'info');
    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('Échec du téléversement');
      showToast(`${file.name} importé avec succès`, 'success');
    } catch (err) {
      showToast(`Erreur : ${err.message}`, 'error');
    }
  }

  loadMedia();
}

async function deleteMedia(filename) {
  if (!confirm(`Supprimer définitivement '${filename}' ?`)) return;

  try {
    const res = await fetch(`/api/media/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Impossible de supprimer le fichier');
    showToast(`Fichier ${filename} supprimé`, 'success');
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
    if (allowedType === 'image') titleEl.innerText = '🖼️ Choisir une Image Source';
    else if (allowedType === 'video') titleEl.innerText = '🎬 Choisir une Vidéo Source';
    else titleEl.innerText = '📁 Choisir un Média Source';
  }

  const searchInput = document.getElementById('picker-search-input');
  if (searchInput) searchInput.value = '';

  // Rafraîchir systématiquement la liste des fichiers depuis le disque
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
    if (currentPickerAllowedType === 'image') return !m.is_video;
    return true;
  });

  if (searchQuery) {
    filtered = filtered.filter(m => m.name.toLowerCase().includes(searchQuery));
  }

  if (filtered.length === 0) {
    container.innerHTML = `
      <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-dim);">
        <p style="font-size: 1.1rem; margin-bottom: 8px;">Aucun média trouvé</p>
        <small>Les fichiers générés ou importés apparaîtront ici.</small>
      </div>
    `;
    return;
  }

  filtered.forEach(item => {
    const card = document.createElement('div');
    card.className = 'media-card';
    card.style.cursor = 'pointer';
    card.onclick = () => currentPickerCallback(item.name, item.is_video);

    const thumbHtml = item.is_video
      ? `<video src="/api/media/${encodeURIComponent(item.name)}" muted playsinline onerror="this.outerHTML='<div class=\\'video-codec-box\\'>🎬</div>'"></video><span class="media-badge">VIDÉO</span>`
      : `<img src="/api/media/${encodeURIComponent(item.name)}" alt="${item.name}"><span class="media-badge">IMAGE</span>`;

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
  } else {
    body.innerHTML = `<img src="/api/media/${encodeURIComponent(item.name)}" alt="${item.name}" style="max-height: 65vh; object-fit: contain;">`;
  }

  const footer = document.getElementById('modal-footer');
  if (item.is_video) {
    footer.innerHTML = `
      <button class="btn-secondary" onclick="openCropModal('${item.name}', 'vid2vid-source'); closeMediaModal();">✂️ Recadrer</button>
      <button class="btn-secondary" onclick="useAsVideoToVideo('${item.name}'); closeMediaModal();">🎞️ Transformer (Vid2Vid)</button>
      <button class="btn-secondary" onclick="useAsFaceTarget('${item.name}', true); closeMediaModal();">🎭 Cible Face Swap</button>
      <button class="btn-primary" onclick="downloadMedia('${item.name}')">⬇️ Télécharger</button>
    `;
  } else {
    footer.innerHTML = `
      <button class="btn-secondary" onclick="openCropModal('${item.name}', 'faceswap-source'); closeMediaModal();">✂️ Recadrer</button>
      <button class="btn-secondary" onclick="useAsImageToImage('${item.name}'); closeMediaModal();">🎨 Image-to-Image</button>
      <button class="btn-secondary" onclick="useAsFaceSource('${item.name}'); closeMediaModal();">🎭 Visage Face Swap</button>
      <button class="btn-secondary" onclick="useAsFaceTarget('${item.name}', false); closeMediaModal();">🎯 Cible Face Swap</button>
      <button class="btn-secondary" onclick="useAsImageToVideo('${item.name}'); closeMediaModal();">🖼️ Animer (Img2Vid)</button>
      <button class="btn-primary" onclick="downloadMedia('${item.name}')">⬇️ Télécharger</button>
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
  showToast(`Image sélectionnée pour Image-to-Image : ${filename}`, 'success');
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
    showToast('Veuillez d\'abord choisir ou uploader un média', 'info');
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
  if (titleEl) titleEl.innerText = `✂️ Recadrer : ${filename}`;

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

    // Calcul taille initiale (carré 1:1 au centre)
    const initialSize = Math.min(cropState.mediaWidth, cropState.mediaHeight) * 0.85;
    cropState.boxWidth = initialSize;
    cropState.boxHeight = initialSize;
    cropState.boxLeft = cropState.mediaLeft + (cropState.mediaWidth - initialSize) / 2;
    // Positionner vers le haut où se trouve le visage
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

  // Vérifier limites
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
  btnText.innerText = '⏳ Traitement FFmpeg en cours...';

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

    if (!res.ok) throw new Error('Erreur lors du recadrage');
    const data = await res.json();
    const newFilename = data.filename;

    showToast(`✓ Recadrage réussi : ${newFilename}`, 'success');
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
    btnText.innerText = '✂️ Valider et Recadrer';
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
    if (submitText) submitText.textContent = '💻 Transformer Localement (GPU Diffusers)';
    if (stepsInput && stepsInput.value === '4') stepsInput.value = '4';
    if (outputInput && outputInput.value === 'pickleball_player.png') outputInput.value = 'local_img2img.png';
  } else {
    if (localGroup) localGroup.style.display = 'none';
    if (submitText) submitText.textContent = '🎨 Transformer l\'Image (Image-to-Image)';
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
    if (submitText) submitText.textContent = '💻 Générer Localement (GPU Diffusers)';
    if (stepsInput && stepsInput.value === '4') stepsInput.value = '2';
    if (outputInput && outputInput.value === 'flux_image.png') outputInput.value = 'local_sdxl.png';
  } else {
    if (localGroup) localGroup.style.display = 'none';
    if (submitText) submitText.textContent = '✨ Générer l\'Image (Flux)';
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
    if (!prompt) return showToast('Veuillez saisir un prompt', 'error');

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
    if (!image) return showToast('Veuillez sélectionner une image source', 'error');

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
    if (!image) return showToast('Veuillez sélectionner une image source', 'error');

    const prompt = document.getElementById('i2i-prompt').value.trim();
    if (!prompt) return showToast('Veuillez saisir un prompt de transformation', 'error');

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
    };

    submitJob('img2img', payload, e.target);
  });

  // Face Swap
  document.getElementById('form-faceswap').addEventListener('submit', async e => {
    e.preventDefault();
    const source = document.getElementById('fs-source-path').value.trim();
    const target = document.getElementById('fs-target-path').value.trim();
    if (!source || !target) return showToast('Veuillez sélectionner le visage source ET le média cible', 'error');

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
    if (!video) return showToast('Veuillez sélectionner une vidéo source', 'error');

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
    if (!prompt) return showToast('Veuillez saisir un prompt', 'error');

    const [width, height] = document.getElementById('t2i-format').value.split('x').map(Number);
    const isLocal = document.getElementById('t2i-mode') ? document.getElementById('t2i-mode').value === 'local' : false;
    const localModel = document.getElementById('t2i-local-model') ? document.getElementById('t2i-local-model').value : undefined;
    const payload = {
      prompt,
      width,
      height,
      steps: Number(document.getElementById('t2i-steps').value),
      output: document.getElementById('t2i-output').value.trim() || (isLocal ? 'local_image.png' : 'flux_image.png'),
      local: isLocal,
      local_model: isLocal ? localModel : undefined,
    };

    submitJob('txt2img', payload, e.target);
  });

  // Prompt Enhancer
  document.getElementById('form-enhance').addEventListener('submit', async e => {
    e.preventDefault();
    const prompt = document.getElementById('enh-prompt').value.trim();
    if (!prompt) return showToast('Veuillez saisir un prompt', 'error');

    const submitBtn = e.target.querySelector('button[type="submit"]');
    const origHtml = submitBtn ? submitBtn.innerHTML : '';
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.innerHTML = `<span class="btn-spinner"></span> <span>Optimisation en cours...</span>`;
    }

    const resultBox = document.getElementById('enh-result');
    resultBox.innerText = 'Optimisation en cours...';

    try {
      const res = await fetch('/api/generate/enhance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      if (data.enhanced_prompt) {
        resultBox.innerText = data.enhanced_prompt;
        showToast('Prompt optimisé avec succès !', 'success');
        loadPrompts();
      } else {
        resultBox.innerText = 'Erreur lors de l’enrichissement.';
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
}

// Inline prompt enhancer helper
async function enhanceCurrentPrompt(textareaId, btnEvent) {
  const el = document.getElementById(textareaId);
  const current = el.value.trim();
  if (!current) return showToast('Saisissez une description d\'abord', 'error');

  const btn = (btnEvent && btnEvent.currentTarget) || (event && event.currentTarget);
  let origHtml = '';
  if (btn) {
    origHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `⏳ Traitement...`;
  }

  showToast('Optimisation du prompt en cours...', 'info');
  try {
    const res = await fetch('/api/generate/enhance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: current }),
    });
    const data = await res.json();
    if (data.enhanced_prompt) {
      el.value = data.enhanced_prompt;
      showToast('Prompt enrichi appliqué !', 'success');
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

  showToast(`Lancement du job ${type.toUpperCase()}...`, 'info');
  appendTerminalLog(`🚀 Lancement du job ${type.toUpperCase()}`, 'info');

  updateLiveBadge('running', 'En cours');

  try {
    const res = await fetch(`/api/generate/${type}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Erreur de soumission');

    showToast(`Job ${data.id} démarré avec succès !`, 'success');
    appendTerminalLog(`✓ Job initialisé [ID: ${data.id}]`, 'info');
    pollJobs();
    loadPrompts();
  } catch (err) {
    showToast(`Erreur : ${err.message}`, 'error');
    appendTerminalLog(`❌ Échec : ${err.message}`, 'error');
    updateLiveBadge('error', 'Échec');
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

function updateLivePanel(latestJob) {
  const badge = document.getElementById('live-job-status-badge');
  const previewBox = document.getElementById('live-preview-box');

  // Trouver en priorité un job en cours d'exécution dans toute la file
  const runningJob = allJobs.find(j => j.status === 'RUNNING' || j.status === 'IN_PROGRESS' || j.status === 'IN_QUEUE');
  const activeOrLatest = runningJob || latestJob;

  if (!activeOrLatest) {
    badge.className = 'badge';
    badge.innerText = 'En veille';
    previewBox.innerHTML = `
      <div class="preview-placeholder">
        <div class="pulsing-circle"></div>
        <p>Aucun job en cours d'exécution.</p>
        <small>Lancez une génération depuis le panneau de gauche pour voir la progression et le résultat ici.</small>
      </div>
    `;
    return;
  }

  if (activeOrLatest.status === 'RUNNING' || activeOrLatest.status === 'IN_PROGRESS' || activeOrLatest.status === 'IN_QUEUE') {
    const rStatus = activeOrLatest.runpod_status || 'IN_QUEUE';
    
    let subStatusText = '⚡ Traitement GPU en cours sur RunPod Serverless...';
    let badgeLabel = `En cours : ${activeOrLatest.pipeline_type.toUpperCase()}`;
    let badgeClass = 'badge running';

    if (rStatus === 'IN_QUEUE') {
      subStatusText = '⏳ En file d\'attente RunPod (Recherche d\'un GPU disponible / Cold Start)...';
      badgeLabel = '⏳ File d\'attente RunPod';
    } else if (rStatus === 'IN_PROGRESS') {
      subStatusText = '⚡ Worker GPU actif : Rendu en cours sur le GPU...';
      badgeLabel = '⚡ Rendu GPU Actif';
    }

    badge.className = badgeClass;
    badge.innerText = badgeLabel;

    const pipelineLabels = {
      txt2vid: '🎬 Text-to-Video',
      img2vid: '🖼️ Animation Image-to-Video',
      img2img: '🎨 Transformation Image-to-Image',
      faceswap: '🎭 Face Swap (ReActor / InsightFace)',
      vid2vid: '🎞️ Transformation Video-to-Video',
      txt2img: '✨ Génération Image (Flux)',
      enhance: '🧠 Optimisation Prompt (Qwen3)'
    };

    const label = pipelineLabels[activeOrLatest.pipeline_type] || activeOrLatest.pipeline_type.toUpperCase();
    const promptText = activeOrLatest.prompt || activeOrLatest.source || 'Traitement du média...';
    const runpodJobId = activeOrLatest.runpod_job_id || 'En cours de soumission...';
    const endpointName = activeOrLatest.runpod_endpoint || '1peyap2qc3tx31';

    previewBox.innerHTML = `
      <div class="live-running-card">
        <div class="spinner-ring"></div>
        <div class="live-running-title">${label}</div>
        <div class="live-running-sub">${subStatusText}</div>
        <div class="live-running-meta-grid">
          <div class="live-meta-pill"><span>Temps écoulé</span><strong>⏱️ ${activeOrLatest.elapsed_seconds || 0}s</strong></div>
          <div class="live-meta-pill"><span>Statut RunPod</span><strong class="status-${rStatus.toLowerCase()}">${rStatus}</strong></div>
          <div class="live-meta-pill" style="grid-column: 1 / -1;"><span>Job ID :</span> <code>${runpodJobId}</code></div>
          <div class="live-meta-pill" style="grid-column: 1 / -1;"><span>Endpoint :</span> <code>${endpointName}</code></div>
        </div>
        <div class="live-running-prompt">"${escapeHtml(promptText)}"</div>
      </div>
    `;
  } else if (activeOrLatest.status === 'COMPLETED') {
    badge.className = 'badge success';
    badge.innerText = 'Terminé avec succès';

    if (activeOrLatest.result_file) {
      const isVideo = activeOrLatest.result_file.endsWith('.mp4') || activeOrLatest.result_file.endsWith('.webm');
      if (isVideo) {
        previewBox.innerHTML = `<video src="/api/media/${encodeURIComponent(activeOrLatest.result_file)}" controls autoplay loop></video>`;
      } else {
        previewBox.innerHTML = `<img src="/api/media/${encodeURIComponent(activeOrLatest.result_file)}" alt="Résultat">`;
      }
    }
  } else if (activeOrLatest.status === 'FAILED') {
    badge.className = 'badge error';
    badge.innerText = 'Échec du job';
    previewBox.innerHTML = `
      <div class="preview-placeholder">
        <div style="font-size: 2.5rem; margin-bottom: 8px;">❌</div>
        <p style="color: var(--accent-red); font-weight: 600;">La génération a échoué</p>
        <small>Consultez le terminal de logs ci-dessous pour voir le détail de l'erreur.</small>
      </div>
    `;
  }

  // Update logs
  if (activeOrLatest.logs && activeOrLatest.logs.length > 0) {
    const container = document.getElementById('live-logs-container');
    container.innerHTML = activeOrLatest.logs.map(log => `<div class="log-line ${log.level}">${log.message}</div>`).join('');
    container.scrollTop = container.scrollHeight;
  }
}

function renderJobsList() {
  const container = document.getElementById('jobs-list-container');
  if (!container) return;

  if (allJobs.length === 0) {
    container.innerHTML = `<div style="text-align: center; padding: 40px; color: var(--text-dim);">Aucun job dans l'historique</div>`;
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
          <div class="job-prompt">${job.prompt || job.source || 'Aucun prompt'}</div>
          <div class="job-meta">ID: ${job.id} • Durée: ${job.elapsed_seconds || 0}s • Sortie: ${job.result_file || 'N/A'}</div>
        </div>
        ${job.result_file ? `<button class="btn-secondary btn-sm" onclick="openMediaModal('${job.result_file}')">👁️ Voir</button>` : ''}
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
    if (!res.ok) throw new Error('Échec de la sauvegarde');
    showToast('Paramètres sauvegardés avec succès !', 'success');
    fetchBalance(true);
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
      if (sidebarEmail) sidebarEmail.innerText = data.email || 'RunPod Connecté';
      if (sidebarDot) {
        sidebarDot.className = 'status-dot online';
      }

      if (settingsBalance) settingsBalance.innerText = `${formattedBalance}`;
      if (settingsEmail) settingsEmail.innerText = data.email || 'Compte RunPod Actif';
      if (settingsId) settingsId.innerText = data.id || 'Connecté';
      if (settingsStatus) {
        settingsStatus.innerHTML = '<span class="status-dot online"></span> Clé API Valide';
      }
      if (settingsTime) {
        const now = new Date();
        settingsTime.innerText = `Actualisé à ${now.toLocaleTimeString()}`;
      }

      if (showToastOnManual) {
        showToast(`Solde actualisé : ${formattedBalance} $ USD`, 'success');
      }
    } else {
      const errMsg = data.error || 'Impossible de récupérer le solde';
      if (sidebarBalance) sidebarBalance.innerText = 'Err $';
      if (sidebarEmail) sidebarEmail.innerText = 'Clé invalide / non configurée';
      if (sidebarDot) {
        sidebarDot.className = 'status-dot offline';
      }

      if (settingsBalance) settingsBalance.innerText = '--';
      if (settingsEmail) settingsEmail.innerText = errMsg;
      if (settingsStatus) {
        settingsStatus.innerHTML = '<span class="status-dot offline"></span> Erreur Clé API';
      }

      if (showToastOnManual) {
        showToast(`Erreur solde : ${errMsg}`, 'error');
      }
    }
  } catch (err) {
    console.error('Erreur fetchBalance:', err);
    if (showToastOnManual) {
      showToast(`Erreur réseau : ${err.message}`, 'error');
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
    container.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; padding: 60px; color: var(--text-dim);">Aucun prompt trouvé</div>`;
    return;
  }

  container.innerHTML = filtered.map(p => {
    const dateStr = p.created_at ? new Date(p.created_at * 1000).toLocaleString('fr-FR', { dateStyle: 'short', timeStyle: 'short' }) : '';
    const starClass = p.is_favorite ? 'active' : '';

    return `
      <div class="prompt-card">
        <div class="prompt-card-header">
          <span class="prompt-tag">${p.pipeline_type}</span>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span class="prompt-date">${dateStr}</span>
            <button class="btn-star-favorite ${starClass}" title="Favori" onclick="toggleFavoritePrompt('${p.id}')">⭐</button>
          </div>
        </div>
        <div class="prompt-card-body">${escapeHtml(p.text)}</div>
        <div class="prompt-card-footer">
          <div class="prompt-card-actions">
            <button class="btn-primary btn-sm" title="Utiliser dans le studio" onclick="usePromptInStudio('${escapeJsStr(p.text)}', '${p.pipeline_type}')">⚡ Utiliser</button>
            <button class="btn-secondary btn-sm" title="Copier" onclick="copyPromptText('${escapeJsStr(p.text)}')">📋 Copier</button>
          </div>
          <button class="btn-danger btn-sm" title="Supprimer" onclick="deletePrompt('${p.id}')">🗑️</button>
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
  if (!confirm('Supprimer ce prompt de votre historique ?')) return;
  try {
    await fetch(`/api/prompts/${id}`, { method: 'DELETE' });
    showToast('Prompt supprimé', 'info');
    await loadPrompts();
  } catch (err) {
    showToast(`Erreur : ${err.message}`, 'error');
  }
}

function copyPromptText(text) {
  navigator.clipboard.writeText(text).then(() => {
    showToast('Prompt copié dans le presse-papier !', 'success');
  }).catch(() => {
    showToast('Impossible de copier', 'error');
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
  }

  switchPipeline(targetPipeline);
  const el = document.getElementById(targetInput);
  if (el) {
    el.value = text;
    el.focus();
  }
  showToast(`Prompt inséré dans le module ${targetPipeline.toUpperCase()} !`, 'success');
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
    container.innerHTML = `<div style="text-align:center; padding:30px; color:var(--text-dim);">Aucun prompt dans l'historique</div>`;
    return;
  }

  container.innerHTML = filtered.map(p => {
    return `
      <div class="prompt-selector-item" onclick="selectPromptForInput('${escapeJsStr(p.text)}')">
        <div class="prompt-selector-item-text">
          <span class="prompt-tag" style="margin-right: 8px;">${p.pipeline_type}</span>
          ${escapeHtml(p.text)}
        </div>
        <button class="btn-primary btn-sm">Insérer</button>
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
  showToast('Prompt inséré avec succès !', 'success');
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
    if (!res.ok) throw new Error('Erreur lors de l’enregistrement');
    closeNewPromptModal();
    showToast('Prompt ajouté à la bibliothèque !', 'success');
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
      showToast('Sélectionnez d\'abord une image source avant de dessiner un masque', 'info');
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
    return showToast('Veuillez d\'abord choisir ou uploader une image source', 'error');
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
    showToast('Aucune zone n\'a été masquée. Masque réinitialisé.', 'info');
    clearCurrentMask();
    closeMaskModal();
    return;
  }

  const maskDataUrl = exportCanvas.toDataURL('image/png');
  const inputEl = document.getElementById('i2i-mask-data');
  const previewEl = document.getElementById('i2i-mask-preview');

  if (inputEl) inputEl.value = maskDataUrl;
  if (previewEl) {
    previewEl.innerHTML = `<img src="${maskDataUrl}" alt="Masque Inpaint" style="background:#000;">`;
  }

  const checkbox = document.getElementById('i2i-enable-mask');
  if (checkbox) checkbox.checked = true;
  toggleMaskSection(true);

  closeMaskModal();
  showToast('✓ Masque enregistré avec succès pour Inpainting !', 'success');
}

function clearCurrentMask() {
  const inputEl = document.getElementById('i2i-mask-data');
  const previewEl = document.getElementById('i2i-mask-preview');
  if (inputEl) inputEl.value = '';
  if (previewEl) previewEl.innerHTML = `<span class="mask-placeholder-text">Aucun tracé</span>`;
  showToast('Masque effacé', 'info');
}
