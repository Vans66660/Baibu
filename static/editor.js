const $ = id => document.getElementById(id);
const canvas = $('view');
const ctx = canvas.getContext('2d');
let file = null;
let original = null;
let clean = null;
let product = null;
let objectVisible = true;
let crop = null;
let selection = null;
let pendingBox = null;
let mode = 'idle';
let drag = null;
let fineReferenceBox = null;
let fineCandidate = null;
const inFineMode = () => mode.startsWith('fine');
let busy = false;
let busyTimer = null;
function setBusy(active, repairing = false) {
  busy = active;
  if (active) $('performanceInfo').hidden = true;
  clearInterval(busyTimer);
  busyTimer = null;
  $('processingOverlay').hidden = !active;
  if (active) {
    $('processingTitle').textContent = repairing ? '正在修补背景' : '正在识别所选物品';
    $('processingDescription').textContent = repairing
      ? 'AI 正在补全涂抹区域，让背景自然衔接'
      : 'AI 正在抠出物品并补全原位置背景';
    const started = Date.now();
    const updateTime = () => {
      const seconds = Math.floor((Date.now() - started) / 1000);
      $('processingTime').textContent = seconds < 60
        ? `已等待 ${seconds} 秒`
        : `已等待 ${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`;
    };
    updateTime();
    busyTimer = setInterval(updateTime, 1000);
  }
  syncButtons();
}
const repairMask = document.createElement('canvas');
const repairCtx = repairMask.getContext('2d');
let painting = false;
let hasRepairPaint = false;

const state = {x: 0, y: 0, sx: 1, sy: 1, depth: 1, side: 'right', angle: 0};
const loadImage = src => new Promise((resolve, reject) => {
  const image = new Image();
  image.onload = () => resolve(image);
  image.onerror = reject;
  image.src = src;
});

function status(message) { $('status').textContent = message; }
function showPerformance(answer) {
  const timings = answer.timings_ms;
  $('performanceInfo').hidden = !timings;
  if (!timings) return '';
  const seconds = ms => (ms / 1000).toFixed(ms < 1000 ? 3 : 2) + ' 秒';
  const reused = answer.cache?.result ? ' · 已复用处理结果'
    : answer.cache?.image_embedding ? ' · 已复用图片特征' : '';
  $('performanceSummary').textContent = `服务端处理 ${seconds(timings.total_ms)}${reused}`;
  const names = {decode_ms: '图片读取', fingerprint_ms: '图片指纹', cache_lookup_ms: '缓存查询',
    model_load_ms: '模型准备', translation_ms: '名称转换', detection_ms: '物品定位',
    sam_preprocess_ms: '抠图预处理', sam_embedding_ms: '图像特征' + (answer.cache?.image_embedding ? '（复用）' : ''),
    gpu_cleanup_ms: 'GPU 临时内存释放', background_model_load_ms: '修补模型准备', sam_mask_ms: '物品分割', sam_postprocess_ms: '蒙版整理', background_ms: '背景修补与图层合成', reference_texture_ms: '参考纹理合成', generation_ms: '精细生成', encode_ms: '结果编码'};
  $('performanceSteps').replaceChildren();
  for (const [key, label] of Object.entries(names)) {
    if (timings[key] == null) continue;
    const term = document.createElement('dt'); term.textContent = label;
    const value = document.createElement('dd'); value.textContent = seconds(timings[key]);
    $('performanceSteps').append(term, value);
  }
  return ` 服务端用时 ${seconds(timings.total_ms)}${reused}。`;
}
function coords(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * canvas.width / rect.width,
    y: (event.clientY - rect.top) * canvas.height / rect.height,
  };
}
function parts() {
  const side = Math.max(1, Math.min(crop[2] - 1, Math.round(crop[2] * .08)));
  const front = crop[2] - side;
  return {side, front, width: state.sx * (front + side * state.depth), height: crop[3] * state.sy};
}
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (mode === 'fineReference' && original) ctx.drawImage(original, 0, 0, canvas.width, canvas.height);
  else if (mode === 'finePreview' && fineCandidate && !$('fineBefore').checked) ctx.drawImage(fineCandidate, 0, 0, canvas.width, canvas.height);
  else if (clean) ctx.drawImage(clean, 0, 0, canvas.width, canvas.height);
  else if (original) ctx.drawImage(original, 0, 0, canvas.width, canvas.height);
  if (product && objectVisible && mode !== 'repair' && !inFineMode()) {
    const p = parts();
    const sideWidth = p.side * state.sx * state.depth;
    const frontWidth = p.front * state.sx;
    ctx.save();
    ctx.translate(state.x + p.width / 2, state.y + p.height / 2);
    ctx.rotate(state.angle * Math.PI / 180);
    ctx.translate(-state.x - p.width / 2, -state.y - p.height / 2);
    if (state.side === 'right') {
      ctx.drawImage(product, 0, 0, p.front, crop[3], state.x, state.y, frontWidth, p.height);
      ctx.drawImage(product, p.front, 0, p.side, crop[3], state.x + frontWidth, state.y, sideWidth, p.height);
    } else {
      ctx.drawImage(product, 0, 0, p.side, crop[3], state.x, state.y, sideWidth, p.height);
      ctx.drawImage(product, p.side, 0, p.front, crop[3], state.x + sideWidth, state.y, frontWidth, p.height);
    }
    ctx.restore();
  }
  if (mode === 'repair' || mode === 'finePaint' || mode === 'fineReference') {
    ctx.save();
    ctx.globalAlpha = .48;
    ctx.drawImage(repairMask, 0, 0);
    ctx.restore();
  }
  if (fineReferenceBox && inFineMode() && mode !== 'finePreview') {
    ctx.save(); ctx.strokeStyle = '#e1c18c'; ctx.lineWidth = Math.max(2, canvas.width / 350);
    ctx.strokeRect(fineReferenceBox[0], fineReferenceBox[1], fineReferenceBox[2]-fineReferenceBox[0], fineReferenceBox[3]-fineReferenceBox[1]); ctx.restore();
  }
  if (selection) {
    ctx.save();
    ctx.strokeStyle = '#d4b0ff';
    ctx.lineWidth = Math.max(2, canvas.width / 350);
    ctx.setLineDash([10, 6]);
    ctx.strokeRect(selection.x, selection.y, selection.w, selection.h);
    ctx.restore();
  }
}
function syncButtons() {
  $('dropZone').classList.toggle('is-busy', busy);
  $('dropZone').setAttribute('aria-busy', String(busy));
  $('selectBox').classList.toggle('is-active', mode === 'box' || mode === 'confirm');
  $('selectionConfirm').hidden = !pendingBox || busy;
  $('confirmSelection').disabled = !pendingBox || busy;
  $('reselectSelection').disabled = busy;
  const available = !!original && !busy && location.protocol !== 'file:';
  $('file').disabled = busy;
  $('process').disabled = !available || (mode === 'repair' || inFineMode()) || !$('prompt').value.trim();
  $('selectBox').disabled = !available || mode === 'repair' || inFineMode();
  $('save').disabled = !product || busy || mode === 'repair' || inFineMode();
  $('reset').disabled = !product || busy || mode === 'repair' || inFineMode();
  $('startRepair').disabled = !product || busy || mode === 'repair' || inFineMode();
  $('deleteObject').disabled = !product || busy || mode === 'repair' || inFineMode() || !objectVisible;
  $('restoreObject').disabled = !product || busy || mode === 'repair' || inFineMode() || objectVisible;
  $('applyRepair').disabled = busy || mode !== 'repair' || !hasRepairPaint;
  $('clearRepair').disabled = busy || mode !== 'repair' || !hasRepairPaint;
  $('cancelRepair').disabled = busy || mode !== 'repair';
  syncFineButtons();
  for (const id of ['width', 'height', 'depth', 'side', 'rotation', 'rotateLeft', 'rotateRight', 'rotationReset']) $(id).disabled = !product || busy || mode === 'repair' || inFineMode();
}
function showOriginalForSelection() {
  if (!original || busy) return;
  clean = null;
  product = null;
  objectVisible = true;
  crop = null;
  selection = null;
  pendingBox = null;
  $('performanceInfo').hidden = true;
  mode = 'box';
  canvas.style.cursor = 'crosshair';
  syncButtons();
  draw();
  status('请从物品左上角拖到右下角，框住整个物品。');
}

function clearRepairMask() {
  repairCtx.clearRect(0, 0, repairMask.width, repairMask.height);
  hasRepairPaint = false;
  draw();
  syncButtons();
}

function paintAt(p, first = false) {
  repairCtx.strokeStyle = '#ff3e66';
  repairCtx.fillStyle = '#ff3e66';
  repairCtx.lineWidth = Number($(inFineMode() ? 'fineBrush' : 'brushSize').value);
  repairCtx.lineCap = 'round';
  repairCtx.lineJoin = 'round';
  if (first) {
    repairCtx.beginPath();
    repairCtx.arc(p.x, p.y, repairCtx.lineWidth / 2, 0, Math.PI * 2);
    repairCtx.fill();
    repairCtx.beginPath();
    repairCtx.moveTo(p.x, p.y);
  } else {
    repairCtx.lineTo(p.x, p.y);
    repairCtx.stroke();
  }
  hasRepairPaint = true;
  draw();
  syncButtons();
}

const canvasBlob = surface => new Promise(resolve => surface.toBlob(resolve, 'image/png'));

async function applyRepair() {
  if (mode !== 'repair' || !hasRepairPaint || busy) return;
  setBusy(true, true);
  status('正在重新修补涂抹区域。');
  try {
    const background = document.createElement('canvas');
    background.width = canvas.width;
    background.height = canvas.height;
    background.getContext('2d').drawImage(clean, 0, 0);
    const data = new FormData();
    data.append('file', await canvasBlob(background), 'background.png');
    data.append('mask', await canvasBlob(repairMask), 'mask.png');
    const response = await fetch('/api/repair', {method: 'POST', body: data});
    const answer = await response.json();
    if (!response.ok) throw new Error(answer.detail || '修补失败');
    clean = await loadImage(answer.clean);
    mode = 'idle';
    canvas.style.cursor = 'grab';
    clearRepairMask();
    status('背景已重新修补。请移动物品检查原位置；如仍有残影，可继续涂抹。' + showPerformance(answer));
  } catch (error) {
    status('修补失败：' + error.message);
  } finally {
    setBusy(false);
  }
}

async function process(selectionInput) {
  if (!file || busy) return;
  if (location.protocol === 'file:') {
    status('当前以 file:// 打开，识别服务不可用。请打开服务提供的网页地址。');
    return;
  }
  const data = new FormData();
  data.append('file', file);
  if (selectionInput.box) data.append('box', selectionInput.box.join(','));
  else data.append('prompt', $('prompt').value.trim());
  pendingBox = null;
  selection = null;
  mode = 'idle';
  canvas.style.cursor = 'grab';
  draw();
  setBusy(true);
  status('正在抠出物品并补全背景。');
  try {
    const response = await fetch('/api/process', {method: 'POST', body: data});
    const answer = await response.json();
    if (!response.ok) throw new Error(answer.detail || '处理失败');
    [clean, product] = await Promise.all([loadImage(answer.clean), loadImage(answer.product)]);
    objectVisible = true;
    fineReferenceBox = null; fineCandidate = null;
    crop = answer.crop;
    state.x = crop[0]; state.y = crop[1];
    state.sx = 1; state.sy = 1; state.depth = 1;
    state.angle = 0;
    $('rotation').value = '0';
    $('rotationValue').textContent = '0°';
    for (const id of ['width', 'height', 'depth']) {
      $(id).value = '100';
      $(id + 'Value').textContent = '100%';
    }
    selection = null;
    draw();
    const timingNote = showPerformance(answer);
    status((answer.confidence == null
      ? '已抠出所选物品。请检查边缘和背景，拖动物品或调整大小。'
      : `已找到物品（定位置信度 ${Math.round(answer.confidence * 100)}%）。请检查边缘和背景。`) + timingNote);
  } catch (error) {
    clean = null; product = null; crop = null; selection = null;
    draw();
    status('处理失败：' + (error instanceof TypeError ? '无法连接识别服务，请确认服务仍在运行。' : error.message));
  } finally {
    setBusy(false);
  }
}

async function openFile(chosenFile) {
  if (!chosenFile || busy) return;
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(chosenFile.type)) {
    status('请上传 PNG、JPG 或 WebP 图片。');
    return;
  }
  if (chosenFile.size > 12 * 1024 * 1024) {
    status('图片超过 12 MB，请先压缩后再上传。');
    return;
  }
  $('performanceInfo').hidden = true;
  file = chosenFile;
  const url = URL.createObjectURL(file);
  try {
    original = await loadImage(url);
    if (original.naturalWidth * original.naturalHeight > 4_000_000) {
      file = null;
      original = null;
      status('图片超过 400 万像素，请缩小后再上传。');
      syncButtons();
      return;
    }
    canvas.width = original.naturalWidth;
    canvas.height = original.naturalHeight;
    repairMask.width = canvas.width;
    repairMask.height = canvas.height;
    hasRepairPaint = false;
    clean = null; product = null; crop = null; selection = null; pendingBox = null; fineReferenceBox = null; fineCandidate = null; objectVisible = true; mode = 'idle';
    canvas.style.cursor = 'grab';
    $('emptyState').classList.add('hidden');
    $('canvasInfo').textContent = `${canvas.width} × ${canvas.height} PX`;
    syncButtons();
    draw();
    status(location.protocol === 'file:'
      ? '当前以 file:// 打开，识别服务不可用。请打开服务提供的网页地址。'
      : '图片已载入。输入中文物品名自动查找，或框选图片中的物品。');
  } catch {
    file = null; original = null; syncButtons(); status('图片载入失败，请换一张图片。');
  } finally {
    URL.revokeObjectURL(url);
  }
}
$('file').onchange = event => openFile(event.target.files[0]);

async function openDemo() {
  if (busy) return;
  try {
    const response = await fetch('/demo.webp?v=4');
    if (!response.ok) throw new Error('示例图读取失败。');
    await openFile(new File([await response.blob()], '示例场景.webp', {type: 'image/webp'}));
    status('示例图已载入。可以框选橙色椅子，也可以试试蓝色花瓶或落地灯。');
    $('editor').scrollIntoView({behavior: 'smooth'});
  } catch (error) {
    status('示例图载入失败：' + error.message);
  }
}
$('heroDemo').onclick = openDemo;
$('emptyDemo').onclick = openDemo;
const dropZone = $('dropZone');
dropZone.ondragover = event => { event.preventDefault(); dropZone.classList.add('drag-over'); };
dropZone.ondragleave = () => dropZone.classList.remove('drag-over');
dropZone.ondrop = event => {
  event.preventDefault();
  dropZone.classList.remove('drag-over');
  openFile(event.dataTransfer.files[0]);
};
$('prompt').oninput = syncButtons;
$('process').onclick = () => process({});
$('selectBox').onclick = () => showOriginalForSelection();
$('confirmSelection').onclick = () => {
  if (!pendingBox || busy) return;
  process({box: [...pendingBox]});
};
$('reselectSelection').onclick = showOriginalForSelection;
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && mode === 'fineReference' && !busy) {
    selection = null; mode = 'finePaint'; draw(); syncButtons();
    status('已取消参考框选，可以继续涂抹或重新选择参考。');
  }
  if (event.key === 'Escape' && mode === 'confirm' && !busy) {
    showOriginalForSelection();
    status('已取消选区，请重新框选物品。');
  }
});
$('brushSize').oninput = () => { $('brushSizeValue').textContent = $('brushSize').value + ' px'; };
$('startRepair').onclick = () => {
  if (!product || busy) return;
  mode = 'repair';
  canvas.style.cursor = 'crosshair';
  clearRepairMask();
  status('物品已暂时隐藏。请涂抹原位置的阴影、模糊和边缘残留，稍微覆盖到干净背景。');
};
$('applyRepair').onclick = applyRepair;
$('clearRepair').onclick = clearRepairMask;
$('cancelRepair').onclick = () => {
  mode = 'idle';
  canvas.style.cursor = 'grab';
  clearRepairMask();
  status('已取消背景修补。');
};

canvas.onpointerdown = event => {
  if (busy || event.button !== 0 || event.isPrimary === false) return;
  const p = coords(event);
  if (mode === 'repair' || mode === 'finePaint') {
    painting = true;
    canvas.setPointerCapture(event.pointerId);
    paintAt(p, true);
    return;
  }
  if (mode === 'box' || mode === 'confirm' || mode === 'fineReference') {
    if (mode !== 'fineReference') mode = 'box';
    pendingBox = null;
    syncButtons();
    selection = {x: p.x, y: p.y, w: 0, h: 0};
    canvas.setPointerCapture(event.pointerId);
    draw();
    return;
  }
  if (!product || inFineMode()) return;
  const {width, height} = parts();
  const radians = -state.angle * Math.PI / 180;
  const dx = p.x - state.x - width / 2;
  const dy = p.y - state.y - height / 2;
  const localX = dx * Math.cos(radians) - dy * Math.sin(radians) + width / 2;
  const localY = dx * Math.sin(radians) + dy * Math.cos(radians) + height / 2;
  if (localX >= 0 && localX <= width && localY >= 0 && localY <= height) {
    drag = {dx: p.x - state.x, dy: p.y - state.y};
    canvas.setPointerCapture(event.pointerId);
  }
};
canvas.onpointermove = event => {
  const p = coords(event);
  if ((mode === 'repair' || mode === 'finePaint') && painting) {
    paintAt(p);
  } else if ((mode === 'box' || mode === 'fineReference') && selection) {
    selection.w = p.x - selection.x;
    selection.h = p.y - selection.y;
    draw();
  } else if (drag) {
    state.x = p.x - drag.dx;
    state.y = p.y - drag.dy;
    draw();
  }
};
canvas.onpointerup = () => {
  painting = false;
  drag = null;
  if ((mode !== 'box' && mode !== 'fineReference') || !selection) return;
  const s = selection;
  const box = [
    Math.max(0, Math.min(s.x, s.x + s.w)),
    Math.max(0, Math.min(s.y, s.y + s.h)),
    Math.min(canvas.width, Math.max(s.x, s.x + s.w)),
    Math.min(canvas.height, Math.max(s.y, s.y + s.h)),
  ];
  if (mode === 'fineReference') {
    if (box[2]-box[0] >= 16 && box[3]-box[1] >= 16) {
      fineReferenceBox = [Math.floor(box[0]), Math.floor(box[1]), Math.ceil(box[2]), Math.ceil(box[3])];
      mode = 'finePaint';
      status('参考纹理已选好。继续涂抹错误背景，或生成修复预览。');
    } else status('参考区域太小，请至少选择 16 × 16 像素的干净背景。');
    selection = null; draw(); syncButtons(); return;
  }
  if (box[2] - box[0] > 20 && box[3] - box[1] > 20) {
    pendingBox = box;
    selection = {x: box[0], y: box[1], w: box[2] - box[0], h: box[3] - box[1]};
    mode = 'confirm';
    status('选区已保留。检查范围后点击“确认识别”，也可重新框选。');
  } else {
    selection = null;
    pendingBox = null;
    status('框选太小，请重新框选。');
  }
  draw();
  syncButtons();
};
canvas.onpointercancel = () => {
  drag = null; painting = false;
  if (mode === 'box' || mode === 'fineReference') { selection = null; pendingBox = null; }
  draw(); syncButtons();
};

for (const id of ['width', 'height', 'depth']) {
  $(id).oninput = () => {
    if (!product) return;
    const before = parts();
    const center = state.x + before.width / 2;
    const bottom = state.y + before.height;
    state[id === 'width' ? 'sx' : id === 'height' ? 'sy' : 'depth'] = Number($(id).value) / 100;
    const after = parts();
    if (id === 'height') state.y = bottom - after.height;
    else state.x = center - after.width / 2;
    $(id + 'Value').textContent = $(id).value + '%';
    draw();
  };
}
function setRotation(angle) {
  if (!product || busy) return;
  state.angle = ((angle + 180) % 360 + 360) % 360 - 180;
  $('rotation').value = String(state.angle);
  $('rotationValue').textContent = state.angle + '°';
  draw();
}
$('rotation').oninput = () => setRotation(Number($('rotation').value));
$('rotateLeft').onclick = () => setRotation(state.angle - 90);
$('rotateRight').onclick = () => setRotation(state.angle + 90);
$('rotationReset').onclick = () => setRotation(0);

$('deleteObject').onclick = () => {
  if (!product || busy || mode === 'repair' || inFineMode()) return;
  objectVisible = false;
  draw();
  syncButtons();
  status('物品已删除。原位置显示 AI 补全后的背景；需要时可点击“恢复物品”。');
};
$('restoreObject').onclick = () => {
  if (!product || busy || mode === 'repair' || inFineMode()) return;
  objectVisible = true;
  draw();
  syncButtons();
  status('物品已恢复，可以继续移动、缩放或旋转。');
};

$('side').onchange = () => { state.side = $('side').value; draw(); };
$('reset').onclick = () => {
  if (!crop) return;
  objectVisible = true;
  state.x = crop[0]; state.y = crop[1];
  state.sx = 1; state.sy = 1; state.depth = 1;
    state.angle = 0;
    $('rotation').value = '0';
    $('rotationValue').textContent = '0°';
  for (const id of ['width', 'height', 'depth']) {
    $(id).value = '100';
    $(id + 'Value').textContent = '100%';
  }
  draw();
};
$('save').onclick = () => canvas.toBlob(blob => {
  if (!blob) return;
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = '摆布-画面编辑.png';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}, 'image/png');

function syncFineButtons() {
  const active = inFineMode();
  const editable = active && mode !== 'finePreview' && !busy;
  $('startFine').disabled = !product || busy || active || mode === 'repair';
  $('fineReference').disabled = !editable;
  $('finePaint').disabled = !editable;
  $('fineClear').disabled = !editable || !hasRepairPaint;
  $('generateFine').disabled = !editable || !hasRepairPaint || !fineReferenceBox;
  $('cancelFine').disabled = !active || busy;
  $('fineEngine').disabled = busy || mode === 'finePreview';
  $('backgroundDescription').disabled = busy || mode === 'finePreview';
  $('fineBrush').disabled = !editable;
  $('finePreview').hidden = mode !== 'finePreview';
  $('fineBefore').disabled = busy;
  $('acceptFine').disabled = busy || !fineCandidate;
  $('discardFine').disabled = busy || !fineCandidate;
  $('fineStatus').textContent = !product ? '先识别一个物品，再开始修复背景。'
    : mode === 'finePreview' ? '预览已生成，可切换修复前后对比。'
    : !active ? '点击开始修复，涂抹补错的背景。'
    : `① 涂抹错误区域${hasRepairPaint ? ' ✓' : ''}　② 框选参考纹理${fineReferenceBox ? ' ✓' : ''}`;
}
function leaveFine() {
  fineCandidate = null; fineReferenceBox = null; selection = null;
  mode = 'idle'; canvas.style.cursor = 'grab'; clearRepairMask();
}
$('startFine').onclick = () => {
  if (!product || busy) return;
  fineCandidate = null; fineReferenceBox = null; selection = null; pendingBox = null; drag = null;
  mode = 'finePaint'; canvas.style.cursor = 'crosshair'; clearRepairMask();
  status('物品已暂时隐藏。只涂抹需要延续同一种背景的区域，保留地毯或地板边界。');
};
$('fineReference').onclick = () => {
  if (busy || !inFineMode()) return;
  mode = 'fineReference'; selection = null; canvas.style.cursor = 'crosshair'; syncButtons(); draw();
  status('当前显示原图。框选干净的参考纹理，不要包含物品、阴影、红色涂抹或不同材质。');
};
$('finePaint').onclick = () => { mode = 'finePaint'; selection = null; draw(); syncButtons(); };
$('fineClear').onclick = clearRepairMask;
$('cancelFine').onclick = () => { leaveFine(); status('已退出精细修复，背景未改变。'); };
$('fineBrush').oninput = () => { $('fineBrushValue').textContent = $('fineBrush').value + ' px'; };
$('fineBefore').onchange = draw;
$('discardFine').onclick = () => {
  fineCandidate = null; mode = 'finePaint'; draw(); syncButtons(); status('已放弃预览，背景未改变。可重新涂抹或选择参考。');
};
$('acceptFine').onclick = () => {
  if (!fineCandidate || busy) return;
  clean = fineCandidate; leaveFine(); status('已采用修复背景。物品位置、尺寸和角度已保留，可继续编辑并导出。');
};
$('generateFine').onclick = async () => {
  if (!inFineMode() || !hasRepairPaint || !fineReferenceBox || busy) return;
  if (!$('backgroundDescription').value.trim()) { status('请填写要延续的背景。'); return; }
  setBusy(true, true);
  $('processingTitle').textContent = '正在生成背景修复预览';
  $('processingDescription').textContent = $('fineEngine').value === 'gpu'
    ? '根据参考纹理和背景说明进行局部重绘' : '根据所选参考区域拼接纹理，原背景会保留到确认采用';
  try {
    const background = document.createElement('canvas'); background.width = canvas.width; background.height = canvas.height;
    background.getContext('2d').drawImage(clean, 0, 0);
    const [x0,y0,x1,y1] = fineReferenceBox;
    const reference = document.createElement('canvas'); reference.width = x1-x0; reference.height = y1-y0;
    reference.getContext('2d').drawImage(original,x0,y0,x1-x0,y1-y0,0,0,x1-x0,y1-y0);
    const data = new FormData();
    data.append('file',await canvasBlob(background),'background.png');
    data.append('mask',await canvasBlob(repairMask),'mask.png');
    data.append('reference',await canvasBlob(reference),'reference.png');
    data.append('reference_box',fineReferenceBox.join(','));
    data.append('description',$('backgroundDescription').value.trim());
    data.append('engine',$('fineEngine').value);
    const response = await fetch('/api/structure-repair',{method:'POST',body:data});
    const answer = await response.json(); if (!response.ok) throw new Error(answer.detail || '生成失败');
    fineCandidate = await loadImage(answer.clean); mode = 'finePreview'; $('fineBefore').checked = false;
    selection = null; draw(); status('修复预览已生成。比较前后效果，满意后点击“确认采用”。' + showPerformance(answer));
  } catch (error) { status('精细修复失败：' + error.message); }
  finally { setBusy(false); }
};
$('fineEngine').onchange = () => {
  $('fineEngineHint').textContent = $('fineEngine').value === 'gpu'
    ? '将背景、修补蒙版及背景说明发送至配置的 GPU 服务；参考纹理用于预填充。生成结果只会应用于涂抹范围。'
    : '本地按参考区域拼接纹理，文字仅作说明；不自动理解材质或重建透视。请把涂抹范围控制在同一种表面内。';
};
fetch('/api/background-capabilities').then(r => r.ok ? r.json() : {}).then(capabilities => {
  const option = $('fineEngine').querySelector('option[value="gpu"]');
  if (capabilities.gpu_generation) { option.disabled = false; option.textContent = 'GPU 生成修复'; }
}).catch(() => {});

if (location.protocol === 'file:') status('当前以 file:// 打开，识别服务不可用。请打开服务提供的网页地址。');
syncButtons();
