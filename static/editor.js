const $ = id => document.getElementById(id);
const canvas = $('view');
const ctx = canvas.getContext('2d');
let file = null;
let original = null;
let clean = null;
let product = null;
let crop = null;
let selection = null;
let mode = 'idle';
let drag = null;
let busy = false;
const repairMask = document.createElement('canvas');
const repairCtx = repairMask.getContext('2d');
let painting = false;
let hasRepairPaint = false;

const state = {x: 0, y: 0, sx: 1, sy: 1, depth: 1, side: 'right'};
const loadImage = src => new Promise((resolve, reject) => {
  const image = new Image();
  image.onload = () => resolve(image);
  image.onerror = reject;
  image.src = src;
});

function status(message) { $('status').textContent = message; }
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
  if (clean) ctx.drawImage(clean, 0, 0, canvas.width, canvas.height);
  else if (original) ctx.drawImage(original, 0, 0, canvas.width, canvas.height);
  if (product && mode !== 'repair') {
    const p = parts();
    const sideWidth = p.side * state.sx * state.depth;
    const frontWidth = p.front * state.sx;
    if (state.side === 'right') {
      ctx.drawImage(product, 0, 0, p.front, crop[3], state.x, state.y, frontWidth, p.height);
      ctx.drawImage(product, p.front, 0, p.side, crop[3], state.x + frontWidth, state.y, sideWidth, p.height);
    } else {
      ctx.drawImage(product, 0, 0, p.side, crop[3], state.x, state.y, sideWidth, p.height);
      ctx.drawImage(product, p.side, 0, p.front, crop[3], state.x + sideWidth, state.y, frontWidth, p.height);
    }
  }
  if (mode === 'repair') {
    ctx.save();
    ctx.globalAlpha = .48;
    ctx.drawImage(repairMask, 0, 0);
    ctx.restore();
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
  $('selectPoint').classList.toggle('is-active', mode === 'point');
  $('selectBox').classList.toggle('is-active', mode === 'box');
  const available = !!original && !busy && location.protocol !== 'file:';
  $('file').disabled = busy;
  $('process').disabled = !available || mode === 'repair' || !$('prompt').value.trim();
  $('selectPoint').disabled = !available || mode === 'repair';
  $('selectBox').disabled = !available || mode === 'repair';
  $('save').disabled = !product || busy;
  $('reset').disabled = !product || busy;
  $('startRepair').disabled = !product || busy || mode === 'repair';
  $('applyRepair').disabled = busy || mode !== 'repair' || !hasRepairPaint;
  $('clearRepair').disabled = busy || mode !== 'repair' || !hasRepairPaint;
  $('cancelRepair').disabled = busy || mode !== 'repair';
  for (const id of ['width', 'height', 'depth', 'side']) $(id).disabled = !product || busy;
}
function showOriginalForSelection(nextMode) {
  if (!original || busy) return;
  clean = null;
  product = null;
  crop = null;
  selection = null;
  mode = nextMode;
  canvas.style.cursor = 'crosshair';
  syncButtons();
  draw();
  status(nextMode === 'point' ? '请点击要编辑的物品内部。' : '请从物品左上角拖到右下角，框住整个物品。');
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
  repairCtx.lineWidth = Number($('brushSize').value);
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
  busy = true;
  syncButtons();
  status('正在重新修补涂抹区域，可能需要几分钟。');
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
    status('背景已重新修补。请移动物品检查原位置；如仍有残影，可继续涂抹。');
  } catch (error) {
    status('修补失败：' + error.message);
  } finally {
    busy = false;
    syncButtons();
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
  else if (selectionInput.point) data.append('point', selectionInput.point.join(','));
  else data.append('prompt', $('prompt').value.trim());
  busy = true;
  mode = 'idle';
  canvas.style.cursor = 'grab';
  syncButtons();
  status('正在抠出物品并补全背景。本机处理长图可能需要几分钟，请保持页面打开。');
  try {
    const response = await fetch('/api/process', {method: 'POST', body: data});
    const answer = await response.json();
    if (!response.ok) throw new Error(answer.detail || '处理失败');
    [clean, product] = await Promise.all([loadImage(answer.clean), loadImage(answer.product)]);
    crop = answer.crop;
    state.x = crop[0]; state.y = crop[1];
    state.sx = 1; state.sy = 1; state.depth = 1;
    for (const id of ['width', 'height', 'depth']) {
      $(id).value = '100';
      $(id + 'Value').textContent = '100%';
    }
    selection = null;
    draw();
    status(answer.confidence == null
      ? '已抠出所选物品。请检查边缘和背景，拖动物品或调整大小。'
      : `已找到物品（定位置信度 ${Math.round(answer.confidence * 100)}%）。请检查边缘和背景。`);
  } catch (error) {
    clean = null; product = null; crop = null; selection = null;
    draw();
    status('处理失败：' + (error instanceof TypeError ? '无法连接识别服务，请确认服务仍在运行。' : error.message));
  } finally {
    busy = false;
    syncButtons();
  }
}

async function openFile(chosenFile) {
  if (!chosenFile) return;
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(chosenFile.type)) {
    status('请上传 PNG、JPG 或 WebP 图片。');
    return;
  }
  if (chosenFile.size > 12 * 1024 * 1024) {
    status('图片超过 12 MB，请先压缩后再上传。');
    return;
  }
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
    clean = null; product = null; crop = null; selection = null; mode = 'idle';
    canvas.style.cursor = 'grab';
    $('emptyState').classList.add('hidden');
    $('canvasInfo').textContent = `${canvas.width} × ${canvas.height} PX`;
    syncButtons();
    draw();
    status(location.protocol === 'file:'
      ? '当前以 file:// 打开，识别服务不可用。请打开服务提供的网页地址。'
      : '图片已载入。输入英文物品名自动查找，或直接点击／框选图片中的物品。');
  } catch {
    file = null; original = null; syncButtons(); status('图片载入失败，请换一张图片。');
  } finally {
    URL.revokeObjectURL(url);
  }
}
$('file').onchange = event => openFile(event.target.files[0]);

async function openDemo() {
  try {
    const response = await fetch('/demo.webp?v=4');
    if (!response.ok) throw new Error('示例图读取失败。');
    await openFile(new File([await response.blob()], '示例场景.webp', {type: 'image/webp'}));
    status('示例图已载入。可以点选橙色椅子，也可以试试蓝色花瓶或落地灯。');
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
$('selectPoint').onclick = () => showOriginalForSelection('point');
$('selectBox').onclick = () => showOriginalForSelection('box');
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
  if (busy) return;
  const p = coords(event);
  if (mode === 'repair') {
    painting = true;
    canvas.setPointerCapture(event.pointerId);
    paintAt(p, true);
    return;
  }
  if (mode === 'point') {
    process({point: [p.x, p.y]});
    return;
  }
  if (mode === 'box') {
    selection = {x: p.x, y: p.y, w: 0, h: 0};
    canvas.setPointerCapture(event.pointerId);
    draw();
    return;
  }
  if (!product) return;
  const {width, height} = parts();
  if (p.x >= state.x && p.x <= state.x + width && p.y >= state.y && p.y <= state.y + height) {
    drag = {dx: p.x - state.x, dy: p.y - state.y};
    canvas.setPointerCapture(event.pointerId);
  }
};
canvas.onpointermove = event => {
  const p = coords(event);
  if (mode === 'repair' && painting) {
    paintAt(p);
  } else if (mode === 'box' && selection) {
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
  if (mode !== 'box' || !selection) return;
  const s = selection;
  const box = [
    Math.max(0, Math.min(s.x, s.x + s.w)),
    Math.max(0, Math.min(s.y, s.y + s.h)),
    Math.min(canvas.width, Math.max(s.x, s.x + s.w)),
    Math.min(canvas.height, Math.max(s.y, s.y + s.h)),
  ];
  selection = null;
  draw();
  if (box[2] - box[0] > 20 && box[3] - box[1] > 20) process({box});
  else status('框选太小，请重新框选。');
};
canvas.onpointercancel = () => { drag = null; painting = false; selection = null; draw(); };

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
$('side').onchange = () => { state.side = $('side').value; draw(); };
$('reset').onclick = () => {
  if (!crop) return;
  state.x = crop[0]; state.y = crop[1];
  state.sx = 1; state.sy = 1; state.depth = 1;
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

if (location.protocol === 'file:') status('当前以 file:// 打开，识别服务不可用。请打开服务提供的网页地址。');
syncButtons();
