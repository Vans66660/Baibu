"""Self-hosted object selection and layered image editor service."""
import base64
import io
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).parent
MAX_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 4_000_000
PUBLIC_MODE = os.getenv("PUBLIC_MODE", "false").lower() == "true"
HOURLY_LIMIT = max(1, int(os.getenv("HOURLY_LIMIT", "12")))
app = FastAPI(title="摆布", docs_url=None if PUBLIC_MODE else "/docs",
              redoc_url=None if PUBLIC_MODE else "/redoc",
              openapi_url=None if PUBLIC_MODE else "/openapi.json")
model_lock = threading.Lock()
models = {}
app.mount("/assets", StaticFiles(directory=ROOT / "static" / "assets"), name="assets")
request_times = defaultdict(deque)
limit_lock = threading.Lock()


@app.middleware("http")
async def public_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def check_limit(request):
    if not PUBLIC_MODE:
        return
    # Uvicorn should trust only the private reverse proxy; do not parse a
    # client-supplied X-Forwarded-For header here.
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with limit_lock:
        entries = request_times[ip]
        while entries and now - entries[0] >= 3600:
            entries.popleft()
        if len(entries) >= HOURLY_LIMIT:
            raise HTTPException(429, "试用额度已用完，请一小时后再试。")
        entries.append(now)


def png_data(image):
    out = io.BytesIO()
    image.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def load_models(need_detector=False):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
    from simple_lama_inpainting import SimpleLama

    if "sam" not in models:
        preferred = os.getenv("MODEL_DEVICE", "auto").lower()
        if preferred not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("MODEL_DEVICE must be auto, cpu, mps or cuda")
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu") if preferred == "auto" else preferred
        segmenter_id = "facebook/sam-vit-base"
        models.update({
            "torch": torch,
            "device": device,
            "sam_processor": SamProcessor.from_pretrained(segmenter_id),
            "sam": SamModel.from_pretrained(segmenter_id).to(device).eval(),
            "lama": SimpleLama(),
        })
    if need_detector and "detector" not in models:
        detector_id = "IDEA-Research/grounding-dino-tiny"
        models.update({
            "detector_processor": AutoProcessor.from_pretrained(detector_id),
            "detector": AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(models["device"]).eval(),
        })
    return models


def detect_box(image, prompt, m):
    prompt = prompt.strip().rstrip(".") + "."
    inputs = m["detector_processor"](images=image, text=prompt, return_tensors="pt").to(m["device"])
    with m["torch"].no_grad():
        output = m["detector"](**inputs)
    result = m["detector_processor"].post_process_grounded_object_detection(
        output, inputs.input_ids, threshold=0.24, text_threshold=0.2,
        target_sizes=[(image.height, image.width)],
    )[0]
    if len(result["scores"]) == 0:
        raise ValueError("没有可靠地找到该物品。请在图片上点击或框选目标后重试。")
    index = int(result["scores"].argmax().item())
    score = float(result["scores"][index].item())
    return result["boxes"][index].tolist(), score


def segment(image, m, box=None, point=None):
    if point is not None:
        inputs = m["sam_processor"](image, input_points=[[[point[0], point[1]]]],
                                    input_labels=[[1]], return_tensors="pt")
    else:
        inputs = m["sam_processor"](image, input_boxes=[[[box]]], return_tensors="pt")
    # SAM's box coordinates may arrive as float64; Apple MPS accepts float32 only.
    for key, value in inputs.items():
        if m["torch"].is_tensor(value) and value.is_floating_point():
            inputs[key] = value.float()
    inputs = inputs.to(m["device"])
    with m["torch"].no_grad():
        output = m["sam"](**inputs)
    masks = m["sam_processor"].image_processor.post_process_masks(
        output.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )[0][0]
    best = int(output.iou_scores[0, 0].argmax().item())
    mask = (masks[best].numpy() > 0).astype(np.uint8) * 255
    # Keep the selected object only if SAM also included unrelated islands.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        selected = 0
        if point is not None:
            selected = int(labels[min(image.height - 1, int(point[1])),
                                  min(image.width - 1, int(point[0]))])
        if selected == 0:
            selected = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
        mask = np.where(labels == selected, 255, 0).astype(np.uint8)
    return mask


def compose_layers(image, mask, lama):
    arr = np.asarray(image)
    ys, xs = np.where(mask > 0)
    if len(xs) < 100:
        raise ValueError("物品遮罩为空。请重新点击或框选完整物品。")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if (x1 - x0) * (y1 - y0) > image.width * image.height * 0.55:
        raise ValueError("识别范围过大。请手动框选物品后重试。")
    alpha = cv2.GaussianBlur(mask, (3, 3), 0.65)
    product = Image.fromarray(np.dstack((arr, alpha)), "RGBA").crop((x0, y0, x1, y1))
    # SAM outlines the solid object, but its contact shadow and antialiased halo
    # often extend beyond that outline. Leave a larger safety margin for LaMa.
    radius = max(8, round(min(image.size) * 0.018))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    repair_mask = cv2.dilate(mask, kernel)
    # Inpaint a context crop, not the whole poster. This avoids processing unrelated text
    # and greatly reduces CPU work for a small selected object.
    pad = max(radius + 48, round(max(x1 - x0, y1 - y0) * 0.28))
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(image.width, x1 + pad), min(image.height, y1 + pad)
    region = image.crop((rx0, ry0, rx1, ry1))
    region_mask = Image.fromarray(repair_mask[ry0:ry1, rx0:rx1], "L")
    repaired = lama(region, region_mask).convert("RGB")
    # LaMa pads dimensions to a multiple of eight and may return that padding.
    repaired = repaired.crop((0, 0, region.width, region.height))
    clean = blend_repair(arr, np.asarray(repaired), repair_mask, (rx0, ry0, rx1, ry1))
    return Image.fromarray(clean), product, [x0, y0, x1 - x0, y1 - y0]


def blend_repair(original, repaired, mask, bounds):
    """Use generated pixels throughout the hole; blend only across its outer rim."""
    x0, y0, x1, y1 = bounds
    local_mask = mask[y0:y1, x0:x1]
    # Blurring the entire mask with the original reintroduces the very shadow
    # being removed. A narrow transition outside the mask avoids that halo.
    transition = cv2.GaussianBlur(local_mask, (0, 0), 1.4).astype(np.float32) / 255.0
    alpha = np.maximum(local_mask.astype(np.float32) / 255.0, transition)
    clean = original.copy()
    base = original[y0:y1, x0:x1].astype(np.float32)
    clean[y0:y1, x0:x1] = (repaired.astype(np.float32) * alpha[..., None]
                            + base * (1 - alpha[..., None])).clip(0, 255).astype(np.uint8)
    return clean


def repair_background(image, mask, lama):
    if image.size != mask.size:
        raise ValueError("修补蒙版与图片尺寸不一致。")
    arr = np.asarray(image)
    raw_mask = np.where(np.asarray(mask.convert("L")) > 0, 255, 0).astype(np.uint8)
    ys, xs = np.where(raw_mask > 0)
    if len(xs) < 25:
        raise ValueError("请先在残留阴影或模糊位置涂抹。")
    if len(xs) > image.width * image.height * 0.35:
        raise ValueError("涂抹范围太大，请缩小后重试。")
    radius = max(3, round(min(image.size) * 0.004))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    repair_mask = cv2.dilate(raw_mask, kernel)
    pad = max(64, round(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.28))
    x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x1, y1 = min(image.width, int(xs.max()) + pad + 1), min(image.height, int(ys.max()) + pad + 1)
    local_mask = repair_mask[y0:y1, x0:x1]
    local_image = arr[y0:y1, x0:x1]
    # For a plain wall or tabletop, surrounding pixels are a stronger guide
    # than a generative model, which can invent a visible gray patch.
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    ring = (cv2.dilate(local_mask, ring_kernel) > 0) & (local_mask == 0)
    ring_pixels = local_image[ring]
    if len(ring_pixels) >= 100 and np.max(np.std(ring_pixels.astype(np.float32), axis=0)) < 18:
        repaired = cv2.inpaint(local_image, local_mask, 5, cv2.INPAINT_TELEA)
    else:
        region = image.crop((x0, y0, x1, y1))
        repaired_image = lama(region, Image.fromarray(local_mask, "L")).convert("RGB")
        repaired = np.asarray(repaired_image.crop((0, 0, region.width, region.height)))
    return Image.fromarray(blend_repair(arr, repaired, repair_mask, (x0, y0, x1, y1)))


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/style.css")
def css():
    return FileResponse(ROOT / "static" / "style.css", media_type="text/css",
                        headers={"Cache-Control": "no-store"})


@app.get("/demo.webp")
def demo_image():
    return FileResponse(ROOT / "static" / "demo.webp", media_type="image/webp",
                        headers={"Cache-Control": "no-store"})


@app.get("/editor.js")
def editor_js():
    return FileResponse(ROOT / "static" / "editor.js", media_type="text/javascript",
                        headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health():
    return {"ok": True, "models_loaded": bool(models), "busy": model_lock.locked()}


def prepare_layers(image, box, point, prompt):
    if not model_lock.acquire(blocking=False):
        raise HTTPException(429, "服务器正在处理另一张图片，请稍后再试。")
    try:
        m = load_models(need_detector=not (box or point))
        if box:
            coords = [float(v) for v in box.split(",")]
            if len(coords) != 4:
                raise ValueError("框选坐标格式不正确。")
            selected, score = coords, None
        elif point:
            coords = [float(v) for v in point.split(",")]
            if len(coords) != 2:
                raise ValueError("点击坐标格式不正确。")
            x, y = coords
            if not (0 <= x < image.width and 0 <= y < image.height):
                raise ValueError("点击位置超出了图片。")
            selected, score = None, None
        else:
            selected, score = detect_box(image, prompt, m)
        if selected is not None:
            x0, y0, x1, y1 = selected
            if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
                raise ValueError("框选范围超出了图片。")
        mask = segment(image, m, box=selected, point=coords if point and not box else None)
        clean, product, crop = compose_layers(image, mask, m["lama"])
        return {"width": image.width, "height": image.height, "crop": crop,
                "confidence": score, "clean": png_data(clean), "product": png_data(product)}
    except ValueError as error:
        raise HTTPException(422, str(error))
    finally:
        model_lock.release()


@app.post("/api/process")
async def process(request: Request, file: UploadFile = File(...), box: str = Form(""),
                  point: str = Form(""), prompt: str = Form("")):
    check_limit(request)
    if file.content_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise HTTPException(415, "请上传 PNG、JPG 或 WebP 图片。")
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "图片不能超过 12 MB。")
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "无法读取这张图片。")
    if image.width * image.height > MAX_PIXELS:
        raise HTTPException(413, "图片像素过大，请缩小到 400 万像素以内。")
    if not box and not point and not prompt.strip():
        raise HTTPException(422, "请输入物品英文名称，或在图片上点击、框选目标。")
    if len(prompt) > 120:
        raise HTTPException(422, "物品名称请控制在 120 个字符以内。")
    return await run_in_threadpool(prepare_layers, image, box, point, prompt)


def prepare_repair(image, mask):
    if not model_lock.acquire(blocking=False):
        raise HTTPException(429, "服务器正在处理另一张图片，请稍后再试。")
    try:
        m = load_models()
        return {"clean": png_data(repair_background(image, mask, m["lama"]))}
    except ValueError as error:
        raise HTTPException(422, str(error))
    finally:
        model_lock.release()


@app.post("/api/repair")
async def repair(request: Request, file: UploadFile = File(...), mask: UploadFile = File(...)):
    check_limit(request)
    raw = await file.read(MAX_BYTES + 1)
    raw_mask = await mask.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or len(raw_mask) > MAX_BYTES:
        raise HTTPException(413, "图片不能超过 12 MB。")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        mask_image = Image.open(io.BytesIO(raw_mask)).convert("L")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "无法读取修补图片。")
    if image.width * image.height > MAX_PIXELS:
        raise HTTPException(413, "图片像素过大，请缩小到 400 万像素以内。")
    return await run_in_threadpool(prepare_repair, image, mask_image)
