"""Self-hosted object selection and layered image editor service."""
import base64
import hashlib
import logging
import json
import math
import io
import os
import re
import threading
import time
from collections import defaultdict, deque, OrderedDict
from functools import lru_cache
from contextlib import contextmanager
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


class MemoryCache:
    """Bounded, sliding-expiry LRU. No images or features are written to disk."""
    def __init__(self, max_bytes, max_items, ttl=300, clock=time.monotonic):
        self.max_bytes, self.max_items, self.ttl = max_bytes, max_items, ttl
        self.clock = clock
        self.entries = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()

    def _purge(self):
        for key, (_, size, expires) in list(self.entries.items()):
            if expires <= self.clock():
                self.bytes -= size
                del self.entries[key]

    def get(self, key):
        with self.lock:
            self._purge()
            entry = self.entries.pop(key, None)
            if entry is None:
                return None
            value, size, _ = entry
            self.entries[key] = (value, size, self.clock() + self.ttl)
            return value

    def put(self, key, value, size):
        with self.lock:
            self._purge()
            old = self.entries.pop(key, None)
            if old:
                self.bytes -= old[1]
            if size > self.max_bytes:
                return
            self.entries[key] = (value, size, self.clock() + self.ttl)
            self.bytes += size
            while self.bytes > self.max_bytes or len(self.entries) > self.max_items:
                _, (_, removed_size, _) = self.entries.popitem(last=False)
                self.bytes -= removed_size


embedding_cache = MemoryCache(16 * 1024 * 1024, 4)
result_cache = MemoryCache(32 * 1024 * 1024, 8)
perf_logger = logging.getLogger("uvicorn.error")


@contextmanager
def measured(timings, name, m=None):
    def sync():
        if m and m.get("device") == "cuda":
            m["torch"].cuda.synchronize()
        elif m and m.get("device") == "mps":
            m["torch"].mps.synchronize()
    sync()
    started = time.perf_counter()
    try:
        yield
    finally:
        sync()
        timings[name] = round((time.perf_counter() - started) * 1000, 2)


def performance(payload, timings, cache, started):
    timings = dict(timings, total_ms=round((time.perf_counter() - started) * 1000, 2))
    perf_logger.info("image_processing %s", json.dumps({"timings_ms": timings, "cache": cache}))
    return dict(payload, timings_ms=timings, cache=cache)


def selection_key(box, point, prompt):
    # Equivalent float spellings share a result; different selections never do.
    kind, text = ("box", box) if box else ("point", point) if point else ("prompt", prompt)
    if kind == "prompt":
        return kind, text.strip().casefold()
    coords = tuple(float(v) for v in text.split(","))
    if len(coords) != (4 if kind == "box" else 2) or not all(math.isfinite(v) for v in coords):
        raise ValueError("选择坐标格式不正确。")
    return kind, coords


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

    if "sam" not in models:
        preferred = os.getenv("MODEL_DEVICE", "auto").lower()
        if preferred not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("MODEL_DEVICE must be auto, cpu, mps or cuda")
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu") if preferred == "auto" else preferred
        sam_device = os.getenv("SAM_DEVICE", device).lower()
        if sam_device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("SAM_DEVICE must be auto, cpu, mps or cuda")
        if sam_device == "auto":
            sam_device = ("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
        backend = os.getenv("SEGMENTER_BACKEND", "mobile_sam").lower()
        if backend not in {"sam", "mobile_sam"}:
            raise ValueError("SEGMENTER_BACKEND must be sam or mobile_sam")
        if backend == "mobile_sam":
            from mobile_sam import sam_model_registry
            revision = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
            checksum = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
            state = torch.hub.load_state_dict_from_url(
                f"https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/{revision}/weights/mobile_sam.pt",
                file_name=f"mobile_sam-{checksum}.pt", check_hash=True,
                map_location="cpu", progress=False, weights_only=True)
            sam = sam_model_registry["vit_t"]()
            sam.load_state_dict(state)
            processor = None
        else:
            segmenter_id = "facebook/sam-vit-base"
            sam = SamModel.from_pretrained(segmenter_id)
            processor = SamProcessor.from_pretrained(segmenter_id)
        models.update({
            "torch": torch, "device": device, "sam_device": sam_device,
            "segmenter_backend": backend,
            "sam_processor": processor, "sam": sam.to(sam_device).eval(),
        })
    if need_detector and "detector" not in models:
        detector_id = "IDEA-Research/grounding-dino-tiny"
        models.update({
            "detector_processor": AutoProcessor.from_pretrained(detector_id),
            "detector": AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(models["device"]).eval(),
        })
    return models


@lru_cache(maxsize=128)
def detection_prompt(prompt):
    """Translate Chinese locally; the image detector consumes English labels."""
    prompt = prompt.strip()
    if not re.search(r"[\u3400-\u9fff]", prompt):
        return prompt
    try:
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        if "translator" not in models:
            model_id = "Helsinki-NLP/opus-mt-zh-en"
            models["translator_tokenizer"] = MarianTokenizer.from_pretrained(model_id)
            models["translator"] = MarianMTModel.from_pretrained(model_id).eval()
        tokenizer = models["translator_tokenizer"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        with torch.inference_mode():
            output = models["translator"].generate(**inputs, max_new_tokens=80, num_beams=4)
        translated = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    except Exception as error:
        raise HTTPException(503, "中文名称处理暂时不可用，请稍后重试，或直接框选物品。") from error
    if not translated or re.search(r"[\u3400-\u9fff]", translated):
        raise ValueError("未能理解这个物品名称，请换一个常见名称，或直接框选物品。")
    return translated


def detect_box(image, prompt, m):
    prompt = prompt.strip().rstrip(".。") + "."
    inputs = m["detector_processor"](images=image, text=prompt, return_tensors="pt").to(m["device"])
    with m["torch"].no_grad():
        output = m["detector"](**inputs)
    result = m["detector_processor"].post_process_grounded_object_detection(
        output, inputs.input_ids, threshold=0.24, text_threshold=0.2,
        target_sizes=[(image.height, image.width)],
    )[0]
    if len(result["scores"]) == 0:
        raise ValueError("没有可靠地找到该物品。请在图片上框选目标后重试。")
    index = int(result["scores"].argmax().item())
    score = float(result["scores"][index].item())
    return result["boxes"][index].tolist(), score


def isolate_selected_mask(mask, image, point=None):
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


def segment_mobile(image, m, box, point, image_key, timings, cache):
    from mobile_sam.utils.transforms import ResizeLongestSide
    torch, model, device = m["torch"], m["sam"], m["sam_device"]
    runtime = {"torch": torch, "device": device}
    original_size = (image.height, image.width)
    transform = ResizeLongestSide(model.image_encoder.img_size)
    feature_key = (image_key, device, "mobile-sam-vit-t-f706ad9")
    with measured(timings, "sam_preprocess_ms", runtime):
        embedding = embedding_cache.get(feature_key) if image_key else None
        cache["image_embedding"] = embedding is not None
        input_size = transform.get_preprocess_shape(*original_size, model.image_encoder.img_size)
        pixels = None
        if embedding is None:
            resized = transform.apply_image(np.asarray(image))
            pixels = torch.as_tensor(resized.copy(), device=device).permute(2, 0, 1).contiguous()[None]
            pixels = model.preprocess(pixels)
    with measured(timings, "sam_embedding_ms", runtime), torch.no_grad():
        if embedding is None:
            embedding = model.image_encoder(pixels)
            if image_key:
                stored = embedding.detach().cpu()
                embedding_cache.put(feature_key, stored, stored.nelement() * stored.element_size())
        embedding = embedding.to(device)
    with measured(timings, "sam_mask_ms", runtime), torch.no_grad():
        points, boxes = None, None
        if point is not None:
            xy = transform.apply_coords(np.asarray([point], dtype=np.float32), original_size)
            points = (torch.as_tensor(xy, device=device)[None], torch.ones((1, 1), dtype=torch.int, device=device))
        else:
            xyxy = transform.apply_boxes(np.asarray(box, dtype=np.float32), original_size)
            boxes = torch.as_tensor(xyxy, device=device).reshape(1, 4)
        sparse, dense = model.prompt_encoder(points=points, boxes=boxes, masks=None)
        low_masks, scores = model.mask_decoder(
            image_embeddings=embedding, image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=True)
        full_masks = model.postprocess_masks(low_masks, input_size, original_size)
        candidate_masks = full_masks[0].detach().cpu().numpy() > model.mask_threshold
        best = choose_mask_index(candidate_masks, scores[0].detach().cpu().numpy(), box)
        mask = candidate_masks[best].astype(np.uint8) * 255
    with measured(timings, "sam_postprocess_ms"):
        mask = isolate_selected_mask(mask, image, point)
    if device == "mps":
        with measured(timings, "gpu_cleanup_ms", runtime):
            del pixels, embedding, sparse, dense, low_masks, scores, full_masks, points, boxes
            torch.mps.empty_cache()
    return mask


def segment(image, m, box=None, point=None, image_key=None, timings=None, cache=None):
    timings = timings if timings is not None else {}
    cache = cache if cache is not None else {}
    if m.get("segmenter_backend") == "mobile_sam":
        return segment_mobile(image, m, box, point, image_key, timings, cache)
    sam_device = m.get("sam_device", m["device"])
    sam_runtime = {"torch": m["torch"], "device": sam_device}
    started = time.perf_counter()
    if point is not None:
        inputs = m["sam_processor"](image, input_points=[[[point[0], point[1]]]],
                                    input_labels=[[1]], return_tensors="pt")
    else:
        inputs = m["sam_processor"](image, input_boxes=[[[box]]], return_tensors="pt")
    # SAM's box coordinates may arrive as float64; Apple MPS accepts float32 only.
    for key, value in inputs.items():
        if m["torch"].is_tensor(value) and value.is_floating_point():
            inputs[key] = value.float()
    inputs = inputs.to(sam_device)
    timings["sam_preprocess_ms"] = round((time.perf_counter() - started) * 1000, 2)
    feature_key = (image_key, sam_device, "sam-vit-base")
    embedding = embedding_cache.get(feature_key) if image_key else None
    cache["image_embedding"] = embedding is not None
    with measured(timings, "sam_embedding_ms", sam_runtime), m["torch"].no_grad():
        if embedding is None:
            embedding = m["sam"].get_image_embeddings(inputs["pixel_values"])
            if image_key:
                stored = embedding.detach().cpu()
                embedding_cache.put(feature_key, stored, stored.nelement() * stored.element_size())
        embedding = embedding.to(sam_device)
    inputs.pop("pixel_values")
    with measured(timings, "sam_mask_ms", sam_runtime), m["torch"].no_grad():
        output = m["sam"](image_embeddings=embedding, **inputs)
    post_started = time.perf_counter()
    masks = m["sam_processor"].image_processor.post_process_masks(
        output.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )[0][0]
    candidate_masks = masks.numpy() > 0
    best = choose_mask_index(candidate_masks, output.iou_scores[0, 0].detach().cpu().numpy(), box)
    mask = candidate_masks[best].astype(np.uint8) * 255
    mask = isolate_selected_mask(mask, image, point)
    timings["sam_postprocess_ms"] = round((time.perf_counter() - post_started) * 1000, 2)
    if sam_device == "mps":
        # Release transient attention buffers before the CPU inpainting model runs.
        with measured(timings, "gpu_cleanup_ms", sam_runtime):
            del inputs, output, embedding
            m["torch"].mps.empty_cache()
    return mask


def load_inpainter(m):
    if "lama" not in m:
        from simple_lama_inpainting import SimpleLama
        m["lama"] = SimpleLama()
    return m["lama"]


def choose_mask_index(masks, scores, box=None):
    """Balance SAM's predicted quality with fit to the requested object extent."""
    quality = np.asarray(scores, dtype=np.float32).copy()
    if box is None:
        return int(quality.argmax())
    x0, y0, x1, y1 = box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    for i, candidate in enumerate(masks):
        ys, xs = np.where(candidate)
        if len(xs) < 25:
            quality[i] = -10
            continue
        # Penalize masks leaking outside the box, without forcing a rectangle.
        inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
        spill = 1 - float(inside.mean())
        extent = min(1., (xs.max() - xs.min() + 1) / bw) * min(1., (ys.max() - ys.min() + 1) / bh)
        quality[i] += .16 * extent - .45 * spill
    return int(quality.argmax())


def fit_background_plane(arr, support, tolerance=6.):
    """Fit a smooth RGB surface only when surrounding samples agree with it."""
    ys, xs = np.where(support)
    if len(xs) < 150:
        return None
    step = max(1, len(xs) // 6000)
    ys, xs = ys[::step], xs[::step]
    h, w = arr.shape[:2]
    design = np.column_stack((np.ones(len(xs)), xs / max(1, w - 1), ys / max(1, h - 1)))
    rgb = arr[ys, xs].astype(np.float32)
    keep = np.ones(len(xs), dtype=bool)
    # Trim isolated foreground clutter, but reject a genuinely textured surround.
    for _ in range(3):
        coefficients = np.linalg.lstsq(design[keep], rgb[keep], rcond=None)[0]
        errors = np.max(np.abs(design @ coefficients - rgb), axis=1)
        keep = errors <= max(tolerance, float(np.quantile(errors, .8)))
    if np.quantile(errors, .9) > tolerance:
        return None
    yy, xx = np.mgrid[:h, :w]
    return (coefficients[0] + xx[..., None] / max(1, w - 1) * coefficients[1]
            + yy[..., None] / max(1, h - 1) * coefficients[2]).clip(0, 255).astype(np.uint8)


def object_repair_mask(arr, mask):
    """Expand background removal separately from the object's alpha matte.

    Shadow detection is deliberately bounded and requires a smooth, consistent
    surrounding surface. It is a color/geometry heuristic, not a shadow model.
    """
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return mask.copy()
    h, w = mask.shape
    bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    radius = max(8, round(min(h, w) * .018))
    disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    base = cv2.dilate(mask, disk(radius))
    reach = min(96, max(radius + 12, round(max(bw, bh) * .6)))
    pad = reach + 24
    x0, y0 = max(0, int(xs.min())-pad), max(0, int(ys.min())-pad)
    x1, y1 = min(w, int(xs.max())+pad+1), min(h, int(ys.max())+pad+1)
    local = mask[y0:y1, x0:x1]
    nearby = cv2.dilate(local, disk(reach)) > 0
    outer = (cv2.dilate(local, disk(reach+20)) > 0) & ~nearby
    plane = fit_background_plane(arr[y0:y1, x0:x1], outer, tolerance=10.)
    if plane is None:
        return base
    lab = cv2.cvtColor(arr[y0:y1, x0:x1], cv2.COLOR_RGB2LAB).astype(np.float32)
    reference = cv2.cvtColor(plane, cv2.COLOR_RGB2LAB).astype(np.float32)
    darkness = reference[..., 0] - lab[..., 0]
    chroma = np.linalg.norm(lab[..., 1:] - reference[..., 1:], axis=2)
    candidates = nearby & (local == 0) & (darkness > 7) & (darkness < 110) & (chroma < 13)
    # Include only dark regions connected to the object's existing safety rim.
    local_base = base[y0:y1, x0:x1] > 0
    connected = (candidates | local_base).astype(np.uint8)
    _, labels = cv2.connectedComponents(connected, connectivity=8)
    anchored = np.unique(labels[local_base])
    added = candidates & np.isin(labels, anchored[anchored != 0])
    if np.count_nonzero(added) > np.count_nonzero(local) * .7:
        return base
    extra = cv2.dilate(added.astype(np.uint8)*255, disk(3))
    result = base.copy()
    result[y0:y1, x0:x1] |= extra
    return result


def inpaint_region(image, repair_mask, bounds, lama):
    x0, y0, x1, y1 = bounds
    region = image.crop(bounds)
    arr = np.asarray(region)
    local_mask = repair_mask[y0:y1, x0:x1]
    ring = (cv2.dilate(local_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))) > 0) & (local_mask == 0)
    plane = fit_background_plane(arr, ring)
    if plane is not None:
        return plane
    repaired = lama(region, Image.fromarray(local_mask, 'L')).convert('RGB')
    return np.asarray(repaired.crop((0, 0, region.width, region.height)))


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
    # The product alpha and the background removal mask serve different purposes.
    repair_mask = object_repair_mask(arr, mask)
    rys, rxs = np.where(repair_mask > 0)
    radius = max(8, round(min(image.size) * .018))
    # Inpaint a context crop, not the whole poster. This avoids processing unrelated text
    # and greatly reduces CPU work for a small selected object.
    pad = max(radius + 48, round(max(x1 - x0, y1 - y0) * 0.28))
    rx0, ry0 = max(0, int(rxs.min()) - pad), max(0, int(rys.min()) - pad)
    rx1, ry1 = min(image.width, int(rxs.max()) + 1 + pad), min(image.height, int(rys.max()) + 1 + pad)
    bounds = (rx0, ry0, rx1, ry1)
    repaired = inpaint_region(image, repair_mask, bounds, lama)
    clean = blend_repair(arr, repaired, repair_mask, bounds)
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
    repaired = inpaint_region(image, repair_mask, (x0, y0, x1, y1), lama)
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
    return {"ok": True, "models_loaded": bool(models), "busy": model_lock.locked(),
            "sam_device": models.get("sam_device"), "detector_device": models.get("device"),
            "segmenter_backend": models.get("segmenter_backend")}


def prepare_layers(image, box, point, prompt, started=None, initial_timings=None):
    started = started if started is not None else time.perf_counter()
    timings = dict(initial_timings or {})
    cache = {"result": False, "image_embedding": False, "translation": False}
    try:
        request_selection = selection_key(box, point, prompt)
    except ValueError as error:
        raise HTTPException(422, str(error))
    with measured(timings, "fingerprint_ms"):
        image_key = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    result_key = (image_key, request_selection)
    with measured(timings, "cache_lookup_ms"):
        cached = result_cache.get(result_key)
    if cached is not None:
        cache["result"] = True
        return performance(cached, timings, cache, started)
    if not model_lock.acquire(blocking=False):
        raise HTTPException(429, "服务器正在处理另一张图片，请稍后再试。")
    try:
        with measured(timings, "model_load_ms"):
            m = load_models(need_detector=not (box or point))
        if box:
            coords = list(request_selection[1])
            selected, score = coords, None
        elif point:
            coords = list(request_selection[1])
            x, y = coords
            if not (0 <= x < image.width and 0 <= y < image.height):
                raise ValueError("点击位置超出了图片。")
            selected, score = None, None
        else:
            before = detection_prompt.cache_info().hits
            with measured(timings, "translation_ms"):
                translated = detection_prompt(prompt.strip())
            cache["translation"] = detection_prompt.cache_info().hits > before
            with measured(timings, "detection_ms", m):
                selected, score = detect_box(image, translated, m)
        if selected is not None:
            x0, y0, x1, y1 = selected
            if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
                raise ValueError("框选范围超出了图片。")
        mask = segment(image, m, box=selected, point=coords if point and not box else None,
                       image_key=image_key, timings=timings, cache=cache)
        with measured(timings, "background_model_load_ms"):
            load_inpainter(m)
        with measured(timings, "background_ms", m):
            clean, product, crop = compose_layers(image, mask, m["lama"])
        with measured(timings, "encode_ms"):
            payload = {"width": image.width, "height": image.height, "crop": crop,
                       "confidence": score, "clean": png_data(clean), "product": png_data(product)}
        result_cache.put(result_key, payload, len(payload["clean"]) + len(payload["product"]) + 1024)
        return performance(payload, timings, cache, started)
    except ValueError as error:
        raise HTTPException(422, str(error))
    finally:
        model_lock.release()


@app.post("/api/process")
async def process(request: Request, file: UploadFile = File(...), box: str = Form(""),
                  point: str = Form(""), prompt: str = Form("")):
    started = time.perf_counter()
    check_limit(request)
    if file.content_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise HTTPException(415, "请上传 PNG、JPG 或 WebP 图片。")
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "图片不能超过 12 MB。")
    decode_started = time.perf_counter()
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "无法读取这张图片。")
    if image.width * image.height > MAX_PIXELS:
        raise HTTPException(413, "图片像素过大，请缩小到 400 万像素以内。")
    if not box and not point and not prompt.strip():
        raise HTTPException(422, "请输入物品名称（支持中文），或在图片上框选目标。")
    if len(prompt) > 120:
        raise HTTPException(422, "物品名称请控制在 120 个字符以内。")
    return await run_in_threadpool(prepare_layers, image, box, point, prompt, started,
                                   {"decode_ms": round((time.perf_counter() - decode_started) * 1000, 2)})


def prepare_repair(image, mask):
    started, timings = time.perf_counter(), {}
    if not model_lock.acquire(blocking=False):
        raise HTTPException(429, "服务器正在处理另一张图片，请稍后再试。")
    try:
        with measured(timings, "model_load_ms"):
            m = load_models()
        with measured(timings, "background_model_load_ms"):
            load_inpainter(m)
        with measured(timings, "background_ms", m):
            clean = repair_background(image, mask, m["lama"])
        with measured(timings, "encode_ms"):
            payload = {"clean": png_data(clean)}
        return performance(payload, timings, {"result": False}, started)
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


@app.get('/api/background-capabilities')
def background_capabilities():
    return {'reference_texture': True,
            'gpu_generation': bool(os.getenv('BACKGROUND_WORKER_URL') and os.getenv('BACKGROUND_WORKER_TOKEN'))}


def prepare_structure_repair(image, mask, reference, reference_box, description, engine):
    from background_repair import validate_inputs, repair_texture
    import httpx
    started, timings = time.perf_counter(), {}
    if engine not in {'reference', 'gpu'}:
        raise HTTPException(422, '未知修复方式。')
    if engine == 'gpu' and not background_capabilities()['gpu_generation']:
        raise HTTPException(503, '尚未连接 GPU 修复服务，请先使用本地参考纹理修复。')
    try:
        raw_mask = validate_inputs(image, mask, reference, reference_box)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    identity = hashlib.sha256(image.tobytes()+raw_mask.tobytes()+reference.tobytes()
                              +str((image.size,reference.size,reference_box,description,engine)).encode()).hexdigest()
    key = ('structure-v2', identity)
    cached = result_cache.get(key)
    if cached:
        return performance(cached, timings, {'result': True}, started)
    if not model_lock.acquire(blocking=False):
        raise HTTPException(429, '服务器正在处理另一张图片，请稍后再试。')
    try:
        with measured(timings, 'reference_texture_ms'):
            candidate = repair_texture(image, raw_mask, reference)
        if engine == 'gpu':
            with measured(timings, 'translation_ms'):
                instruction = detection_prompt(description)
            with measured(timings, 'generation_ms'):
                def encoded(im):
                    buffer = io.BytesIO(); im.save(buffer,format='PNG'); return buffer.getvalue()
                try:
                    # URL is configured by the operator, never accepted from a visitor.
                    response = httpx.post(os.environ['BACKGROUND_WORKER_URL'].rstrip('/')+'/inpaint',
                        headers={'Authorization':'Bearer '+os.environ['BACKGROUND_WORKER_TOKEN']},
                        files={'image':('scene.png',encoded(candidate),'image/png'),
                               'mask':('mask.png',encoded(Image.fromarray(raw_mask)),'image/png')},
                        data={'prompt':instruction}, timeout=300, follow_redirects=False, trust_env=False)
                    if response.status_code != 200 or len(response.content) > MAX_BYTES:
                        raise ValueError('worker response')
                    generated = Image.open(io.BytesIO(response.content))
                    if generated.size != image.size:raise ValueError('worker dimensions')
                    generated = np.asarray(generated.convert('RGB'))
                except (httpx.HTTPError, OSError, ValueError) as error:
                    raise HTTPException(502, 'GPU 修复服务暂不可用，原背景未改变。可以切回本地参考纹理修复。') from error
                # Even if a generation model changes other regions, only the requested
                # hole is composited. Do not trust the worker to enforce preservation.
                alpha = np.minimum(1., cv2.distanceTransform((raw_mask>0).astype(np.uint8),cv2.DIST_L2,3)/2.)[...,None]
                candidate = Image.fromarray(np.rint(generated*alpha+np.asarray(image)*(1-alpha)).clip(0,255).astype(np.uint8))
        payload = {'clean':png_data(candidate),'engine':engine,'description':description}
        result_cache.put(key,payload,len(payload['clean'])+1024)
        return performance(payload,timings,{'result':False},started)
    finally:
        model_lock.release()


@app.post('/api/structure-repair')
async def structure_repair(request: Request, file: UploadFile = File(...), mask: UploadFile = File(...),
                           reference: UploadFile = File(...), reference_box: str = Form(...),
                           description: str = Form('延续原有背景'), engine: str = Form('reference')):
    check_limit(request)
    if not description.strip() or len(description)>240:
        raise HTTPException(422,'请用不超过 240 个字描述要延续的背景。')
    images = []
    for upload in (file,mask,reference):
        raw = await upload.read(MAX_BYTES+1)
        if len(raw)>MAX_BYTES:raise HTTPException(413,'图片不能超过 12 MB。')
        try:
            im = Image.open(io.BytesIO(raw))
            if im.width*im.height>MAX_PIXELS:raise HTTPException(413,'图片不能超过 400 万像素。')
            images.append(im.convert('L' if upload is mask else 'RGB'))
        except (UnidentifiedImageError,OSError,ValueError) as error:
            raise HTTPException(400,'无法读取修复图片。') from error
    try:
        bounds = [int(v) for v in reference_box.split(',')]
        if len(bounds)!=4:raise ValueError()
    except ValueError as error:
        raise HTTPException(422,'参考范围格式不正确。') from error
    return await run_in_threadpool(prepare_structure_repair,*images,bounds,description.strip(),engine)
