"""Reference-guided local texture quilting. Does not infer hidden scene geometry."""
import hashlib
import cv2
import numpy as np
from PIL import Image


def validate_inputs(image, mask, reference, reference_box):
    if image.size != mask.size:
        raise ValueError('修补蒙版与场景尺寸不一致。')
    raw = (np.asarray(mask.convert('L')) > 0).astype(np.uint8) * 255
    area = int(np.count_nonzero(raw))
    if area < 25:
        raise ValueError('请先涂抹需要恢复的背景。')
    if area > image.width * image.height * .35:
        raise ValueError('一次最多修补画面的 35%，请分区域处理。')
    if reference.width < 16 or reference.height < 16:
        raise ValueError('参考区域太小，请至少框选 16 × 16 像素的干净背景。')
    x0, y0, x1, y1 = reference_box
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError('参考范围超出了图片。')
    if reference.size != (x1-x0, y1-y0):
        raise ValueError('参考图片与选框尺寸不一致。')
    if np.any(raw[y0:y1, x0:x1]):
        raise ValueError('参考区域与涂抹范围重叠，请选择未损坏的背景。')
    return raw


def minimum_seam(cost):
    """A top-to-bottom minimum-cost cut through an overlap strip."""
    h, w = cost.shape
    total = cost.copy()
    parents = np.zeros((h, w), np.int32)
    for y in range(1, h):
        previous = np.pad(total[y-1], (1,1), constant_values=np.inf)
        choices = np.stack((previous[:w], previous[1:w+1], previous[2:w+2]))
        delta = choices.argmin(axis=0)-1
        parents[y] = np.arange(w)+delta
        total[y] += choices.min(axis=0)
    seam = np.empty(h, np.int32);seam[-1] = int(total[-1].argmin())
    for y in range(h-1, 0, -1):seam[y-1] = parents[y, seam[y]]
    return seam


def repair_texture(image, raw_mask, reference):
    """Quilt reference patches, retaining original pixels exactly outside the mask."""
    arr = np.asarray(image.convert('RGB'))
    ref = np.asarray(reference.convert('RGB'))
    # Keep fine weave/grain, remove broad lighting bands before repeating patches.
    low = cv2.GaussianBlur(ref.astype(np.float32), (0,0), max(3.,min(ref.shape[:2])/5.))
    tone = np.median(ref.reshape(-1,3),axis=0)
    ref = np.rint(ref.astype(np.float32)-low+tone).clip(0,255).astype(np.uint8)
    ys, xs = np.where(raw_mask > 0)
    tile = min(48, ref.shape[0], ref.shape[1])
    overlap = max(4, tile//4);stride = tile-overlap
    pad = tile
    x0, y0 = max(0, int(xs.min())-pad), max(0, int(ys.min())-pad)
    x1, y1 = min(image.width, int(xs.max())+pad+1), min(image.height, int(ys.max())+pad+1)
    target = arr[y0:y1, x0:x1].copy()
    hole = raw_mask[y0:y1, x0:x1] > 0
    known = ~hole.copy()
    seed = int.from_bytes(hashlib.sha256(ref.tobytes()+raw_mask.tobytes()).digest()[:4], 'little')
    rng = np.random.default_rng(seed)
    for y in range(0, target.shape[0], stride):
        for x in range(0, target.shape[1], stride):
            th, tw = min(tile,target.shape[0]-y), min(tile,target.shape[1]-x)
            local_hole = hole[y:y+th, x:x+tw]
            if not local_hole.any():continue
            current = target[y:y+th, x:x+tw]
            valid = known[y:y+th, x:x+tw]
            best, score = None, float('inf')
            for _ in range(24):
                sy = int(rng.integers(0,ref.shape[0]-th+1));sx = int(rng.integers(0,ref.shape[1]-tw+1))
                patch = ref[sy:sy+th, sx:sx+tw]
                error = float(np.mean((patch[valid].astype(float)-current[valid])**2)) if valid.any() else float(rng.random())
                if error < score:best, score = patch, error
            use = np.ones((th,tw),bool)
            if x and tw >= overlap:
                cost = np.mean((current[:,:overlap].astype(float)-best[:,:overlap])**2,axis=2)
                seam = minimum_seam(cost)
                use[:,:overlap] = np.arange(overlap)[None,:] >= seam[:,None]
            if y and th >= overlap:
                cost = np.mean((current[:overlap].astype(float)-best[:overlap])**2,axis=2)
                seam = minimum_seam(cost.T)
                use[:overlap] &= np.arange(overlap)[:,None] >= seam[None,:]
            # The known background is never overwritten, including inside patch overlaps.
            use &= local_hole
            use |= local_hole & ~valid
            current[use] = best[use]
            known[y:y+th,x:x+tw][use] = True
    # Match boundary illumination in the gradient domain. Only use the clone
    # inside the painted hole; explicitly restore exterior pixels below.
    if hole.shape[0] > 8 and hole.shape[1] > 8 and hole.any():
        try:
            clone_mask = hole.astype(np.uint8)*255
            center = ((int(xs.min())+int(xs.max())+1)//2-x0,
                      (int(ys.min())+int(ys.max())+1)//2-y0)
            cloned = cv2.seamlessClone(target, arr[y0:y1,x0:x1], clone_mask,
                                       center, cv2.NORMAL_CLONE)
            # Poisson's low-frequency color follows the destination (possibly
            # an incorrect gray floor). Restrict that adjustment to the rim.
            rim = (.5*np.exp(-cv2.distanceTransform(hole.astype(np.uint8),cv2.DIST_L2,3)/5.))[...,None]
            target = np.rint(cloned*rim+target*(1-rim)).clip(0,255).astype(np.uint8)
        except cv2.error:
            pass
    # Feather inward: no outside pixel is modified, and old gray patches are not
    # mixed through the interior. This does not generate perspective or objects.
    distance = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.minimum(1., distance/2.)[...,None]
    base = arr[y0:y1,x0:x1]
    result = arr.copy()
    result[y0:y1,x0:x1] = np.rint(target*alpha + base*(1-alpha)).clip(0,255).astype(np.uint8)
    return Image.fromarray(result)
