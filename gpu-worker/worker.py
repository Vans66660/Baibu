"""Optional authenticated SDXL inpainting worker. Requires an NVIDIA GPU."""
import hmac
import io
import os
import threading
from contextlib import asynccontextmanager

import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

TOKEN = os.getenv('BACKGROUND_WORKER_TOKEN','')
if len(TOKEN)<24:
    raise RuntimeError('Set BACKGROUND_WORKER_TOKEN to a random secret of at least 24 characters.')
MODEL_ID = os.getenv('BACKGROUND_MODEL_ID','diffusers/stable-diffusion-xl-1.0-inpainting-0.1')
lock = threading.Lock()
pipe = None
MAX_BYTES = 12*1024*1024


@asynccontextmanager
async def lifespan(app):
    global pipe
    if not torch.cuda.is_available():raise RuntimeError('This worker requires CUDA. Use the local reference mode on a Mac.')
    pipe = AutoPipelineForInpainting.from_pretrained(MODEL_ID,torch_dtype=torch.float16,
                                                     variant='fp16',use_safetensors=True)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    yield

app = FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)


def authorize(header):
    if not hmac.compare_digest((header or '').encode(), ('Bearer '+TOKEN).encode()):
        raise HTTPException(401,'Unauthorized')


def generate(image, mask, prompt):
    if not lock.acquire(blocking=False):raise HTTPException(429,'Worker busy')
    try:
        raw = np.asarray(mask)>0
        ys,xs = np.where(raw)
        if len(xs)<25 or len(xs)>image.width*image.height*.35:raise HTTPException(422,'Invalid mask area')
        # Keep the model's aspect ratio intact. Restrict generation to a context
        # crop rather than distorting the whole image into a square.
        pad = 96
        bounds=(max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),
                min(image.width,int(xs.max())+pad+1),min(image.height,int(ys.max())+pad+1))
        region=image.crop(bounds);hole=mask.crop(bounds)
        scale=1024/max(region.size)
        target=tuple(max(64,int(round(v*scale/8))*8) for v in region.size)
        result=pipe(prompt='Photorealistic empty background. Continue the existing surface: '+prompt+'. Preserve background boundaries, texture direction and perspective.',
                    negative_prompt='chair, sofa, furniture in the hole, object silhouette, cast shadow, text, watermark, blurry patch',
                    image=region.resize(target,Image.Resampling.LANCZOS),
                    mask_image=hole.resize(target,Image.Resampling.NEAREST),
                    width=target[0],height=target[1],num_inference_steps=30,strength=.95,guidance_scale=7.5,
                    generator=torch.Generator('cpu').manual_seed(42)).images[0]
        result=result.resize(region.size,Image.Resampling.LANCZOS)
        candidate=image.copy();candidate.paste(result,bounds[:2],hole)
        output=io.BytesIO();candidate.save(output,format='PNG');return output.getvalue()
    finally:lock.release()


@app.get('/health')
def health(authorization: str | None = Header(None)):
    authorize(authorization)
    return {'ready':pipe is not None,'busy':lock.locked()}


@app.post('/inpaint')
async def inpaint(image: UploadFile = File(...),mask: UploadFile = File(...),
                  prompt: str = Form(...), authorization: str | None = Header(None)):
    authorize(authorization)
    if not prompt.strip() or len(prompt)>1200:raise HTTPException(422,'Invalid prompt')
    decoded=[]
    for upload in (image,mask):
        raw=await upload.read(MAX_BYTES+1)
        if len(raw)>MAX_BYTES:raise HTTPException(413,'Image too large')
        try:
            im=Image.open(io.BytesIO(raw))
            if im.width*im.height>4_000_000:raise HTTPException(413,'Too many pixels')
            decoded.append(im.convert('L' if upload is mask else 'RGB'))
        except (UnidentifiedImageError,OSError,ValueError) as error:raise HTTPException(400,'Invalid image') from error
    if decoded[0].size!=decoded[1].size:raise HTTPException(422,'Mask dimensions differ')
    output=await run_in_threadpool(generate,*decoded,prompt.strip())
    return Response(output,media_type='image/png',headers={'Cache-Control':'no-store'})
