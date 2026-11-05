# Third-party model sources

## MobileSAM

- Project: https://github.com/ChaoningZhang/MobileSAM
- Source revision: `f706ad9c4eb7f219c00d9050e46328518ffb65d2`
- License: Apache-2.0; copy in `licenses/MobileSAM-Apache-2.0.txt`.
- Model: `vit_t`, official `weights/mobile_sam.pt` at the same revision.
- Checkpoint SHA-256: `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`.
- Code is installed from a pinned archive; weights are downloaded to the PyTorch model cache on first use, with hash verification on download.

## Other models

The original SAM option uses `facebook/sam-vit-base`; object detection uses `IDEA-Research/grounding-dino-tiny`; background inpainting uses the Simple LaMa package. Chinese name translation uses Helsinki-NLP/opus-mt-zh-en (CC-BY-4.0), developed by the Helsinki-NLP group: https://huggingface.co/Helsinki-NLP/opus-mt-zh-en . See README for their roles and ASSETS.md for the demonstration image sources.

## Optional SDXL background worker

The optional worker references `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (OpenRAIL++ model license). Model weights are not included in this project and have not been downloaded for the development Mac. Inspect the model license and restrictions before deploying:
https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1

Diffusers (Apache-2.0) and Accelerate (Apache-2.0) are installed only in the optional worker image. See `gpu-worker/requirements.txt` and the upstream repositories for full licenses.
