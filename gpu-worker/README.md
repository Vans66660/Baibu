# Optional GPU background repair worker

Status: deployment source prepared; CUDA model inference has NOT been run or benchmarked on the development Mac. The local reference-texture repair is independently usable. This worker does not yet implement ControlNet, background semantic segmentation or automatic boundary completion. SDXL image quality must be evaluated on real user images before enabling it publicly.

## Deployment

Use a Linux NVIDIA CUDA server with NVIDIA Container Toolkit and a recent Docker Compose supporting `gpus: all`. Start with at least 16 GB system RAM and a GPU with approximately 16 GB VRAM; CPU offload is enabled, but actual memory and speed need measurement on the target machine. First startup downloads the SDXL inpainting model. Inspect its OpenRAIL++ license and usage restrictions for your deployment.

1. Add `BACKGROUND_WORKER_TOKEN` to your server's `.env`. Generate a random secret (at least 24 characters); never commit it. Keep it identical in the app and worker.
2. Run `docker compose -f compose.yaml -f compose.gpu.yaml up -d --build`.
3. The worker starts only when CUDA is available. The model must finish loading before it serves requests. Inspect its logs if needed; do not expose port 8010 to the Internet.
4. After connecting, the website's “GPU 生成修复” option becomes available. Capability availability means the server configuration exists, not a guarantee that the worker is healthy. Failed requests leave the original background intact.

For separate machines configure `BACKGROUND_WORKER_URL` and `BACKGROUND_WORKER_TOKEN` in the app's process environment. Use HTTPS across an untrusted network; use a private network or authenticated tunnel. The client never receives the token. The app sends background pixels, the repair mask and a translated background instruction to your worker. The chosen reference texture is quilted into the masked area first. The worker receives this prefilled image, not a separate reference-image conditioning channel.

Generation uses an aspect-preserving context crop, mask and prompt. Only pixels inside the painted mask are adopted; all exterior pixels are copied exactly from the current background. The result is a preview until the visitor confirms. Local reference mode does not transmit images to this worker.

## Sources

- https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1
- https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint
