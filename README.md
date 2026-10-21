# 摆布 · 公开试用版 0.1

访客上传图片后，可点选、框选或用英文名称识别物品，拖动位置、独立调整宽高、模拟盒状物品的侧面厚度、手动修补原位置并导出 PNG。首页、编辑器和 API 均由同一个 FastAPI 服务提供。当前没有账号、收费、图库或多用户任务队列。

## 本地预览

需要 Python 3.11。首次处理图片会下载 SAM、LaMa 和按需使用的 Grounding DINO 权重。

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
MODEL_DEVICE=cpu uvicorn app:app --host 127.0.0.1 --port 8001
```

打开 `http://127.0.0.1:8001/`。如果已安装 Docker，也可运行 `docker compose -f compose.preview.yaml up --build`，同样访问 8001 端口。直接双击 HTML 文件无法调用识别 API。
macOS 本地预览建议使用 CPU；`MODEL_DEVICE=auto` 可自动选择 CUDA、MPS 或 CPU，也可显式指定设备。

## 公网部署

准备一台可运行 Docker 的 Linux 服务器和指向服务器的域名。建议至少 16 GB 内存；CPU 处理长图可能需要数分钟。对外试用需要更快的处理速度时，使用 NVIDIA GPU 服务器并配置对应的容器运行环境。

1. 将整个目录复制到服务器。
2. 复制 `.env.example` 为 `.env`，将 `DOMAIN` 改成自己的域名；按需要调整 `HOURLY_LIMIT`。
3. 让域名 A/AAAA 记录指向服务器，并开放 80、443 端口。
4. 运行 `docker compose up -d --build`。Caddy 将为域名申请 HTTPS 证书，应用本身只暴露在内部容器网络。
5. 浏览器打开 `https://你的域名/`，先用示例图走完选物、移动、修补和导出流程。

容器配置尚未在实际云服务器上验证。部署前请在目标环境确认模型下载、可用内存和推理时长。

## 公开试用的边界

- 默认每个来源 IP 每小时最多 12 次处理请求，包括首次抠图和再次修补。这个限制保存在单进程内存中，重启会清零；多人正式运营应换成共享限额存储与任务队列。
- 服务一次处理一张图片；繁忙时返回 429。上传文件上限 12 MB、图片上限 400 万像素。
- 服务不创建用户图片库。上传数据在请求过程中处理，浏览器负责保存导出的文件。反向代理日志不应记录请求体。
- 单张照片无法还原被遮挡的真实背景。复杂纹理、文字、人物和长阴影可能仍需手动修补。宽高与厚度编辑属于二维视觉调整。
- 公开发布前需确定域名、运营主体、对外隐私说明和支持渠道；当前产品名称为“摆布”。

## 技术组成

- Grounding DINO Tiny：按英文名称定位目标。
- SAM ViT Base：根据点或框生成物品蒙版。
- LaMa：补全遮挡区域；平滑背景的手动修补可改用邻近像素填补。
- Canvas：在浏览器中拖动、调整宽高与导出。
- FastAPI + Caddy：推理 API、页面服务与 HTTPS 入口。

`GET /health` 返回服务状态。`POST /api/process` 接收 multipart `file` 和 `point`、`box` 或 `prompt`；`POST /api/repair` 接收背景图片和涂抹蒙版。设置 `PUBLIC_MODE=true` 会开启限额并关闭交互式 API 文档。

## 首页功能演示

首页以预制背景和透明椅子图层演示点选、移动、独立调整宽高，约 14 秒循环一次。访客无需等待模型处理；支持暂停与继续，并会在离屏时暂停。系统开启“减少动态效果”时默认显示静止画面。演示背景经过单独修整，素材来源与提示词记录在 `ASSETS.md`。


## 在局域网访问

运行 `MODEL_DEVICE=cpu uvicorn app:app --host 0.0.0.0 --port 8001`，其他设备使用 `http://<服务器的局域网IP>:8001/` 访问。`127.0.0.1` 只指向访问者自己的设备，不能作为分享地址。
