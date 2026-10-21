# 图片素材记录

## 客厅示例图

- 文件：`static/demo.webp`
- 用途：首页封面与“用示例图体验”
- 工具：内置 imagegen
- 生成类型：photorealistic-natural
- 最终提示词：

> Photorealistic editorial interior photograph for a general object editing website. A bright modern living room with a clearly visible rust-orange lounge chair centered in the foreground, cobalt blue vase on a side table, cream floor lamp and potted plant. Clear separate silhouettes, natural contact shadows, light oak floor, warm window daylight, wide landscape framing. No people, no text, no logos, no UI elements, no watermarks.

项目内图片为压缩后的 WebP，用于网页快速加载。源 PNG 留在本机的 imagegen 生成记录中。

## 首页循环功能演示

- `static/assets/hero-chair.webp`：通过本产品的 SAM 框选接口提取完整椅子，原始框选范围为 `(230,165)-(572,414)`。
- `static/assets/hero-background.webp`：内置 imagegen 修整的示意背景，用于首页预制动画；不是现场调用 LaMa 的效果保证。
- `static/assets/hero-scene.json`：椅子实际图层坐标。
- `static/assets/hero-demo.js`：Canvas 循环演示点选、移动与分别调整宽高；支持暂停、离屏暂停和减少动态效果设置。

背景图最终提示词：

> Use case: precise-object-edit. Edit target: the attached living room photograph. Remove ONLY the entire rust-orange armchair in the center AND its cast/contact shadow. Reconstruct the beige woven rug, floor, cream wall, window background and any hidden area naturally, with continuous rug fibers, straight architectural lines, consistent window daylight. The chair's old footprint must be completely absent: no chair-shaped ghost, no dark outline, no blur, no flat painted patch. Preserve the exact original camera, crop, image aspect ratio, room geometry, sofa with cushions and blanket on the left, left coffee table and bowl, standing floor lamp, blue vase and pedestal table on the right, trees, window frame and colors. Do not move any other object. Photorealistic, no text, no arrows, no watermarks. This is a clean background plate for placing the ORIGINAL chair back on top in an animation; accurate alignment to the original photo is essential.
