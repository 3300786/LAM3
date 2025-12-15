# perturb_image.py
import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class Cfg:
    # 颜色类扰动幅度（越接近1越轻）
    brightness: float = 0.12   # 亮度: factor in [1-b, 1+b]
    contrast: float = 0.12     # 对比度
    saturation: float = 0.12   # 饱和度
    hue_deg: float = 6.0       # 色相偏移（度），通常 <= 10 比较温和

    # 噪声/模糊
    gaussian_noise_std: float = 6.0   # 像素噪声标准差（0-255空间），建议 2~10
    blur_radius: float = 0.6          # 高斯模糊半径，建议 0~1

    # 轻微几何扰动（像素/比例）
    translate_px: int = 6             # 平移像素
    rotate_deg: float = 2.0           # 旋转角度
    scale_jitter: float = 0.015       # 缩放抖动比例
    perspective: float = 0.03         # 透视扰动强度（0~0.05 比较轻）

    # JPEG 压缩（有些任务对其敏感，默认轻度）
    jpeg_prob: float = 0.5
    jpeg_quality_min: int = 75
    jpeg_quality_max: int = 95

    # 每次随机挑选哪些扰动
    prob_color: float = 0.95
    prob_noise: float = 0.80
    prob_blur: float = 0.35
    prob_geom: float = 0.60


def clamp_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def apply_color_jitter(img: Image.Image, cfg: Cfg) -> Image.Image:
    # brightness/contrast/saturation
    def rand_factor(mag: float) -> float:
        return 1.0 + random.uniform(-mag, mag)

    img = ImageEnhance.Brightness(img).enhance(rand_factor(cfg.brightness))
    img = ImageEnhance.Contrast(img).enhance(rand_factor(cfg.contrast))
    img = ImageEnhance.Color(img).enhance(rand_factor(cfg.saturation))

    # hue shift (via HSV)
    if cfg.hue_deg > 0:
        hue_shift = random.uniform(-cfg.hue_deg, cfg.hue_deg) / 360.0  # [0,1) space
        hsv = img.convert("HSV")
        h, s, v = hsv.split()
        h_np = (np.array(h, dtype=np.float32) / 255.0)
        h_np = (h_np + hue_shift) % 1.0
        h = Image.fromarray((h_np * 255.0).astype(np.uint8), mode="L")
        img = Image.merge("HSV", (h, s, v)).convert("RGB")

    return img


def apply_gaussian_noise(img: Image.Image, cfg: Cfg) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    std = random.uniform(0.0, cfg.gaussian_noise_std)
    noise = np.random.normal(0.0, std, size=arr.shape).astype(np.float32)
    arr = arr + noise
    return Image.fromarray(clamp_u8(arr), mode="RGB")


def apply_blur(img: Image.Image, cfg: Cfg) -> Image.Image:
    r = random.uniform(0.0, cfg.blur_radius)
    if r <= 1e-6:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=r))


def _perspective_coeffs(src_pts, dst_pts):
    # Solve for perspective transform coefficients.
    # Adapted from common PIL perspective transform derivations.
    matrix = []
    for (x, y), (u, v) in zip(dst_pts, src_pts):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(src_pts).reshape(8)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return res.tolist()


def apply_geometry(img: Image.Image, cfg: Cfg) -> Image.Image:
    w, h = img.size

    # mild affine: rotation + translation + scale
    angle = random.uniform(-cfg.rotate_deg, cfg.rotate_deg)
    scale = 1.0 + random.uniform(-cfg.scale_jitter, cfg.scale_jitter)
    tx = random.uniform(-cfg.translate_px, cfg.translate_px)
    ty = random.uniform(-cfg.translate_px, cfg.translate_px)

    # rotate/scale around center
    img2 = img.transform(
        (w, h),
        Image.AFFINE,
        (scale * np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), tx,
         np.sin(np.deg2rad(angle)), scale * np.cos(np.deg2rad(angle)), ty),
        resample=Image.BICUBIC,
        fillcolor=(255, 255, 255),
    )

    # mild perspective
    if cfg.perspective > 0:
        m = cfg.perspective
        dx = w * random.uniform(-m, m)
        dy = h * random.uniform(-m, m)
        src = [(0, 0), (w, 0), (w, h), (0, h)]
        dst = [
            (0 + random.uniform(-dx, dx), 0 + random.uniform(-dy, dy)),
            (w + random.uniform(-dx, dx), 0 + random.uniform(-dy, dy)),
            (w + random.uniform(-dx, dx), h + random.uniform(-dy, dy)),
            (0 + random.uniform(-dx, dx), h + random.uniform(-dy, dy)),
        ]
        coeffs = _perspective_coeffs(src_pts=src, dst_pts=dst)
        img2 = img2.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor=(255, 255, 255))

    return img2


def maybe_jpeg(img: Image.Image, cfg: Cfg) -> Image.Image:
    if random.random() > cfg.jpeg_prob:
        return img
    q = random.randint(cfg.jpeg_quality_min, cfg.jpeg_quality_max)
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=q, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def perturb(img: Image.Image, cfg: Cfg) -> Image.Image:
    img = img.convert("RGB")

    if random.random() < cfg.prob_geom:
        img = apply_geometry(img, cfg)

    if random.random() < cfg.prob_color:
        img = apply_color_jitter(img, cfg)

    if random.random() < cfg.prob_noise:
        img = apply_gaussian_noise(img, cfg)

    if random.random() < cfg.prob_blur:
        img = apply_blur(img, cfg)

    img = maybe_jpeg(img, cfg)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="input image path")
    ap.add_argument("--out_path", required=True, help="output image path")
    ap.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    img = Image.open(args.in_path)
    cfg = Cfg()
    out = perturb(img, cfg)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    out.save(args.out_path)
    print(f"Saved: {args.out_path}")


if __name__ == "__main__":
    main()
