# src/utils/render_text_image.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont


@dataclass
class RenderCfg:
    width: int = 768
    height: int = 768
    margin: int = 40
    line_spacing: float = 1.25
    font_size: int = 24
    header_size: int = 26
    font_path: Optional[str] = None
    bg: Tuple[int, int, int] = (255, 255, 255)
    fg: Tuple[int, int, int] = (0, 0, 0)


def _load_font(font_path: Optional[str], size: int) -> ImageFont.ImageFont:
    candidates = []
    if font_path:
        candidates.append(font_path)

    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    ]

    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue

    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> List[str]:
    lines: List[str] = []
    for raw in text.split("\n"):
        if raw.strip() == "":
            lines.append("")
            continue
        words = raw.split(" ")
        cur = ""
        for w in words:
            cand = w if cur == "" else cur + " " + w
            if draw.textlength(cand, font=font) <= max_w:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                # hard wrap long token
                if draw.textlength(w, font=font) <= max_w:
                    cur = w
                else:
                    tmp = ""
                    for ch in w:
                        cand2 = tmp + ch
                        if draw.textlength(cand2, font=font) <= max_w:
                            tmp = cand2
                        else:
                            if tmp:
                                lines.append(tmp)
                            tmp = ch
                    cur = tmp
        if cur:
            lines.append(cur)
    return lines


def render_panels(
    panels: List[Tuple[str, str]],
    cfg: RenderCfg,
    tag: Optional[str] = None,
) -> Image.Image:
    img = Image.new("RGB", (cfg.width, cfg.height), cfg.bg)
    draw = ImageDraw.Draw(img)

    font_h = _load_font(cfg.font_path, cfg.header_size)
    font_b = _load_font(cfg.font_path, cfg.font_size)

    x0 = cfg.margin
    y = cfg.margin
    max_w = cfg.width - 2 * cfg.margin

    if tag:
        draw.text((x0, y), tag, fill=cfg.fg, font=font_h)
        y += int(cfg.header_size * cfg.line_spacing) + 10

    for i, (title, body) in enumerate(panels):
        draw.text((x0, y), title, fill=cfg.fg, font=font_h)
        y += int(cfg.header_size * cfg.line_spacing) + 6

        for ln in _wrap(draw, body, font_b, max_w):
            draw.text((x0, y), ln, fill=cfg.fg, font=font_b)
            y += int(cfg.font_size * cfg.line_spacing)

        if i < len(panels) - 1:
            y += 12
            draw.line((x0, y, x0 + max_w, y), fill=cfg.fg, width=1)
            y += 12

        if y > cfg.height - cfg.margin:
            break

    return img


def save_image(img: Image.Image, path: str) -> None:
    img.save(path)
