#!/usr/bin/env python3
"""
Instax-style polaroid cards from stage stills, and the collage outro of an edit (video/make_tiktok.py).

A card (make_card) is the still cropped to the Instax Mini picture window (46 x 62 mm on a 54 x 86 mm card), given a faded
film grade (lifted blacks, cool shadows, warm highlights, soft, grain, vignette), framed in off-white paper with the wide
bottom strip, a short handwritten caption on that strip (Caveat, video/fonts/; a trailing "♡" is drawn by hand: the font
has no heart), a small tilt and a soft drop shadow -- one RGBA image, plus a "blank" twin whose picture is still
undeveloped grey for the developing effect. Tilt and caption placement are seeded by the stage id, so a re-render matches.

The collage (render_collage) piles the cards up over a blurred, darkened copy of a still: each card drops in on its frame
(lands slightly too big, settles, a small shake of the whole frame), then develops from grey over ~0.9 s, while the
whole frame pushes in slowly. Frames are composed with Pillow and piped to ffmpeg, so the segment is exactly `frames` long.

Try it alone (writes the cards next to the output for a look):
    python3 video/polaroid.py data/pipelines/<run>/hires 1 2 5 8 --out /tmp/collage.mp4
"""
import argparse
import math
import random
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = Path(__file__).resolve().parent / "fonts" / "Caveat.ttf"
MM = 20  # card pixels per mm at render size (the collage scales cards down from here)
CARD_MM, PHOTO_MM = (54, 86), (46, 62)
BORDER_MM = (4, 6)  # side, top; the bottom strip is the rest
PAPER = (242, 239, 231)
INK = (38, 40, 66)
UNDEVELOPED = (86, 90, 84)  # the grey-green of a fresh instax before the picture comes up
CAPTIONS = ["daddy's girl ♡", "good girl", "all yours", "his ♡", "morning fun", "xoxo", "bad girl", "best morning ♡"]
W, H = 1080, 1920
POST = ("vignette=angle=PI/10", "noise=alls=10:allf=t+u")  # the edit's own vignette + grain over the composed frames


def _fill_crop(img: Image.Image, size, y_focus=0.42) -> Image.Image:
    """Scale to cover size and crop; vertically centred on y_focus (faces sit in the upper half of a portrait still)."""
    tw, th = size
    s = max(tw / img.width, th / img.height)
    img = img.resize((max(tw, round(img.width * s)), max(th, round(img.height * s))), Image.LANCZOS)
    x = (img.width - tw) // 2
    y = min(max(round(img.height * y_focus - th / 2), 0), img.height - th)
    return img.crop((x, y, x + tw, y + th))


def film_grade(img: Image.Image, seed=0) -> Image.Image:
    """Faded instant-film look: softened, a little desaturated, lifted blacks, cool shadows / warm highlights, vignette, grain."""
    a = np.asarray(img.filter(ImageFilter.GaussianBlur(0.9)).convert("RGB"), dtype=np.float32) / 255
    gray = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    a = gray[..., None] + (a - gray[..., None]) * 0.86
    a = 0.07 + a * 0.87
    a += (1 - a) * np.array([-0.012, 0.004, 0.03], dtype=np.float32)  # cool shadows
    a *= 1 + a * np.array([0.035, 0.0, -0.05], dtype=np.float32)  # warm highlights
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    a *= (1 - 0.16 * r2 / 2)[..., None]
    a += np.random.default_rng(seed).normal(0, 0.022, (h, w, 1)).astype(np.float32)
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def _paper(size, seed) -> Image.Image:
    rng = np.random.default_rng(seed + 1)
    a = np.asarray(Image.new("RGB", size, PAPER), dtype=np.float32) + rng.normal(0, 2.6, (size[1], size[0], 1))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))


def _heart(draw, x, y, s, width):
    """A hand-drawn heart outline, s px wide, its top-left at (x, y)."""
    pts = [(16 * math.sin(t) ** 3, 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
           for t in np.linspace(0, 2 * math.pi, 60)]
    pts = [(x + s / 2 + px * s / 34, y + s / 2 - py * s / 34) for px, py in pts]
    draw.line(pts + pts[:1], fill=INK, width=width, joint="curve")


def _caption(text: str, strip_w, strip_h, rng) -> Image.Image:
    """The caption in ink on a transparent layer the size of the bottom strip, slightly tilted and off-centre."""
    heart = text.endswith("♡")
    text = text.rstrip("♡ ").strip()
    size = round(strip_h * 0.44)
    font = ImageFont.truetype(str(FONT), size)
    font.set_variation_by_axes([620])
    tw = font.getlength(text)
    hs = round(size * 0.55) if heart else 0
    layer = Image.new("RGBA", (strip_w * 2, strip_h * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x0 = layer.width / 2 - (tw + (hs * 1.3 if heart else 0)) / 2
    y0 = layer.height / 2 - size * 0.62
    d.text((x0, y0), text, font=font, fill=INK)
    if heart:
        _heart(d, x0 + tw + hs * 0.35, y0 + size * 0.38, hs, max(3, size // 22))
    layer = layer.rotate(rng.uniform(-4, 3), resample=Image.BICUBIC)
    dx, dy = rng.uniform(-0.06, 0.06) * strip_w, rng.uniform(-0.05, 0.08) * strip_h
    return layer.crop((round(strip_w / 2 - dx), round(strip_h / 2 - dy), round(strip_w * 1.5 - dx), round(strip_h * 1.5 - dy)))


@lru_cache(maxsize=64)  # an edit shows a still's card in several cuts, and the collage reuses them
def make_card(still: Path, caption: str | None, seed: int) -> tuple:
    """(developed, blank): RGBA images of the tilted card with its drop shadow, same size and placement."""
    rng = random.Random(seed)
    cw, ch = CARD_MM[0] * MM, CARD_MM[1] * MM
    pw, ph = PHOTO_MM[0] * MM, PHOTO_MM[1] * MM
    bx, by = BORDER_MM[0] * MM, BORDER_MM[1] * MM
    photo = film_grade(_fill_crop(Image.open(still).convert("RGB"), (pw, ph)), seed)
    paper = _paper((cw, ch), seed)
    frame = ImageDraw.Draw(paper)
    frame.rectangle((bx - 2, by - 2, bx + pw + 1, by + ph + 1), outline=(214, 210, 200), width=2)  # the window's recessed edge
    if caption:
        strip_top = by + ph
        cap = _caption(caption, cw, ch - strip_top, rng)
        paper.paste(cap, (0, strip_top), cap)
    angle = rng.uniform(3, 8) * rng.choice((-1, 1))
    out = []
    for pic in (photo, Image.new("RGB", (pw, ph), UNDEVELOPED)):
        card = paper.copy()
        card.paste(pic, (bx, by))
        out.append(_tilt_and_shadow(card.convert("RGBA"), angle))
    return tuple(out)


def _tilt_and_shadow(card: Image.Image, angle) -> Image.Image:
    pad = 90
    big = Image.new("RGBA", (card.width + 2 * pad, card.height + 2 * pad), (0, 0, 0, 0))
    big.paste(card, (pad, pad))
    big = big.rotate(angle, resample=Image.BICUBIC, expand=True)
    alpha = big.getchannel("A")
    shadow = Image.new("RGBA", big.size, (0, 0, 0, 0))
    shadow.putalpha(alpha.point(lambda v: v * 0.5).filter(ImageFilter.GaussianBlur(26)))
    canvas = Image.new("RGBA", big.size, (0, 0, 0, 0))
    canvas.alpha_composite(shadow, (12, 26))  # light from the upper left
    canvas.alpha_composite(big)
    return canvas


def blurred_background(still: Path) -> Image.Image:
    """The collage backdrop: the still filling the frame, heavily blurred, darkened and a little desaturated."""
    img = _fill_crop(Image.open(still).convert("RGB"), (W // 4, H // 4), 0.5).filter(ImageFilter.GaussianBlur(9))
    a = np.asarray(img.resize((W, H), Image.BICUBIC), dtype=np.float32)
    gray = a.mean(axis=2, keepdims=True)
    a = (gray + (a - gray) * 0.75) * 0.42
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")


def auto_size(n):
    """The default card width (px of the 1080 frame) for n cards: big, overlapping a lot; one card alone 900."""
    return max(700, min(900, 1000 - 50 * n))


def _layout(n, seed, w=None):
    """(centre x, centre y, card width) per card: a loose pile spiralling in, the last card on top near the middle. The pile's
    spread grows with the card size, so small cards still show more of each other and big ones stay in frame."""
    rng = random.Random(seed)
    w = w or auto_size(n)
    spread = 170 * min(max(w, 400), 1000) / 760
    spots = []
    for k in range(n):
        r = 0.0 if n == 1 else math.sqrt((n - 1 - k) / (n - 1))
        a = k * 2.4 + rng.uniform(-0.3, 0.3)
        spots.append((W / 2 + r * spread * math.cos(a), H / 2 - 30 + r * spread * 1.75 * math.sin(a), w))
    return spots


LAND = 8  # frames a card takes to drop in and settle
SHAKE = 6  # frames the whole frame shakes after a landing


def _land_scale(age):
    """A card's scale age frames after it starts dropping: 1.35 -> 0.97 (overshoot) -> 1."""
    if age >= LAND:
        return 1.0
    t = age / LAND
    return 1.35 - 0.38 * min(t / 0.7, 1) ** 2 + (0.03 * (t - 0.7) / 0.3 if t > 0.7 else 0)


def render_collage(cards, background: Image.Image, frames: int, fps: float, land_at: list, out_path: Path,
                   push=0.06, develop_s=0.9, seed=0, post_filters=(), card_w=None):
    """Write the collage segment: `frames` frames at fps, card k starting to drop at frame land_at[k] (negative = already
    mid-drop at the first frame); card_w = card width in px of the 1080 frame (None = auto_size); develop_s 0 = no developing."""
    spots = _layout(len(cards), seed, card_w)
    sized = []
    for (dev, blank), (_, _, w) in zip(cards, spots):
        s = w / (CARD_MM[0] * MM)
        size = (round(dev.width * s), round(dev.height * s))
        sized.append((dev.resize(size, Image.LANCZOS), blank.resize(size, Image.LANCZOS)))
    develop = round(develop_s * fps)
    settled, base = 0, background.copy()  # base = backdrop + every card that has landed and developed (they stop changing)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           *(["-vf", ",".join(post_filters)] if post_filters else []),
           "-frames:v", str(frames), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
           str(out_path), "-loglevel", "error"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in range(frames):
            while settled < len(cards) and f - land_at[settled] >= LAND + develop:
                _paste(base, sized[settled][0], spots[settled], 1.0)
                settled += 1
            frame = base.copy()
            shake = 0.0
            for k in range(settled, len(cards)):
                age = f - land_at[k]
                if age < 0:
                    break
                dev, blank = sized[k]
                d = min(max((age - LAND / 2) / develop, 0), 1) if develop else 1.0
                d = d * d * (3 - 2 * d)
                img = dev if d >= 1 else blank if d <= 0 else Image.blend(blank, dev, d)
                _paste(frame, img, spots[k], _land_scale(age), alpha=min(1.0, (age + 1) / 3))
                if LAND * 0.6 <= age < LAND * 0.6 + SHAKE:
                    shake = max(shake, 1 - (age - LAND * 0.6) / SHAKE)
            proc.stdin.write(_camera(frame, 1 + push * f / max(frames - 1, 1), shake, f).tobytes())
    finally:
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError(f"ffmpeg failed writing {out_path}")
    return out_path


def _paste(canvas, img, spot, scale, alpha=1.0):
    cx, cy, _ = spot
    if abs(scale - 1) > 1e-3:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
    if alpha < 1:
        img = img.copy()
        img.putalpha(img.getchannel("A").point(lambda v: v * alpha))
    canvas.alpha_composite(img, (round(cx - img.width / 2), round(cy - img.height / 2)))


def _camera(frame: Image.Image, zoom, shake, f) -> Image.Image:
    """The slow push-in (crop the centre 1/zoom and scale back up), offset by a decaying shake after a landing."""
    cw, ch = W / zoom, H / zoom
    dx = shake * 9 * math.sin(f * 2.1)
    dy = shake * 12 * math.cos(f * 2.7)
    x0 = min(max((W - cw) / 2 + dx, 0), W - cw)
    y0 = min(max((H - ch) / 2 + dy, 0), H - ch)
    return frame.convert("RGB").resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))


def caption_for(stage: int, captions: dict | None = None) -> str:
    """The stage's caption: from captions (stage id -> text; "" = none), else one from CAPTIONS picked by the id."""
    if captions and stage in captions:
        return captions[stage]
    return CAPTIONS[stage % len(CAPTIONS)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stills_dir", type=Path, help="folder with stage_N.png")
    ap.add_argument("stages", type=int, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--every", type=float, default=0.9, help="seconds between landings")
    ap.add_argument("--hold", type=float, default=2.5, help="seconds after the last landing")
    ap.add_argument("--card-size", type=int, default=None, help="card width in px of the 1080 frame (default: by count)")
    a = ap.parse_args()
    cards = []
    for n in a.stages:
        dev, blank = make_card(a.stills_dir / f"stage_{n}.png", caption_for(n), seed=n)
        dev.save(a.out.with_name(f"{a.out.stem}_card_{n}.png"))
        cards.append((dev, blank))
    land = [round(k * a.every * a.fps) for k in range(len(cards))]
    frames = land[-1] + round(a.hold * a.fps)
    bg = blurred_background(a.stills_dir / f"stage_{a.stages[-1]}.png")
    render_collage(cards, bg, frames, a.fps, land, a.out, post_filters=POST, card_w=a.card_size)
    print(f"{a.out} ({frames} frames)")


if __name__ == "__main__":
    main()
