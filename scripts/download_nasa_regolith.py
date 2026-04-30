"""Download a real NASA Mars rover photo and use it as the regolith texture.

NASA imagery is public domain. We download a Curiosity / Perseverance
view of Mars regolith, center-crop it square, resize to 2048², and
seam-blend it so it tiles. The result overwrites the procedural
regolith.png used by the Three.js viewer (and the MuJoCo viewer).

Run:
    python3 scripts/download_nasa_regolith.py
    # or with a specific NASA Photojournal ID:
    python3 scripts/download_nasa_regolith.py --nasa-id PIA22871
    # or your own URL (e.g. a photo you found you like):
    python3 scripts/download_nasa_regolith.py --url https://...

After running, just reload the viewer in your browser — the texture will
hot-load.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
RAW_DIR = REPO / "data" / "raw" / "nasa"
OUT_PATH = REPO / "src" / "mars" / "assets" / "regolith.png"
NORMAL_PATH = REPO / "src" / "mars" / "assets" / "regolith_normal.png"

# Public-domain Mars rover photos hosted on NASA's images.nasa.gov CDN.
# Tried in order — the first that downloads wins. All are top-down or
# oblique views of regolith without rover hardware in frame.
NASA_CANDIDATES: list[tuple[str, str]] = [
    ("PIA17944", "https://images-assets.nasa.gov/image/PIA17944/PIA17944~orig.jpg"),
    ("PIA22871", "https://images-assets.nasa.gov/image/PIA22871/PIA22871~orig.jpg"),
    ("PIA15282", "https://images-assets.nasa.gov/image/PIA15282/PIA15282~orig.jpg"),
    ("PIA22228", "https://images-assets.nasa.gov/image/PIA22228/PIA22228~orig.jpg"),
    # Smaller "large" sizes as fallbacks if `~orig` isn't available
    ("PIA17944", "https://images-assets.nasa.gov/image/PIA17944/PIA17944~large.jpg"),
    ("PIA22871", "https://images-assets.nasa.gov/image/PIA22871/PIA22871~large.jpg"),
]


def fetch(url: str) -> bytes:
    print(f"  trying {url}")
    r = requests.get(url, timeout=60, headers={"User-Agent": "MarsWorldModel/0.1"})
    r.raise_for_status()
    return r.content


def fetch_with_fallback(candidates: list[tuple[str, str]]) -> tuple[str, bytes]:
    last_err = None
    for nid, url in candidates:
        try:
            data = fetch(url)
            if len(data) < 50_000:
                raise ValueError(f"response too small ({len(data)} bytes) — likely an error page")
            return nid, data
        except Exception as e:
            print(f"    failed: {e}")
            last_err = e
    raise RuntimeError(f"every NASA URL failed; last error: {last_err}")


def make_seamless(img: Image.Image, blend_frac: float = 0.12) -> Image.Image:
    """Make a square photo tile seamlessly with the offset-and-blend trick.

    Roll the image by half along both axes (so the original outer seams now
    sit on the inner cross), then cosine-fade between rolled and original
    along those inner seams. Mars regolith is homogeneous enough that the
    result is invisible at 40× tile density.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    rolled = np.roll(arr, (h // 2, w // 2), axis=(0, 1))
    bw = max(1, int(w * blend_frac))
    bh = max(1, int(h * blend_frac))

    cx = w // 2
    for i in range(-bw, bw):
        a = 0.5 * (1.0 + np.cos(np.pi * i / bw))
        col = cx + i
        rolled[:, col] = rolled[:, col] * (1 - a) + arr[:, col] * a

    cy = h // 2
    for i in range(-bh, bh):
        a = 0.5 * (1.0 + np.cos(np.pi * i / bh))
        row = cy + i
        rolled[row, :] = rolled[row, :] * (1 - a) + arr[row, :] * a

    return Image.fromarray(np.clip(rolled, 0, 255).astype(np.uint8))


def height_to_normal(height: np.ndarray, strength: float = 4.0) -> np.ndarray:
    """Convert a luminance heightmap to a tangent-space normal map PNG."""
    gy, gx = np.gradient(height.astype(np.float32))
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(height, dtype=np.float32)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx /= norm; ny /= norm; nz /= norm
    r = ((nx + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    g = ((ny + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    b = ((nz + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def process(image_bytes: bytes, *, size: int = 2048,
            seamless: bool = True, crop: str = "bottom") -> Image.Image:
    """Center-crop or bottom-crop to square, resize, optionally seam-blend.

    Most NASA rover photos are panoramas with sky at the top and close-range
    regolith at the bottom. Default to bottom-crop so we grab the part of
    the photo that actually looks like ground texture instead of horizon.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    print(f"  original size: {w}×{h}, crop={crop}")
    # For landscape rover panoramas, the bottom-center holds close-range
    # terrain (regolith) and the top has sky + distant horizon. We crop
    # a square smaller than the full height so we have room to position
    # it within the photo.
    side = int(min(w, h) * 0.55)
    if crop == "center":
        left = (w - side) // 2
        top = (h - side) // 2
    elif crop == "bottom":
        left = (w - side) // 2
        top = h - side - max(20, h // 50)   # tiny margin off the very bottom
    elif crop == "top":
        left = (w - side) // 2
        top = max(20, h // 50)
    else:
        raise ValueError(f"unknown crop: {crop}")
    print(f"  crop window: ({left},{top}) size {side}×{side}")
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    if seamless:
        img = make_seamless(img)
    return img


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", help="Use this URL instead of the NASA candidates.")
    p.add_argument("--nasa-id", help="Specific NASA Photojournal ID (e.g. PIA22871).")
    p.add_argument("--no-seamless", action="store_true",
                   help="Skip seam-blending. Use if your photo already tiles.")
    p.add_argument("--no-normal", action="store_true",
                   help="Skip regenerating regolith_normal.png from the new photo.")
    p.add_argument("--crop", choices=["center", "bottom", "top"], default="bottom",
                   help="Which part of the photo to crop. 'bottom' grabs the "
                        "close-range regolith from rover panoramas (default).")
    p.add_argument("--size", type=int, default=2048)
    args = p.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.url:
        candidates = [("custom", args.url)]
    elif args.nasa_id:
        nid = args.nasa_id.upper()
        candidates = [
            (nid, f"https://images-assets.nasa.gov/image/{nid}/{nid}~orig.jpg"),
            (nid, f"https://images-assets.nasa.gov/image/{nid}/{nid}~large.jpg"),
        ]
    else:
        candidates = NASA_CANDIDATES

    nid, data = fetch_with_fallback(candidates)
    print(f"✓ downloaded {nid}  ({len(data) / 1024:.0f} KB)")

    raw_path = RAW_DIR / f"{nid}.jpg"
    raw_path.write_bytes(data)
    print(f"  raw saved → {raw_path}")

    img = process(data, size=args.size,
                  seamless=not args.no_seamless, crop=args.crop)
    img.save(OUT_PATH, optimize=True)
    print(f"✓ regolith.png updated  ({OUT_PATH.stat().st_size // 1024} KB)")

    if not args.no_normal:
        # Re-derive a matching normal map from the new photo's luminance
        from scipy.ndimage import gaussian_filter
        arr = np.asarray(img, dtype=np.float32)
        lum = (0.30 * arr[..., 0] + 0.59 * arr[..., 1] + 0.11 * arr[..., 2]) / 255.0
        normal = height_to_normal(gaussian_filter(lum, sigma=1.2), strength=5.0)
        Image.fromarray(normal).save(NORMAL_PATH, optimize=True)
        print(f"✓ regolith_normal.png updated  ({NORMAL_PATH.stat().st_size // 1024} KB)")

    print()
    print("Reload the browser tab. The new texture is loaded automatically.")


if __name__ == "__main__":
    main()
