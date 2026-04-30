"""Bake a procedural Mars regolith texture to PNG.

The MuJoCo built-in `flat` and `checker` textures look too plastic at
close range. This script generates a 2048×2048 PNG with realistic
regolith characteristics:

  - reddish-brown base (Mars regolith albedo color)
  - large-scale color variation (light dust patches vs darker basalt)
  - scattered small dark pebbles
  - sand-grain micro noise

The output is saved to `src/mars/assets/regolith.png` and referenced from
`src/mars/sim.py`'s MJCF asset block.

Run once:
    python scripts/bake_regolith_texture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from mars.constants import MARS
from mars.terrain import _fbm, _value_noise_2d  # noqa: E402

OUT_PATH = REPO / "src" / "mars" / "assets" / "regolith.png"
NORMAL_PATH = REPO / "src" / "mars" / "assets" / "regolith_normal.png"


def generate_regolith(size: int = 2048, seed: int = 0) -> np.ndarray:
    """Pure-sand Mars regolith texture — no pebbles, no rocks, no debris.

    Just smooth Mars dust with subtle color drift and fine grain noise.
    Tiles cleanly because there's no large recognizable feature to repeat.
    """
    rng = np.random.default_rng(seed)
    base = np.array(MARS.regolith_color_rgb, dtype=np.float32)

    # ── Base color ─────────────────────────────────────────────────────
    img = np.broadcast_to(base, (size, size, 3)).copy()

    # ── Subtle large-scale color drift (lighter/darker dust regions) ───
    drift = _fbm(rng, size, octaves=4, persistence=0.5, base_scale=size // 4)
    drift = (drift - drift.mean()) / (drift.std() + 1e-8)
    img = img * (1.0 + 0.10 * np.clip(drift, -1.5, 1.5)[..., None])

    # ── Mid-scale variation (cm-to-dm patches) ─────────────────────────
    mid = _fbm(rng, size, octaves=3, persistence=0.5, base_scale=size // 32)
    mid = (mid - mid.mean()) / (mid.std() + 1e-8)
    img = img * (1.0 + 0.05 * mid[..., None])

    # ── Sand grain micro noise (sub-mm tooth) ──────────────────────────
    # Two octaves of speckle: very fine + slightly larger grain clusters.
    grain_fine = rng.standard_normal((size, size, 1)).astype(np.float32) * 0.020
    grain_coarse = _fbm(rng, size, octaves=2, persistence=0.3, base_scale=2)
    grain_coarse = (grain_coarse - grain_coarse.mean()) / (grain_coarse.std() + 1e-8)
    img = img + grain_fine + 0.012 * grain_coarse[..., None]

    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def height_to_normal(height: np.ndarray, strength: float = 4.0) -> np.ndarray:
    """Convert a single-channel height array to an RGB normal map.

    Standard tangent-space normal encoding: each pixel's RGB stores the
    surface normal vector where (0.5, 0.5, 1.0) means "flat". Higher
    `strength` exaggerates bumps. The light-catching effect of the bumps
    is what makes the surface read as 3D regolith vs. flat plastic.
    """
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


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"baking 2048×2048 regolith texture → {OUT_PATH.relative_to(REPO)}")
    arr = generate_regolith(size=2048, seed=7)
    Image.fromarray(arr).save(OUT_PATH, optimize=True)
    print(f"✓ wrote {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")

    # Derive a normal map from the regolith's luminance. Pebbles and dust
    # patches produce micro-relief that catches the light, killing the
    # "flat plastic" look of the unbumped material.
    print(f"baking matching normal map → {NORMAL_PATH.relative_to(REPO)}")
    from scipy.ndimage import gaussian_filter
    lum = (0.30 * arr[..., 0] + 0.59 * arr[..., 1] + 0.11 * arr[..., 2]).astype(np.float32) / 255.0
    lum_smooth = gaussian_filter(lum, sigma=1.2)
    normal = height_to_normal(lum_smooth, strength=5.0)
    Image.fromarray(normal).save(NORMAL_PATH, optimize=True)
    print(f"✓ wrote {NORMAL_PATH}  ({NORMAL_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
