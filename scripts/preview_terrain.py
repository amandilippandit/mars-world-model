"""Quick Mars terrain preview — produces a PNG without needing Blender.

Renders the heightmap with:
  - Mars regolith color ramp
  - Lambertian shading from a low-angle sun
  - A 'butterscotch atmosphere' fade with distance

Useful for sanity-checking the terrain pipeline before installing Blender
or going to Colab. Output is just an artistic preview, not a Cycles render.

    python scripts/preview_terrain.py --terrain data/processed/terrain.obj \
        --out data/renders/run01/preview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mars.constants import MARS
from mars.terrain import synthetic_mars_terrain


def load_obj_heightmap(obj_path: Path) -> np.ndarray:
    """Recover a heightmap from our OBJ writer's regular grid."""
    verts = []
    for line in Path(obj_path).read_text().splitlines():
        if line.startswith("v "):
            _, x, y, z = line.split()
            verts.append((float(x), float(y), float(z)))
    arr = np.array(verts)
    # Infer grid: rows = unique y count, cols = unique x count
    n_x = len(np.unique(arr[:, 0]))
    n_y = len(np.unique(arr[:, 1]))
    if n_x * n_y != len(arr):
        # fallback to square assumption
        n = int(np.sqrt(len(arr)))
        n_x = n_y = n
    z = arr[:, 2].reshape(n_y, n_x)
    return z


def shade(heightmap: np.ndarray, sun_az_deg: float = 135.0, sun_el_deg: float = 30.0) -> np.ndarray:
    """Lambertian shading. Returns float intensity in [0, 1]."""
    gy, gx = np.gradient(heightmap)
    nx = -gx
    ny = -gy
    nz = np.ones_like(heightmap)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / norm, ny / norm, nz / norm

    az = np.deg2rad(sun_az_deg)
    el = np.deg2rad(sun_el_deg)
    lx, ly, lz = np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)

    intensity = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    # Add a touch of ambient so shadows aren't black on Mars (dust scatters)
    intensity = 0.25 + 0.75 * intensity
    return intensity


def colorize(heightmap: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Apply Mars regolith color ramp modulated by Lambertian intensity."""
    # Two-stop color ramp: dark basaltic → lighter dust
    cmap = LinearSegmentedColormap.from_list(
        "mars_regolith",
        [(0.0, (0.32, 0.18, 0.10)), (1.0, MARS.regolith_color_rgb)],
    )
    h_norm = (heightmap - heightmap.min()) / (np.ptp(heightmap) + 1e-9)
    base = cmap(h_norm)[..., :3]                # (H, W, 3)
    shaded = base * intensity[..., None]
    # Atmospheric haze: blend toward sky color with vertical position
    rows = np.linspace(0, 1, heightmap.shape[0])[:, None, None]
    horizon = np.array(MARS.sky_color_horizon_rgb)
    haze = 0.15 * (1.0 - rows) ** 2
    shaded = shaded * (1 - haze) + horizon * haze
    return np.clip(shaded, 0.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--terrain", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("data/renders/run01/preview.png"))
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.terrain and args.terrain.exists():
        h = load_obj_heightmap(args.terrain)
        print(f"loaded heightmap from {args.terrain}: shape={h.shape}, range=[{h.min():.2f}, {h.max():.2f}]")
    else:
        h = synthetic_mars_terrain(size=args.size, seed=args.seed)
        print(f"generated synthetic heightmap: shape={h.shape}")

    intensity = shade(h, sun_az_deg=135.0, sun_el_deg=25.0)
    rgb = colorize(h, intensity)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=120)

    ax[0].imshow(rgb, origin="lower")
    ax[0].set_title("Mars terrain — shaded preview", fontsize=11)
    ax[0].axis("off")

    im = ax[1].imshow(h, cmap="terrain", origin="lower")
    ax[1].set_title(f"elevation (m)  range=[{h.min():.1f}, {h.max():.1f}]", fontsize=11)
    plt.colorbar(im, ax=ax[1], shrink=0.8)
    ax[1].axis("off")

    fig.suptitle(
        f"MarsWorldModel  |  g={MARS.gravity_m_s2} m/s²  |  P={MARS.surface_pressure_pa} Pa  "
        f"|  τ={MARS.typical_optical_depth_tau}",
        fontsize=10, color="#444",
    )
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"✓ wrote {args.out}")


if __name__ == "__main__":
    main()
