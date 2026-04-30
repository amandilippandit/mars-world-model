"""Mars terrain pipeline: NASA DEMs → heightmap mesh.

Two data paths are supported:

1. **HiRISE DTM** — high-resolution (~1 m/px) DEMs of specific Mars sites,
   distributed as IMG/LBL pairs by U Arizona. Real, location-accurate. Best
   for "rover-eye view" simulations of named sites (Jezero, Gale, etc.).

2. **Synthetic** — fractal-noise heightmap with Mars-like statistics. Useful
   for local development without bandwidth and for procedural variety.

The output of either path is a triangulated mesh saved as OBJ/PLY, ready for
both Blender (visual) and Genesis (physics) to consume.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm


# ── HiRISE DTM catalog ──────────────────────────────────────────────────
# A small curated catalog of public HiRISE DTMs. The full catalog is at
# https://www.uahirise.org/dtm/ — we hardcode a few canonical sites so the
# scaffold works out of the box.

@dataclass(frozen=True)
class HiriseSite:
    name: str
    description: str
    img_url: str        # the .IMG raster (DTEEC = stereo-derived elevation)
    lbl_url: str        # the PDS .LBL header describing the IMG


HIRISE_SITES: dict[str, HiriseSite] = {
    "jezero": HiriseSite(
        name="Jezero Crater",
        description="Perseverance / Mars 2020 landing site, ancient river delta.",
        img_url="https://www.uahirise.org/PDS/DTM/PSP/ORB_010500_010599/PSP_010573_1985_PSP_010639_1985/DTEEC_010573_1985_010639_1985_U01.IMG",
        lbl_url="https://www.uahirise.org/PDS/DTM/PSP/ORB_010500_010599/PSP_010573_1985_PSP_010639_1985/DTEEC_010573_1985_010639_1985_U01.LBL",
    ),
    "gale": HiriseSite(
        name="Gale Crater",
        description="Curiosity / MSL landing site, Mt. Sharp.",
        img_url="https://www.uahirise.org/PDS/DTM/ESP/ORB_018600_018699/ESP_018854_1755_ESP_018920_1755/DTEEC_018854_1755_018920_1755_U01.IMG",
        lbl_url="https://www.uahirise.org/PDS/DTM/ESP/ORB_018600_018699/ESP_018854_1755_ESP_018920_1755/DTEEC_018854_1755_018920_1755_U01.LBL",
    ),
}


# ── Download ─────────────────────────────────────────────────────────────

def download(url: str, dest: Path, *, chunk: int = 1 << 20) -> Path:
    """Download `url` to `dest` with a progress bar. Skips if dest exists."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for piece in r.iter_content(chunk_size=chunk):
                f.write(piece)
                bar.update(len(piece))
    return dest


def fetch_hirise(site_key: str, data_dir: Path) -> tuple[Path, Path]:
    """Download both IMG and LBL for a named HiRISE site."""
    site = HIRISE_SITES[site_key]
    out_dir = Path(data_dir) / "hirise" / site_key
    img = download(site.img_url, out_dir / Path(site.img_url).name)
    lbl = download(site.lbl_url, out_dir / Path(site.lbl_url).name)
    return img, lbl


# ── PDS IMG parsing ──────────────────────────────────────────────────────
# HiRISE DTMs are PDS3 products. The .LBL is plain text; the .IMG has a
# variable-length ASCII header followed by float32 elevation samples (in
# meters relative to the Mars areoid). We parse just enough of the LBL to
# extract dimensions, byte offset, and missing-data sentinel.

def _parse_lbl(lbl_path: Path) -> dict:
    """Parse a PDS3 LBL into a flat dict of the keys we need."""
    text = Path(lbl_path).read_text(errors="replace")
    out: dict[str, str] = {}
    # Very small parser: KEY = VALUE pairs, ignoring objects.
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().rstrip(";").strip().strip('"')
    return out


def load_hirise_dem(img_path: Path, lbl_path: Path) -> np.ndarray:
    """Load a HiRISE DTM IMG into a 2D float32 array of meters."""
    meta = _parse_lbl(lbl_path)
    lines = int(meta["LINES"])
    samples = int(meta["LINE_SAMPLES"])
    sample_bits = int(meta.get("SAMPLE_BITS", "32"))
    sample_type = meta.get("SAMPLE_TYPE", "PC_REAL")
    missing = float(meta.get("MISSING_CONSTANT", "-3.4028226550889e+38"))

    if sample_bits != 32 or "REAL" not in sample_type:
        raise ValueError(f"unexpected sample format: {sample_bits}-bit {sample_type}")

    # PDS images often have a fixed-record header. Easiest robust approach:
    # the elevation block is the LAST `lines * samples * 4` bytes of the file.
    raw = Path(img_path).read_bytes()
    payload_bytes = lines * samples * 4
    if len(raw) < payload_bytes:
        raise ValueError(f"IMG too small: {len(raw)} < {payload_bytes}")
    payload = raw[-payload_bytes:]

    dtype = np.dtype("<f4") if sample_type == "PC_REAL" else np.dtype(">f4")
    dem = np.frombuffer(payload, dtype=dtype).reshape(lines, samples).copy()
    dem[dem == missing] = np.nan
    return dem


# ── Synthetic Mars terrain ──────────────────────────────────────────────

def _value_noise_2d(rng: np.random.Generator, size: int, scale: int) -> np.ndarray:
    """Bilinearly upsampled coarse value noise. `scale` = coarse grid spacing."""
    scale = max(1, int(scale))
    step = max(2, size // scale)
    coarse = rng.standard_normal((step + 1, step + 1)).astype(np.float32)
    ys = np.linspace(0, step, size, dtype=np.float32)
    xs = np.linspace(0, step, size, dtype=np.float32)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, step);  x1 = np.minimum(x0 + 1, step)
    wy = (ys - y0)[:, None];        wx = (xs - x0)[None, :]
    a = coarse[y0[:, None], x0[None, :]]
    b = coarse[y0[:, None], x1[None, :]]
    c = coarse[y1[:, None], x0[None, :]]
    d = coarse[y1[:, None], x1[None, :]]
    return (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy


def _fbm(rng: np.random.Generator, size: int, *, octaves: int = 5,
         persistence: float = 0.5, base_scale: int = 32) -> np.ndarray:
    """Fractal Brownian motion: summed octaves of value noise."""
    base_scale = max(1, base_scale)
    h = np.zeros((size, size), dtype=np.float32)
    amp = 1.0; total = 0.0
    for o in range(octaves):
        scale = max(1, base_scale >> o)
        h += amp * _value_noise_2d(rng, size, scale)
        total += amp
        amp *= persistence
    return h / total


def _ridged_multifractal(rng: np.random.Generator, size: int, *,
                         octaves: int = 6, persistence: float = 0.6,
                         base_scale: int = 64, gain: float = 2.0) -> np.ndarray:
    """Ridged multifractal noise — produces sharp ridges and valleys.

    The classic Musgrave construction: take 1 - |noise|, square it, then
    weight successive octaves by the previous octave's value (so high areas
    accumulate more detail). Generates Mars-like canyon networks.
    """
    base_scale = max(1, base_scale)
    h = np.zeros((size, size), dtype=np.float32)
    weight = np.ones((size, size), dtype=np.float32)
    amp = 1.0; total = 0.0
    for o in range(octaves):
        scale = max(1, base_scale >> o)
        n = _value_noise_2d(rng, size, scale)
        # Normalize to [-1, 1], take ridge transform
        n = n / (np.abs(n).max() + 1e-8)
        signal = (1.0 - np.abs(n)) ** 2
        signal *= weight
        h += signal * amp
        weight = np.clip(signal * gain, 0.0, 1.0)
        total += amp
        amp *= persistence
    return h / total


def _domain_warp(field: np.ndarray, rng: np.random.Generator,
                 strength_cells: float = 12.0) -> np.ndarray:
    """Warp a 2D field by displacing sample positions with low-freq noise.

    Turns straight ridges into organic, twisty canyon-like features.
    """
    size = field.shape[0]
    warp_x = _value_noise_2d(rng, size, scale=24) * strength_cells
    warp_y = _value_noise_2d(rng, size, scale=24) * strength_cells
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    yy = np.clip(yy + warp_y, 0, size - 1)
    xx = np.clip(xx + warp_x, 0, size - 1)
    # Bilinear sample at warped positions
    y0 = np.floor(yy).astype(int); x0 = np.floor(xx).astype(int)
    y1 = np.minimum(y0 + 1, size - 1); x1 = np.minimum(x0 + 1, size - 1)
    wy = (yy - y0); wx = (xx - x0)
    a = field[y0, x0]; b = field[y0, x1]; c = field[y1, x0]; d = field[y1, x1]
    return ((a * (1 - wx) + b * wx) * (1 - wy)
            + (c * (1 - wx) + d * wx) * wy).astype(np.float32)


def _thermal_erosion(h: np.ndarray, *, talus_m: float = 1.5,
                     iterations: int = 25, rate: float = 0.4) -> np.ndarray:
    """Cheap thermal-erosion approximation.

    Each iteration moves a fraction of the excess slope (above `talus_m`)
    from each cell to its lowest 4-neighbor. Smooths out impossibly sharp
    ridges while preserving overall structure.
    """
    h = h.astype(np.float32).copy()
    for _ in range(iterations):
        # Differences to four neighbors (positive = h is higher than neighbor)
        diffs = []
        for axis, shift in [(0, -1), (0, 1), (1, -1), (1, 1)]:
            shifted = np.roll(h, shift, axis=axis)
            d = h - shifted
            np.clip(d - talus_m, 0.0, None, out=d)
            diffs.append((d, axis, shift))
        moved = np.zeros_like(h)
        for d, _, _ in diffs:
            moved += d
        # Take rate * excess from each cell
        h -= rate * moved / 4.0
        # Deposit into the corresponding neighbors (anti-shift)
        for d, axis, shift in diffs:
            h += rate * np.roll(d, -shift, axis=axis) / 4.0
    return h


def synthetic_mars_terrain(
    size: int = 512,
    *,
    seed: int = 0,
    horizontal_m_per_px: float = 1.0,
    style: str = "badlands",
    elevation_amplitude_m: float | None = None,
    octaves: int = 5,
    persistence: float = 0.65,
    n_craters: int = 4,
    rock_density: float = 0.0,
) -> np.ndarray:
    """Procedural Mars heightmap with selectable terrain style.

    style:
      - "badlands"  (default) — Valles-Marineris-like canyons, ridges, mesas.
                    Big elevation swings (~80 m), eroded ridge structure.
      - "plain"     — Jezero-like rolling regolith with sparse craters.
                    Gentle slopes (~4 m amplitude), smooth, walkable.
      - "highland"  — somewhere between: rolling hills with rocky patches.

    horizontal_m_per_px: meters per cell (only affects elevation→slope ratio).
    elevation_amplitude_m: target peak-to-peak elevation. If None, picks a
        style-appropriate default.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)

    if style == "plain":
        amp = elevation_amplitude_m if elevation_amplitude_m is not None else 4.0
        h = _fbm(rng, size, octaves=octaves, persistence=persistence, base_scale=size // 4)
        h = gaussian_filter(h, sigma=4.0).astype(np.float32)
        h = (h - h.mean()) / (h.std() + 1e-8) * (amp / 4.0)
        h = _add_craters(h, rng, n_craters)

    elif style == "highland":
        amp = elevation_amplitude_m if elevation_amplitude_m is not None else 30.0
        base = _fbm(rng, size, octaves=4, persistence=0.55, base_scale=size // 6)
        ridges = _ridged_multifractal(rng, size, octaves=5, persistence=0.55,
                                      base_scale=size // 4)
        h = 0.6 * base + 0.4 * ridges
        h = _domain_warp(h, rng, strength_cells=size * 0.02)
        h = (h - h.mean()) / (h.std() + 1e-8) * (amp / 4.0)
        h = _thermal_erosion(h, talus_m=1.2, iterations=15)
        h = _add_craters(h, rng, n_craters)

    else:  # "badlands"
        # Default amplitude bumped to 150 m so mountains read as MASSIVE
        # next to a 1.7-m human. Real Valles Marineris walls are 1-7 km;
        # at 1-km terrain extent this is the proportional analogue.
        amp = elevation_amplitude_m if elevation_amplitude_m is not None else 150.0
        # Real Mars badlands look heavily eroded: ridges are rounded over
        # geological timescales, not knife-sharp. We construct, then erode
        # *aggressively*, then add fine detail back at low amplitude.
        from scipy.ndimage import gaussian_filter

        base = _fbm(rng, size, octaves=3, persistence=0.55, base_scale=size // 8)
        ridges = _ridged_multifractal(rng, size, octaves=5, persistence=0.55,
                                      base_scale=size // 3)
        # Soften the ridge multifractal — its sharp peaks are the main
        # source of jaggedness in MuJoCo's flat-shaded view.
        ridges = gaussian_filter(ridges, sigma=2.5).astype(np.float32)
        h = 0.45 * base + 0.55 * ridges
        h = _domain_warp(h, rng, strength_cells=size * 0.04)

        h = (h - h.mean()) / (h.std() + 1e-8) * (amp / 4.0)
        # Heavy thermal erosion: 50 iterations with a low talus angle smooths
        # impossible cliffs into rounded slopes. Real Mars terrain has been
        # eroded for billions of years.
        h = _thermal_erosion(h, talus_m=1.5, iterations=50)
        # And a mild Gaussian pass for the final rounding.
        h = gaussian_filter(h, sigma=1.5).astype(np.float32)

        # ── Flatten low-elevation regions into walkable floors ──
        floor_z = float(np.percentile(h, 35))
        softness = max(2.0, amp * 0.06)
        h = floor_z + softness * np.log1p(np.exp(np.clip((h - floor_z) / softness, -50, 50)))

        # Add back a tiny amount of mid-frequency variation so the surface
        # isn't billiard-ball smooth (otherwise it reads as plastic).
        detail = _fbm(rng, size, octaves=3, persistence=0.5, base_scale=size // 32)
        h = h + 0.4 * (detail - detail.mean())
        # Sub-meter dither for surface micro-texture
        dither = gaussian_filter(_value_noise_2d(rng, size, scale=size // 64), sigma=1.0)
        h = h + 0.2 * (dither - dither.mean())

    # ── Optional boulders ──────────────────────────────────────────
    if rock_density > 0:
        ys = np.arange(size)[:, None]
        xs = np.arange(size)[None, :]
        n_rocks = int(size * size * rock_density / 1000)
        for _ in range(n_rocks):
            cy = rng.integers(0, size); cx = rng.integers(0, size)
            r = rng.uniform(0.3, 1.2)
            hgt = rng.uniform(0.05, 0.4)
            h = h + (hgt * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (r * r))).astype(np.float32)

    return h.astype(np.float32)


def _add_craters(h: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """Subtract n procedural craters with raised lips."""
    if n <= 0:
        return h
    from scipy.ndimage import gaussian_filter
    size = h.shape[0]
    ys = np.arange(size)[:, None]; xs = np.arange(size)[None, :]
    for _ in range(n):
        cy = rng.integers(0, size); cx = rng.integers(0, size)
        radius = rng.uniform(size * 0.08, size * 0.22)
        depth = (2 * radius) * rng.uniform(0.05, 0.10)
        r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        bowl = -depth * np.clip(np.cos(np.pi / 2 * r / radius), 0, 1) ** 2
        lip = (depth * 0.35) * np.exp(-((r - radius) ** 2) / (radius * 0.25) ** 2)
        crater = np.where(r < radius * 1.5, bowl + lip, 0.0)
        crater = gaussian_filter(crater, sigma=2.0)
        h = h + crater.astype(np.float32)
    return h


# ── Heightmap → mesh ────────────────────────────────────────────────────

def heightmap_to_mesh(
    heightmap: np.ndarray,
    *,
    horizontal_m_per_px: float = 1.0,
    z_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a 2D heightmap into (vertices Nx3, faces Mx3) arrays.

    Output coordinates are in meters, with X east, Y north, Z up.
    NaN cells are dropped along with their adjacent faces.
    """
    h = np.asarray(heightmap, dtype=np.float32)
    rows, cols = h.shape

    # Center the terrain at world origin (X, Y) so a humanoid spawned at
    # (0, 0, *) lands in the middle of the patch instead of at the corner.
    xs = (np.arange(cols, dtype=np.float32) - (cols - 1) / 2) * horizontal_m_per_px
    ys = (np.arange(rows, dtype=np.float32) - (rows - 1) / 2) * horizontal_m_per_px
    xv, yv = np.meshgrid(xs, ys)
    # Also subtract the mean elevation so the terrain straddles z=0 instead
    # of floating off in elevation. Keeps the humanoid spawn height useful.
    zv = (h - np.nanmean(h)) * z_scale

    verts = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)

    # Build quad → 2-triangle faces, skipping any quad with a NaN corner.
    idx = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    a = idx[:-1, :-1]
    b = idx[:-1, 1:]
    c = idx[1:, :-1]
    d = idx[1:, 1:]
    quad_valid = (
        np.isfinite(h[:-1, :-1]) & np.isfinite(h[:-1, 1:])
        & np.isfinite(h[1:, :-1]) & np.isfinite(h[1:, 1:])
    )
    a, b, c, d = a[quad_valid], b[quad_valid], c[quad_valid], d[quad_valid]
    tri1 = np.stack([a, c, b], axis=-1)
    tri2 = np.stack([b, c, d], axis=-1)
    faces = np.concatenate([tri1, tri2], axis=0)

    # Replace NaN vertex Zs with 0 so OBJ writers don't choke. The faces
    # referencing them have already been removed.
    bad = ~np.isfinite(verts[:, 2])
    verts[bad, 2] = 0.0
    return verts, faces


def save_heightfield_png(heightmap: np.ndarray, path: Path) -> tuple[Path, float, float]:
    """Write a heightmap as an 8-bit grayscale PNG for MuJoCo `<hfield>`.

    Returns (path, z_min, z_range_m). MuJoCo maps PNG values [0..255] to
    elevations [0 .. z_range]; the caller adds the offset z_min back when
    placing the hfield in world coordinates.
    """
    from PIL import Image
    h = np.asarray(heightmap, dtype=np.float32)
    z_min = float(np.nanmin(h))
    z_max = float(np.nanmax(h))
    z_range = max(z_max - z_min, 1e-6)
    norm = ((h - z_min) / z_range * 255.0).clip(0, 255).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(norm, mode="L").save(path)
    return path, z_min, z_range


def save_obj(verts: np.ndarray, faces: np.ndarray, path: Path) -> Path:
    """Minimal OBJ writer (no normals/UVs — those get added in Blender)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    buf.write("# MarsWorldModel terrain\n")
    for v in verts:
        buf.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
    for f in faces + 1:  # OBJ is 1-indexed
        buf.write(f"f {f[0]} {f[1]} {f[2]}\n")
    path.write_text(buf.getvalue())
    return path


# ── High-level convenience ──────────────────────────────────────────────

def build_terrain(
    *,
    site: str | None = None,
    out_path: Path,
    data_dir: Path = Path("data/raw"),
    synthetic_size: int = 512,
    synthetic_seed: int = 0,
    synthetic_style: str = "badlands",
    horizontal_m_per_px: float = 1.0,
) -> Path:
    """Top-level entry: build a Mars terrain mesh, real or synthetic.

    If `site` is given (and present in HIRISE_SITES), download and use the
    real HiRISE DTM. Otherwise generate a synthetic Mars-like heightmap.
    """
    if site:
        img, lbl = fetch_hirise(site, data_dir)
        dem = load_hirise_dem(img, lbl)
    else:
        dem = synthetic_mars_terrain(
            size=synthetic_size, seed=synthetic_seed, style=synthetic_style
        )

    verts, faces = heightmap_to_mesh(
        dem, horizontal_m_per_px=horizontal_m_per_px
    )
    obj_path = save_obj(verts, faces, out_path)
    # Also write a sibling PNG heightfield for MuJoCo's <hfield>.
    png_path = obj_path.with_suffix(".png")
    save_heightfield_png(dem - np.nanmean(dem), png_path)
    # And the raw .npy for any future consumer that wants float precision.
    np.save(obj_path.with_suffix(".npy"), dem - np.nanmean(dem))
    return obj_path
