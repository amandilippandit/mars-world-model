"""Pluggable renderer interface.

The simulator produces *scene specs* (terrain mesh + camera trajectory +
optional text prompt). A renderer turns a spec into one or more image/video
files. The point of this abstraction is that the same scene spec can be
rendered four different ways depending on what's available:

  ┌──────────────┬─────────────┬──────────────┬─────────────────────┐
  │  Renderer    │  Where      │  Cost        │  Realism            │
  ├──────────────┼─────────────┼──────────────┼─────────────────────┤
  │  blender     │  M2 Mac     │  free        │  good (PBR Cycles)  │
  │  wan_local   │  M2 Mac MPS │  free, slow  │  AI photoreal       │
  │  wan_colab   │  Colab T4   │  free        │  AI photoreal       │
  │  cosmos_api  │  NVIDIA API │  ~credits    │  best                │
  └──────────────┴─────────────┴──────────────┴─────────────────────┘

For v1 we ship `blender` + a stub for `wan_colab` (the actual Wan inference
runs in the Colab notebook; this module just packages the spec for upload
and reads results back).
"""

from __future__ import annotations

import json
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class CameraKeyframe:
    t_s: float
    position_m: tuple[float, float, float]
    look_at_m: tuple[float, float, float]
    fov_deg: float = 60.0


@dataclass
class SceneSpec:
    """Everything a renderer needs to produce frames for one rollout."""
    terrain_obj: Path
    camera_keyframes: list[CameraKeyframe]
    fps: int = 24
    duration_s: float = 4.0
    sun_elevation_deg: float = 35.0
    sun_azimuth_deg: float = 135.0
    text_prompt: str = (
        "Mars surface, butterscotch sky, regolith terrain, low-angle sun, "
        "thin dusty atmosphere, photorealistic, cinematic, 4K"
    )
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {**asdict(self), "terrain_obj": str(self.terrain_obj)},
            indent=2, default=str,
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @classmethod
    def load(cls, path: Path) -> "SceneSpec":
        d = json.loads(Path(path).read_text())
        d["terrain_obj"] = Path(d["terrain_obj"])
        d["camera_keyframes"] = [CameraKeyframe(**k) for k in d["camera_keyframes"]]
        d.pop("extra", None)
        return cls(**d, extra={})


# ── Base interface ──────────────────────────────────────────────────────

class Renderer(ABC):
    name: str

    @abstractmethod
    def render(self, spec: SceneSpec, out_dir: Path) -> list[Path]:
        """Produce frames or a video. Returns paths to output files."""


# ── Blender (local, free, deterministic) ────────────────────────────────

class BlenderRenderer(Renderer):
    """Local PBR rendering via Blender Cycles. No AI, no internet, free."""

    name = "blender"

    def __init__(
        self,
        blender_bin: str = None,
        samples: int = 64,
        resolution: tuple[int, int] = (1280, 720),
    ) -> None:
        self.blender_bin = blender_bin or self._find_blender()
        self.samples = samples
        self.resolution = resolution

    @staticmethod
    def _find_blender() -> str:
        for cand in (
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "blender",
        ):
            if cand == "blender" or Path(cand).exists():
                return cand
        raise FileNotFoundError(
            "Blender not found. Install with: brew install --cask blender"
        )

    def render(self, spec: SceneSpec, out_dir: Path) -> list[Path]:
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Persist the spec so the Blender child process can read it
        spec_path = out_dir / "spec.json"
        spec.save(spec_path)

        # The driver script lives at scripts/03_render_blender.py and
        # consumes spec.json
        repo = Path(__file__).resolve().parent.parent.parent
        driver = repo / "scripts" / "03_render_blender.py"

        cmd = [
            self.blender_bin, "--background", "--python", str(driver), "--",
            "--spec", str(spec_path),
            "--out", str(out_dir),
            "--samples", str(self.samples),
            "--resx", str(self.resolution[0]),
            "--resy", str(self.resolution[1]),
        ]
        env = {**os.environ, "PYTHONPATH": str(repo / "src")}
        subprocess.run(cmd, check=True, env=env)

        return sorted(out_dir.glob("frame_*.png"))


# ── Wan-on-Colab (free, AI photoreal, not local) ────────────────────────

class WanColabRenderer(Renderer):
    """Stub: package the spec for a Colab notebook to consume.

    The actual Wan 2.1 inference runs in `notebooks/wan_colab_render.ipynb`.
    This renderer just (a) writes the spec where Colab can pick it up
    (currently expects you to upload the spec dir to Drive yourself), and
    (b) optionally polls a results dir for outputs.
    """

    name = "wan_colab"

    def __init__(self, drive_inbox: Path | None = None) -> None:
        # If you mount Drive on your Mac (e.g. with rclone), point this at it
        # and the spec will land in Colab automatically. Otherwise just
        # upload manually.
        self.drive_inbox = Path(drive_inbox) if drive_inbox else None

    def render(self, spec: SceneSpec, out_dir: Path) -> list[Path]:
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        spec_path = out_dir / "spec.json"
        spec.save(spec_path)

        if self.drive_inbox:
            self.drive_inbox.mkdir(parents=True, exist_ok=True)
            (self.drive_inbox / spec_path.name).write_text(spec_path.read_text())

        print(
            f"[wan_colab] spec written to {spec_path}\n"
            f"[wan_colab] open notebooks/wan_colab_render.ipynb in Colab,\n"
            f"           upload {spec_path}, run all cells, then download\n"
            f"           the resulting MP4 back to {out_dir}/"
        )
        return []


# ── Factory ─────────────────────────────────────────────────────────────

RendererName = Literal["blender", "wan_colab"]


def get_renderer(name: RendererName, **kw) -> Renderer:
    if name == "blender":
        return BlenderRenderer(**kw)
    if name == "wan_colab":
        return WanColabRenderer(**kw)
    raise ValueError(f"unknown renderer: {name}")
