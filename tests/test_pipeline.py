"""End-to-end smoke tests that don't require Blender or Genesis.

These cover everything that runs natively on a 16 GB M2 Mac with just the
core Python deps installed:
  - Mars constants
  - Synthetic terrain generation
  - Heightmap → mesh
  - OBJ writer
  - SceneSpec serialization
  - Renderer factory
  - Atmospheric drag math
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mars import MARS
from mars.physics import mars_drag_force
from mars.render import CameraKeyframe, SceneSpec, get_renderer
from mars.terrain import (
    heightmap_to_mesh, save_obj, synthetic_mars_terrain,
)


def test_mars_constants_sanity():
    assert 3.7 < MARS.gravity_m_s2 < 3.8
    assert 500 < MARS.surface_pressure_pa < 800
    assert 0.015 < MARS.surface_density_kg_m3 < 0.025
    assert MARS.year_sols > 600
    assert MARS.solar_irradiance_w_m2 < 1361   # less than Earth's


def test_synthetic_terrain_shape_and_stats():
    h = synthetic_mars_terrain(size=128, seed=42)
    assert h.shape == (128, 128)
    assert h.dtype == np.float32
    # noise should be near zero-mean and have bounded amplitude
    assert abs(h.mean()) < 5.0
    assert h.std() < 25.0


def test_heightmap_to_mesh_dimensions():
    h = synthetic_mars_terrain(size=32, seed=0)
    v, f = heightmap_to_mesh(h, horizontal_m_per_px=1.0)
    assert v.shape == (32 * 32, 3)
    # 31×31 quads → 31*31*2 = 1922 triangles, all valid (no NaN in synthetic)
    assert f.shape == (31 * 31 * 2, 3)
    # X coords are centered at origin: span [-15.5, +15.5] for a 32-cell grid
    assert v[:, 0].min() == pytest.approx(-15.5)
    assert v[:, 0].max() == pytest.approx(15.5)
    # Z is mean-centered, so the heightmap should average to ~0
    assert abs(v[:, 2].mean()) < 1.0


def test_obj_roundtrip(tmp_path: Path):
    h = synthetic_mars_terrain(size=16, seed=1)
    v, f = heightmap_to_mesh(h)
    out = save_obj(v, f, tmp_path / "t.obj")
    assert out.exists()
    text = out.read_text()
    assert text.startswith("# MarsWorldModel")
    # 16x16 vertices + 15*15*2 face lines + header
    assert text.count("\nv ") == 16 * 16
    assert text.count("\nf ") == 15 * 15 * 2


def test_scene_spec_roundtrip(tmp_path: Path):
    s = SceneSpec(
        terrain_obj=Path("data/processed/terrain.obj"),
        camera_keyframes=[
            CameraKeyframe(t_s=0.0, position_m=(0, 0, 1.7), look_at_m=(10, 0, 1)),
            CameraKeyframe(t_s=4.0, position_m=(5, 0, 1.7), look_at_m=(15, 0, 1)),
        ],
        duration_s=4.0,
    )
    p = s.save(tmp_path / "spec.json")
    s2 = SceneSpec.load(p)
    assert s2.duration_s == 4.0
    assert len(s2.camera_keyframes) == 2
    assert s2.camera_keyframes[1].t_s == 4.0
    assert s2.terrain_obj == Path("data/processed/terrain.obj")


def test_renderer_factory():
    r1 = get_renderer("wan_colab")
    assert r1.name == "wan_colab"
    # blender renderer construction shouldn't fail even if Blender isn't on PATH —
    # we only fail at render time. Skip the blender path here since _find_blender
    # raises when Blender isn't installed.


def test_mars_drag_scaling():
    # Walking-speed drag is negligible on Mars
    f_walk = mars_drag_force(np.array([1.4, 0.0, 0.0]), area_m2=0.7)
    assert np.linalg.norm(f_walk) < 0.05

    # Parachute-speed drag is huge
    f_chute = mars_drag_force(np.array([100.0, 0.0, 0.0]), area_m2=200.0)
    assert np.linalg.norm(f_chute) > 1000

    # Direction is opposite velocity
    v = np.array([5.0, 0.0, 0.0])
    f = mars_drag_force(v, area_m2=1.0)
    assert f[0] < 0
