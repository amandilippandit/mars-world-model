"""Blender scene composer for Mars.

This module is intended to be executed *inside Blender*:

    blender --background --python scripts/02_build_scene.py -- --terrain data/processed/terrain.obj --out data/processed/mars.blend

The `bpy` import only resolves when running under Blender; we keep all bpy
calls inside functions so the module can still be imported (e.g. for tests)
from a normal Python where bpy is unavailable.

What this builds:
- Sun light at Mars TOA irradiance, paler/warmer disk
- Sky shader approximating Mars's butterscotch atmosphere with dust scattering
- A regolith material (PBR) bound to the imported terrain mesh
- Camera placed ~1.7 m above the terrain (humanoid eye height)
"""

from __future__ import annotations

import math
from pathlib import Path

from mars.constants import MARS


def _bpy():
    """Import bpy lazily so this module can be imported outside Blender."""
    import bpy  # noqa: WPS433  (deliberate lazy import)
    return bpy


# ── Scene reset ──────────────────────────────────────────────────────────

def reset_scene() -> None:
    bpy = _bpy()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Use Cycles for physically-based rendering; Eevee is fine but Cycles
    # gives us proper atmospheric scattering.
    bpy.context.scene.render.engine = "CYCLES"
    try:
        bpy.context.scene.cycles.device = "GPU"
    except Exception:
        bpy.context.scene.cycles.device = "CPU"
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720
    # Color management: Filmic looks closer to what Mars cameras produce
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium Contrast"


# ── Terrain ──────────────────────────────────────────────────────────────

def import_terrain(obj_path: Path) -> "object":  # noqa: F821
    bpy = _bpy()
    obj_path = Path(obj_path).resolve()
    if not obj_path.exists():
        raise FileNotFoundError(obj_path)
    # Blender 4.x uses wm.obj_import; 3.x used import_scene.obj. Try both.
    try:
        bpy.ops.wm.obj_import(filepath=str(obj_path))
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=str(obj_path))
    obj = bpy.context.selected_objects[0]
    obj.name = "MarsTerrain"
    return obj


# ── Materials ────────────────────────────────────────────────────────────

def make_regolith_material():
    """PBR material approximating Mars regolith: warm rust, low specular."""
    bpy = _bpy()
    mat = bpy.data.materials.new("Regolith")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    noise = nodes.new("ShaderNodeTexNoise")
    cramp = nodes.new("ShaderNodeValToRGB")
    bump = nodes.new("ShaderNodeBump")

    noise.inputs["Scale"].default_value = 8.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.6

    # Color ramp from dark basaltic to lighter dust
    cramp.color_ramp.elements[0].color = (0.32, 0.18, 0.10, 1.0)
    cramp.color_ramp.elements[1].color = (*MARS.regolith_color_rgb, 1.0)

    bsdf.inputs["Roughness"].default_value = 0.95
    if "Specular IOR Level" in bsdf.inputs:           # Blender 4.x
        bsdf.inputs["Specular IOR Level"].default_value = 0.1
    elif "Specular" in bsdf.inputs:                   # Blender 3.x
        bsdf.inputs["Specular"].default_value = 0.1

    bump.inputs["Strength"].default_value = 0.4

    links.new(noise.outputs["Fac"], cramp.inputs["Fac"])
    links.new(cramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def assign_material(obj, mat) -> None:
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


# ── Lighting / sky ──────────────────────────────────────────────────────

def add_sun(elevation_deg: float = 45.0, azimuth_deg: float = 135.0):
    """Sun light tuned for Mars TOA irradiance and a paler disk color."""
    bpy = _bpy()
    bpy.ops.object.light_add(type="SUN")
    sun = bpy.context.object
    sun.name = "MarsSun"
    sun.data.energy = MARS.solar_irradiance_w_m2 / 100.0   # Blender wants ~5–10 here
    sun.data.color = MARS.sun_disk_color_rgb
    sun.data.angle = math.radians(MARS.sun_angular_diameter_arcmin / 60.0)

    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    # Aim sun by setting its rotation; default sun shines along -Z
    sun.rotation_euler = (math.pi / 2 - el, 0.0, az)
    return sun


def setup_mars_sky() -> None:
    """World shader: warm horizon, deeper zenith, dust scattering."""
    bpy = _bpy()
    world = bpy.context.scene.world or bpy.data.worlds.new("MarsWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    grad = nt.nodes.new("ShaderNodeTexGradient")
    cramp = nt.nodes.new("ShaderNodeValToRGB")
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    mapr = nt.nodes.new("ShaderNodeMapRange")

    # Use the world Z component of incoming direction (geometry → normal Z)
    # to drive a horizon→zenith gradient.
    nt.links.new(geom.outputs["Incoming"], sep.inputs["Vector"])
    mapr.inputs["From Min"].default_value = -0.2
    mapr.inputs["From Max"].default_value = 1.0
    nt.links.new(sep.outputs["Z"], mapr.inputs["Value"])
    nt.links.new(mapr.outputs["Result"], cramp.inputs["Fac"])

    cramp.color_ramp.elements[0].color = (*MARS.sky_color_horizon_rgb, 1.0)
    cramp.color_ramp.elements[1].color = (*MARS.sky_color_zenith_rgb, 1.0)

    # Mars sky is dim — well under Earth's. Tune background strength to
    # roughly match the lower TOA irradiance.
    bg.inputs["Strength"].default_value = 0.4

    nt.links.new(cramp.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    # (grad node kept around in case we want to switch driver later)
    _ = grad


# ── Camera ───────────────────────────────────────────────────────────────

def add_camera(
    location: tuple[float, float, float] = (0.0, 0.0, 1.7),
    look_at: tuple[float, float, float] = (10.0, 0.0, 0.0),
    focal_mm: float = 28.0,
):
    """Camera at humanoid eye height looking forward."""
    bpy = _bpy()
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.object
    cam.name = "MarsCam"
    cam.data.lens = focal_mm
    cam.data.clip_start = 0.05
    cam.data.clip_end = 5000.0

    # Orient toward look_at by aligning -Z axis
    import mathutils  # noqa: WPS433
    direction = mathutils.Vector(look_at) - mathutils.Vector(location)
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()

    bpy.context.scene.camera = cam
    return cam


# ── Top-level orchestrator ──────────────────────────────────────────────

def build_mars_scene(
    terrain_obj: Path,
    *,
    sun_elevation_deg: float = 35.0,
    sun_azimuth_deg: float = 135.0,
) -> None:
    reset_scene()
    terrain = import_terrain(terrain_obj)
    assign_material(terrain, make_regolith_material())
    setup_mars_sky()
    add_sun(elevation_deg=sun_elevation_deg, azimuth_deg=sun_azimuth_deg)

    # Place the camera at the terrain's mean Z + 1.7 m, near its center.
    bbox = [terrain.matrix_world @ v.co for v in terrain.data.vertices[::1024]]
    if bbox:
        xs = [v.x for v in bbox]; ys = [v.y for v in bbox]; zs = [v.z for v in bbox]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        cz = max(zs) + 1.7
        add_camera(location=(cx, cy, cz), look_at=(cx + 10, cy, cz - 1.0))
    else:
        add_camera()


def save_blend(path: Path) -> Path:
    bpy = _bpy()
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    return path
