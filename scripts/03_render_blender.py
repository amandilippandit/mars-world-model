"""Blender driver: read a SceneSpec, build the Mars scene, render frames.

Run via:
    blender --background --python scripts/03_render_blender.py -- \
        --spec data/renders/run01/spec.json \
        --out  data/renders/run01

Invoked indirectly by `mars.render.BlenderRenderer`.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _user_args() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(here / "src"))

    from mars.render import SceneSpec
    from mars.scene import (
        add_camera, add_sun, assign_material,
        build_mars_scene, import_terrain, make_regolith_material,
        reset_scene, setup_mars_sky,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--resx", type=int, default=1280)
    p.add_argument("--resy", type=int, default=720)
    args = p.parse_args(_user_args())

    spec = SceneSpec.load(args.spec)

    import bpy
    build_mars_scene(
        spec.terrain_obj,
        sun_elevation_deg=spec.sun_elevation_deg,
        sun_azimuth_deg=spec.sun_azimuth_deg,
    )

    scn = bpy.context.scene
    scn.cycles.samples = args.samples
    scn.render.resolution_x = args.resx
    scn.render.resolution_y = args.resy

    cam = scn.camera
    n_frames = max(1, int(spec.duration_s * spec.fps))

    args.out.mkdir(parents=True, exist_ok=True)

    # Walk the camera through its keyframes (linear interp between t_s).
    kfs = sorted(spec.camera_keyframes, key=lambda k: k.t_s)
    if not kfs:
        raise SystemExit("spec has no camera_keyframes")

    def lerp(a, b, t):
        return tuple(ai + (bi - ai) * t for ai, bi in zip(a, b))

    def pose_at(t: float):
        if t <= kfs[0].t_s:
            return kfs[0].position_m, kfs[0].look_at_m, kfs[0].fov_deg
        if t >= kfs[-1].t_s:
            return kfs[-1].position_m, kfs[-1].look_at_m, kfs[-1].fov_deg
        for k0, k1 in zip(kfs, kfs[1:]):
            if k0.t_s <= t <= k1.t_s:
                u = (t - k0.t_s) / max(1e-9, (k1.t_s - k0.t_s))
                return (
                    lerp(k0.position_m, k1.position_m, u),
                    lerp(k0.look_at_m, k1.look_at_m, u),
                    k0.fov_deg + (k1.fov_deg - k0.fov_deg) * u,
                )
        return kfs[-1].position_m, kfs[-1].look_at_m, kfs[-1].fov_deg

    import mathutils
    for i in range(n_frames):
        t = i / spec.fps
        pos, target, fov = pose_at(t)
        cam.location = pos
        # Aim camera by aligning -Z
        d = mathutils.Vector(target) - mathutils.Vector(pos)
        cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        cam.data.angle = math.radians(fov)

        scn.render.filepath = str((args.out / f"frame_{i:04d}.png").resolve())
        bpy.ops.render.render(write_still=True)
        print(f"  frame {i+1}/{n_frames} → {scn.render.filepath}")


if __name__ == "__main__":
    main()
