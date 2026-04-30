"""Build a Mars Blender scene from a terrain OBJ.

Run inside Blender:

    blender --background --python scripts/02_build_scene.py -- \
        --terrain data/processed/terrain.obj \
        --out     data/processed/mars.blend

Or render a single frame to a PNG:

    blender --background --python scripts/02_build_scene.py -- \
        --terrain data/processed/terrain.obj \
        --render  data/renders/mars_test.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# When Blender invokes this script, sys.argv contains Blender's own args
# before "--"; the user args come after.
def _user_args() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def main() -> None:
    # Ensure src/ is on the path so we can import `mars.*` from inside Blender.
    here = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(here / "src"))

    from mars.scene import build_mars_scene, save_blend

    p = argparse.ArgumentParser()
    p.add_argument("--terrain", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="Save .blend here")
    p.add_argument("--render", type=Path, default=None, help="Render PNG here")
    p.add_argument("--sun-elev", type=float, default=35.0)
    p.add_argument("--sun-az", type=float, default=135.0)
    args = p.parse_args(_user_args())

    build_mars_scene(
        args.terrain,
        sun_elevation_deg=args.sun_elev,
        sun_azimuth_deg=args.sun_az,
    )

    if args.out:
        save_blend(args.out)
        print(f"saved blend → {args.out}")

    if args.render:
        import bpy
        args.render.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(args.render.resolve())
        bpy.ops.render.render(write_still=True)
        print(f"rendered    → {args.render}")


if __name__ == "__main__":
    main()
