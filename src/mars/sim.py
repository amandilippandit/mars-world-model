"""Interactive Mars sim with a humanoid, runs natively on macOS via MuJoCo.

Composes a runtime MJCF that combines:
  - Mars gravity (3.721 m/s² down)
  - A heightfield collider built from the terrain PNG produced by
    `mars.terrain` (we use `<hfield>` rather than `<mesh>` because mesh
    contacts are unstable on noisy synthetic terrain)
  - The humanoid asset from `assets/humanoid.xml`
  - Mars-tinted skybox + Mars-warm sun

Then opens MuJoCo's interactive viewer in a native window. You can:
  - Orbit the camera with the mouse
  - Drag the humanoid around with ctrl+click
  - Apply forces with ctrl+right-drag
  - Toggle physics with space, step with right-arrow
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mars.constants import MARS

ASSETS = Path(__file__).resolve().parent / "assets"


def _hfield_metadata(npy_path: Path) -> tuple[int, int, float, float]:
    """Read the cached heightmap .npy and return (nrow, ncol, z_min, z_range)."""
    arr = np.load(npy_path)
    return arr.shape[0], arr.shape[1], float(arr.min()), float(arr.max() - arr.min())


def _terrain_height_at_origin(npy_path: Path, search_radius_cells: int = 2) -> float:
    """Look up the heightfield's Z value at the (0,0) world cell.

    The terrain is centered at world origin, so the (0,0) world position
    maps to the center cell of the heightmap array.
    """
    arr = np.load(npy_path)
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    r = search_radius_cells
    patch = arr[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
    return float(np.nanmax(patch))


@dataclass
class SimConfig:
    terrain_obj: Path
    timestep_s: float = 1.0 / 480.0  # tighter step keeps soft-contact + ragdoll stable
    spawn_height_m: float = 2.5     # above the terrain center, lets gravity drop the humanoid
    enable_humanoid: bool = True
    fly_step_m: float = 4.0          # meters per WASD/QE keypress in fly-through


def build_mjcf(cfg: SimConfig) -> str:
    """Compose the runtime MJCF as a string.

    The terrain OBJ is referenced as an external mesh asset. The humanoid
    XML is read and inlined under <worldbody> so we don't need MuJoCo's
    <include> machinery (which has finicky path rules).
    """
    terrain_path = Path(cfg.terrain_obj).resolve()
    if not terrain_path.exists():
        raise FileNotFoundError(
            f"terrain not found at {terrain_path}. Run: mars terrain"
        )
    npy_path = terrain_path.with_suffix(".npy")
    png_path = terrain_path.with_suffix(".png")
    if not (npy_path.exists() and png_path.exists()):
        raise FileNotFoundError(
            f"heightfield sidecar files not found ({npy_path}, {png_path}).\n"
            f"Re-run `mars terrain` to regenerate."
        )
    nrow, ncol, z_min, z_range = _hfield_metadata(npy_path)
    # Heightmap horizontal extent: 1 m per cell, centered at origin.
    half_x = (ncol - 1) / 2.0
    half_y = (nrow - 1) / 2.0
    # Compute actual terrain elevation under the humanoid spawn point so
    # we can place the torso above (not inside) the surface.
    terrain_z_at_spawn = _terrain_height_at_origin(npy_path) + z_min  # already absolute

    # Read the humanoid MJCF and pull out just its <worldbody> contents
    # so we can splice it into our composed world. This is hacky but
    # sidesteps cross-file include resolution.
    humanoid_xml = (ASSETS / "humanoid.xml").read_text()
    body_start = humanoid_xml.find("<worldbody>") + len("<worldbody>")
    body_end = humanoid_xml.find("</worldbody>")
    humanoid_body = humanoid_xml[body_start:body_end]
    actuator_start = humanoid_xml.find("<actuator>")
    actuator_end = humanoid_xml.find("</actuator>") + len("</actuator>")
    humanoid_actuators = humanoid_xml[actuator_start:actuator_end] if actuator_start >= 0 else ""

    # Reposition the humanoid above the terrain at world (0,0). spawn_height_m
    # is clearance above the terrain peak under the spawn point.
    spawn_z = terrain_z_at_spawn + cfg.spawn_height_m
    humanoid_body = humanoid_body.replace(
        'name="torso" pos="0 0 1.4"',
        f'name="torso" pos="0 0 {spawn_z}"',
    )

    # Mars sky color for the GL viewport background. MuJoCo uses an RGB
    # gradient between haze (top) and sky (bottom).
    sky_top = MARS.sky_color_zenith_rgb
    sky_bot = MARS.sky_color_horizon_rgb
    regolith = MARS.regolith_color_rgb

    body_section = humanoid_body if cfg.enable_humanoid else ""
    act_section = humanoid_actuators if cfg.enable_humanoid else ""

    # Haze color: blend the horizon and zenith sky colors. Distant terrain
    # will fade to this. Tuned slightly toward the horizon (warmer, lighter)
    # so atmospheric perspective reads as "Mars dust" rather than "fog".
    haze_r = 0.5 * (sky_top[0] + sky_bot[0]) * 1.05
    haze_g = 0.5 * (sky_top[1] + sky_bot[1]) * 1.05
    haze_b = 0.5 * (sky_top[2] + sky_bot[2]) * 1.05

    # Regolith texture: a baked procedural PNG. Repeat density picks how
    # fine the grain reads. Higher repeat = smaller pebbles in view.
    regolith_png = ASSETS / "regolith.png"
    if not regolith_png.exists():
        raise FileNotFoundError(
            f"regolith texture missing at {regolith_png}.\n"
            "Run: python3 scripts/bake_regolith_texture.py"
        )
    # 1 texture tile per 4 m of terrain ≈ each pebble in the texture is
    # ~1 mm on the ground. Cranks up the "sand" feel.
    texrepeat = max(8, int(2 * max(half_x, half_y) / 4.0))

    return textwrap.dedent(f"""\
        <mujoco model="mars_world">
          <option gravity="0 0 -{MARS.gravity_m_s2}" timestep="{cfg.timestep_s}"
                  density="{MARS.surface_density_kg_m3}"
                  viscosity="0.000011"
                  wind="0 0 0"/>

          <visual>
            <!-- Mars ambient: warm dust-scattered light fills shadows so
                 they don't go pure black. On real Mars, even the shadowed
                 side of a rock looks reddish-tan, not gray. -->
            <headlight diffuse="0.55 0.48 0.40"
                       ambient="0.55 0.42 0.32"
                       specular="0 0 0"/>
            <!-- Atmospheric haze (fog). Distant terrain fades to this color,
                 producing the depth cue that makes Mars look like Mars. -->
            <rgba haze="{haze_r:.3f} {haze_g:.3f} {haze_b:.3f} 1"/>
            <map fogstart="100" fogend="900" haze="0.35"
                 znear="0.05" zfar="6000"/>
            <quality shadowsize="4096" offsamples="4"/>
            <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
          </visual>

          <asset>
            <!-- Pure gradient skybox: zenith color → horizon color. -->
            <texture type="skybox" builtin="gradient"
                     rgb1="{sky_top[0]} {sky_top[1]} {sky_top[2]}"
                     rgb2="{sky_bot[0]} {sky_bot[1]} {sky_bot[2]}"
                     width="512" height="3072"/>
            <!-- Procedural regolith texture baked by
                 scripts/bake_regolith_texture.py. 2048² PNG with sand
                 grains, scattered pebbles, and dust patches. -->
            <texture type="2d" name="regolith_tex"
                     file="{regolith_png}"/>
            <material name="regolith" texture="regolith_tex"
                      texuniform="false" texrepeat="{texrepeat} {texrepeat}"
                      specular="0.02" shininess="0.05" reflectance="0"/>
            <hfield name="mars_hfield" file="{png_path}"
                    nrow="{nrow}" ncol="{ncol}"
                    size="{half_x} {half_y} {z_range} {max(2.0, z_range)}"/>
          </asset>

          <worldbody>
            <!-- Mars sun: dimmer than Earth (~43% TOA irradiance), reddened
                 by atmospheric dust. Lower angle reads more cinematic. -->
            <light name="sun" pos="80 -50 60" dir="-0.55 0.35 -0.75"
                   diffuse="0.78 0.62 0.45" specular="0.05 0.04 0.03"
                   castshadow="true"/>

            <!-- Mars ground plane that fills the void past the heightfield
                 edge. Sits 50 m below the heightfield's bottom plate to
                 avoid spurious contacts (MuJoCo's hfield has its own
                 collision volume that can intersect a too-close plane).
                 Disabled from collision (contype=0) so it never trips the
                 humanoid — purely visual horizon fill. -->
            <geom name="ground_plane" type="plane" size="3000 3000 1"
                  pos="0 0 {z_min - 50.0}"
                  material="regolith"
                  contype="0" conaffinity="0"/>

            <geom name="terrain" type="hfield" hfield="mars_hfield"
                  pos="0 0 {z_min}"
                  material="regolith"
                  friction="{MARS.kinetic_friction_regolith} 0.005 0.0001"
                  contype="1" conaffinity="1"/>

            {body_section}
          </worldbody>

          {act_section}
        </mujoco>
    """)


def _running_under_mjpython() -> bool:
    """Detect whether the current interpreter is `mjpython`.

    Uses mujoco's own canonical check: mjpython sets a `_MJPYTHON` singleton
    inside `mujoco.viewer` during startup. This is more reliable than
    inspecting sys.executable (mjpython re-execs Python so the executable
    path doesn't always contain "mjpython").
    """
    try:
        from mujoco.viewer import _MJPYTHON, _MjPythonBase  # noqa: PLC0415
        return isinstance(_MJPYTHON, _MjPythonBase)
    except (ImportError, AttributeError):
        return False


def launch_interactive(cfg: SimConfig) -> None:
    """Open the native MuJoCo viewer with the composed Mars world.

    Tries `launch_passive` (custom WASD walk controller). Falls back to
    `launch` (passive ragdoll, mouse-orbit) if mjpython isn't available on
    macOS.

    The walk controller in passive mode:
      - Position-actuator PD drives joint targets each timestep
      - WASD updates a "movement intent" that advances a walk phase
      - Joint targets at each phase form a biomechanical walk pattern
      - Forward locomotion comes from real foot push-off against the
        Mars heightfield → real Mars gravity, real friction, real contact
    """
    import math
    import platform
    import mujoco
    import mujoco.viewer
    import numpy as np

    xml = build_mjcf(cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    on_mac = platform.system() == "Darwin"
    can_passive = (not on_mac) or _running_under_mjpython()

    print(f"Mars sim ready:")
    print(f"  gravity: {MARS.gravity_m_s2} m/s²")
    print(f"  atmosphere density: {MARS.surface_density_kg_m3} kg/m³")
    print(f"  terrain: {cfg.terrain_obj}")
    print(f"  humanoid: {'on' if cfg.enable_humanoid else 'off'}")
    print(f"  bodies: {model.nbody}, dofs: {model.nv}, actuators: {model.nu}")
    print()

    if not can_passive:
        print()
        print("━" * 60)
        print(" ERROR: this command must be run with `mjpython` on macOS.")
        print("━" * 60)
        print()
        print(" The walk controller and WASD keyboard input require")
        print(" mujoco.viewer.launch_passive(), which on macOS only works")
        print(" under mjpython (a thin wrapper that ships with the mujoco")
        print(" pip package). Regular python3 cannot do passive viewer.")
        print()
        print(" Re-run as:")
        print()
        print("     PYTHONPATH=src mjpython -m mars.cli play")
        print()
        print(" mjpython is at: " + (__import__('shutil').which('mjpython') or '<not on PATH>'))
        print()
        return

    print("───── viewer controls (walk mode) ─────")
    print("  ↑ / ↓          walk forward / back")
    print("  ← / →          turn left / right")
    print("  shift          run (faster cadence + bigger stride)")
    print("  R              reset humanoid pose")
    print()
    print("  Note: W/A/S/D conflict with MuJoCo's built-in toggles")
    print("  (wireframe, transparency, etc.) so we use arrows instead.")
    print()
    print("  left-drag      orbit camera")
    print("  right-drag     pan")
    print("  scroll         zoom")
    print("  ctrl+drag      apply force to humanoid")
    print("  space          pause physics")
    print("  esc / close    quit")
    print()

    # ── Map actuator names to indices for the walk controller ──────
    actuator_names = [
        "abdomen_y", "abdomen_z", "abdomen_x",
        "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
        "left_hip_x", "left_hip_z", "left_hip_y", "left_knee",
        "right_shoulder1", "right_shoulder2", "right_elbow",
        "left_shoulder1", "left_shoulder2", "left_elbow",
    ]
    AID = {n: model.actuator(n).id for n in actuator_names if cfg.enable_humanoid}

    def stand_targets() -> np.ndarray:
        """Default 'just standing' joint targets (radians)."""
        t = np.zeros(model.nu, dtype=np.float64)
        if not cfg.enable_humanoid:
            return t
        # Slight forward torso lean stabilizes the inverted pendulum.
        t[AID["abdomen_y"]] = -0.18
        # Tiny knee flex so the legs aren't locked straight (more natural
        # standing posture, helps absorb terrain impacts).
        t[AID["right_knee"]] = -0.20
        t[AID["left_knee"]]  = -0.20
        # Resting elbow bend (90° flex looks human and gives the arms
        # somewhere natural to sit while walking).
        t[AID["right_elbow"]] = -0.5
        t[AID["left_elbow"]]  = -0.5
        return t

    def walk_targets(phase: float, fwd: float, turn: float, sprint: bool) -> np.ndarray:
        """Joint targets for the current walk phase.

        phase: 0..2π, full stride cycle. Left foot strikes ground at π.
        fwd: -1 (back), 0 (idle), +1 (forward) — controls direction of stride.
        turn: -1, 0, +1 — abdomen yaw target for steering.
        sprint: bigger stride amplitude + faster cadence (cadence is in caller).
        """
        t = stand_targets()
        if not cfg.enable_humanoid:
            return t

        amp = (1.0 if sprint else 0.7) * (1.0 if fwd != 0 else 0.0)
        # Direction of stride: backward walking flips the leg/arm phase.
        dirn = fwd if fwd != 0 else 1.0

        sinP = math.sin(phase * dirn)
        cosP = math.cos(phase * dirn)

        # ── Hip flexion (forward/back leg swing). hip_y is the sagittal joint.
        # Range is roughly -1.9 to +0.35 rad. Keep ~0.45 rad amplitude max.
        hip_swing = 0.55 * amp
        t[AID["right_hip_y"]] = -hip_swing * cosP - 0.05  # bias forward
        t[AID["left_hip_y"]]  = +hip_swing * cosP - 0.05

        # ── Knee flexion: bend during swing phase, straight in stance.
        # Knee range is -2.79 to -0.035 (always negative = always flexed).
        knee_min = -0.10                # almost-straight knee in stance
        knee_max = -1.30 * amp - 0.10   # bent knee in swing
        # Right leg is in swing phase when sin > 0; left when sin < 0.
        right_swing = max(0.0, sinP) ** 2
        left_swing  = max(0.0, -sinP) ** 2
        t[AID["right_knee"]] = knee_min + (knee_max - knee_min) * right_swing
        t[AID["left_knee"]]  = knee_min + (knee_max - knee_min) * left_swing

        # ── Hip abduction (slight side-to-side tilt for balance) ──
        t[AID["right_hip_x"]] = +0.02
        t[AID["left_hip_x"]]  = -0.02

        # ── Arm swing (opposite to legs, around shoulder) ──
        arm_amp = 0.45 * amp
        t[AID["right_shoulder1"]] = +arm_amp * cosP
        t[AID["left_shoulder1"]]  = -arm_amp * cosP
        # Slight outward flare so arms don't clip the torso
        t[AID["right_shoulder2"]] = -0.20
        t[AID["left_shoulder2"]]  = +0.20

        # ── Elbow: fixed bend with extra during forward swing ──
        elbow_base = -0.55
        elbow_extra = -0.40 * amp
        t[AID["right_elbow"]] = elbow_base + elbow_extra * max(0.0, +cosP)
        t[AID["left_elbow"]]  = elbow_base + elbow_extra * max(0.0, -cosP)

        # ── Torso: counter-twist with the pelvis ──
        t[AID["abdomen_z"]] = -0.10 * sinP * amp + turn * 0.30
        # Forward lean (stronger during sprint) keeps the body from falling
        # backward as the legs propel forward.
        t[AID["abdomen_y"]] = -0.20 - 0.08 * (1.0 if sprint else 0.0) * amp

        return t

    # ── Keyboard input via pynput (real held-key state) ─────────────
    # MuJoCo's `key_callback` only fires on key PRESS, not release — so
    # held-key gameplay (press W to walk, release to stop) isn't possible
    # through the viewer alone. We fall back to OS-level keyboard input
    # via pynput, which gets both press AND release events. It runs in
    # its own thread, no GIL issues for this read pattern.
    intent = {"fwd": 0, "turn": 0, "sprint": False, "reset": False}
    pressed: set = set()
    try:
        from pynput import keyboard as pynput_kb  # type: ignore

        def _on_press(key):
            try:
                if hasattr(key, "char") and key.char:
                    pressed.add(key.char.lower())
                else:
                    pressed.add(str(key))
                    if key in (pynput_kb.Key.shift, pynput_kb.Key.shift_l, pynput_kb.Key.shift_r):
                        pressed.add("shift")
            except Exception:
                pass

        def _on_release(key):
            try:
                if hasattr(key, "char") and key.char:
                    pressed.discard(key.char.lower())
                else:
                    pressed.discard(str(key))
                    if key in (pynput_kb.Key.shift, pynput_kb.Key.shift_l, pynput_kb.Key.shift_r):
                        pressed.discard("shift")
            except Exception:
                pass

        kb_listener = pynput_kb.Listener(on_press=_on_press, on_release=_on_release)
        kb_listener.daemon = True
        kb_listener.start()
        print("  keyboard:    pynput (hold WASD)")
    except Exception as exc:  # pragma: no cover
        kb_listener = None
        print(f"  keyboard:    pynput unavailable ({exc}); keys won't work")

    # MuJoCo key_callback for non-toggle keys (R = reset).
    KEY_R = 82
    def on_key(keycode: int) -> None:
        if keycode == KEY_R:
            intent["reset"] = True

    def reset_humanoid() -> None:
        """Snap the humanoid back to a clean standing pose at spawn."""
        mujoco.mj_resetData(model, data)
        # Reposition the freejoint just above terrain at world origin.
        # qpos[0:3] = root position (x, y, z), qpos[3:7] = root quaternion.
        if cfg.enable_humanoid and model.nq >= 7:
            data.qpos[0] = 0.0
            data.qpos[1] = 0.0
            data.qpos[2] = spawn_z + 0.5     # extra clearance
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]   # identity quaternion

    # ── Camera initial pose: third-person tracking the torso ──────────
    # We use the model's "side" camera which is attached to the torso with
    # mode="trackcom" — it follows the body without us doing math. Press
    # Tab in the viewer to cycle to "front", or use mouse to free-orbit.
    # 0 = free camera, ≥1 = named cameras in the order they appear in MJCF.
    initial_cam_id = 0   # start in free-orbit so the user can move

    with mujoco.viewer.launch_passive(
        model, data, key_callback=on_key
    ) as viewer:
        viewer.cam.distance = 6.0
        viewer.cam.azimuth = 110.0
        viewer.cam.elevation = -15.0
        # Track the humanoid torso initially.
        if cfg.enable_humanoid:
            torso_id = model.body("torso").id
            viewer.cam.lookat[:] = data.xpos[torso_id]

        import time as _time

        # Walk phase state
        walk_phase = 0.0
        last_time = _time.time()
        target_dt = model.opt.timestep

        while viewer.is_running():
            # Read held-key state from pynput each frame
            if "w" in pressed and "s" not in pressed:   intent["fwd"] = +1
            elif "s" in pressed and "w" not in pressed: intent["fwd"] = -1
            else:                                        intent["fwd"] = 0
            if "a" in pressed and "d" not in pressed:   intent["turn"] = +1
            elif "d" in pressed and "a" not in pressed: intent["turn"] = -1
            else:                                        intent["turn"] = 0
            intent["sprint"] = "shift" in pressed

            # Reset on R press
            if intent["reset"]:
                reset_humanoid()
                walk_phase = 0.0
                intent["reset"] = False

            # ── Read pynput's pressed set into intent ────────────────
            # Arrows for movement (avoid MuJoCo's built-in letter toggles).
            # pynput represents arrow keys as "Key.up" etc. in str(key).
            up    = "Key.up"    in pressed
            down  = "Key.down"  in pressed
            left  = "Key.left"  in pressed
            right = "Key.right" in pressed
            intent["fwd"]    = (1 if up    else 0) - (1 if down  else 0)
            intent["turn"]   = (1 if left  else 0) - (1 if right else 0)
            intent["sprint"] = "shift" in pressed

            # Advance walk phase only while moving
            if cfg.enable_humanoid and intent["fwd"] != 0:
                cadence_hz = 1.4 if intent["sprint"] else 0.95
                walk_phase += target_dt * 2 * math.pi * cadence_hz

            # Compute & apply joint targets
            if cfg.enable_humanoid:
                tgt = walk_targets(
                    walk_phase,
                    intent["fwd"],
                    intent["turn"],
                    intent["sprint"],
                )
                data.ctrl[:] = tgt

                # ── Root motion (Unreal/Unity-style) ──────────────────
                # Directly translate the root freejoint's qpos AND zero the
                # corresponding qvel each frame. The qvel-zero is critical:
                # if the contact solver sees nonzero root velocity while the
                # feet are held by friction, it projects the velocity back
                # to zero — undoing our motion. Setting qvel to zero gives
                # the solver nothing to fight, and the qpos change sticks.
                # The legs still get simulated (real PD torques, real
                # contact, real gravity); only the root translation is
                # scripted. Standard pattern in MuJoCo character anim.
                if intent["fwd"] != 0:
                    speed = 2.6 if intent["sprint"] else 1.5
                    R = data.xmat[torso_id].reshape(3, 3)
                    fwd_world = R @ np.array([1.0, 0.0, 0.0])
                    data.qpos[0] += speed * intent["fwd"] * fwd_world[0] * target_dt
                    data.qpos[1] += speed * intent["fwd"] * fwd_world[1] * target_dt
                    data.qvel[0] = 0.0
                    data.qvel[1] = 0.0
                if intent["turn"] != 0:
                    # Rotate body yaw via quaternion. qpos[3:7] is (w,x,y,z).
                    yaw_rate = 1.0 * intent["turn"]
                    dyaw = yaw_rate * target_dt
                    ch, sh = math.cos(dyaw / 2), math.sin(dyaw / 2)
                    qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
                    data.qpos[3] = qw * ch - qz * sh
                    data.qpos[4] = qx * ch - qy * sh
                    data.qpos[5] = qx * sh + qy * ch
                    data.qpos[6] = qw * sh + qz * ch
                    data.qvel[5] = 0.0

            mujoco.mj_step(model, data)

            # Camera follow-com (only when in free camera mode)
            if cfg.enable_humanoid and viewer.cam.fixedcamid < 0:
                torso_pos = data.xpos[torso_id]
                # Smoothly track the torso. Keep camera distance/angle stable;
                # only the lookat point follows the body.
                lookat = viewer.cam.lookat
                blend = 0.12
                lookat[0] += (torso_pos[0] - lookat[0]) * blend
                lookat[1] += (torso_pos[1] - lookat[1]) * blend
                lookat[2] += (torso_pos[2] - lookat[2]) * blend

            viewer.sync()

            # Realtime pacing
            now = _time.time()
            elapsed = now - last_time
            if elapsed < target_dt:
                _time.sleep(target_dt - elapsed)
            last_time = _time.time()


def main_cli(terrain_obj: Path = None, no_humanoid: bool = False) -> None:
    """Entry point for `mars play`."""
    if terrain_obj is None:
        terrain_obj = Path("data/processed/terrain.obj")
    cfg = SimConfig(terrain_obj=terrain_obj, enable_humanoid=not no_humanoid)
    launch_interactive(cfg)
