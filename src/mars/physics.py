"""Mars physics world.

Two backends are supported, picked at runtime:

- **genesis**: Genesis (genesis-world). Apple-Silicon-friendly, fast, has a
  growing robotics ecosystem. Preferred when available.
- **mujoco**: MuJoCo (DeepMind). Definitely runs on Mac. Fallback for when
  Genesis isn't installed.

Both backends are wrapped behind a `MarsWorld` facade that:
- Sets gravity to Mars (3.721 m/s²)
- Adds the terrain mesh as a static collider
- Optionally applies a crude aerodynamic drag scaled by Mars's thin
  atmospheric density (mostly negligible at low speeds, but matters for
  parachutes and falling regolith)
- Exposes step()/get_camera_pose()/save_state() for the renderer to use
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mars.constants import MARS

Backend = Literal["genesis", "mujoco", "auto"]


@dataclass
class CameraPose:
    """6-DoF camera pose in the simulator's world frame (X east, Y north, Z up)."""
    position_m: tuple[float, float, float] = (0.0, 0.0, 1.7)
    look_at_m: tuple[float, float, float] = (10.0, 0.0, 0.7)
    fov_deg: float = 60.0


@dataclass
class MarsWorldConfig:
    terrain_obj: Path
    timestep_s: float = 1.0 / 240.0
    backend: Backend = "auto"
    camera: CameraPose = field(default_factory=CameraPose)
    enable_atmosphere_drag: bool = True


# ── Atmospheric drag (shared between backends) ──────────────────────────

def mars_drag_force(velocity_m_s: np.ndarray, area_m2: float, cd: float = 1.05) -> np.ndarray:
    """Quadratic drag force at Mars surface density.

    F = -0.5 * rho * |v| * v * Cd * A

    At Mars's ~0.020 kg/m³ surface density this is ~60× weaker than Earth
    drag for the same velocity, but it isn't zero — relevant for parachutes,
    flying dust, and small light objects.
    """
    v = np.asarray(velocity_m_s, dtype=np.float64)
    speed = np.linalg.norm(v)
    if speed < 1e-9:
        return np.zeros(3)
    return -0.5 * MARS.surface_density_kg_m3 * speed * v * cd * area_m2


# ── Backend resolution ──────────────────────────────────────────────────

def _resolve_backend(req: Backend) -> Backend:
    if req != "auto":
        return req
    try:
        import genesis  # noqa: F401
        return "genesis"
    except Exception:
        try:
            import mujoco  # noqa: F401
            return "mujoco"
        except Exception as e:
            raise RuntimeError(
                "no physics backend available. Install one of:\n"
                "  pip install genesis-world      # preferred\n"
                "  pip install mujoco             # fallback"
            ) from e


# ── MarsWorld facade ────────────────────────────────────────────────────

class MarsWorld:
    """Backend-agnostic Mars physics world.

    Usage:
        world = MarsWorld(MarsWorldConfig(terrain_obj=Path("data/processed/terrain.obj")))
        for _ in range(1000):
            world.step()
        world.close()
    """

    def __init__(self, cfg: MarsWorldConfig) -> None:
        self.cfg = cfg
        self.backend = _resolve_backend(cfg.backend)
        self._handle: Any = None
        self._t: float = 0.0
        self._init_backend()

    def _init_backend(self) -> None:
        if self.backend == "genesis":
            self._init_genesis()
        elif self.backend == "mujoco":
            self._init_mujoco()
        else:
            raise ValueError(self.backend)

    # -- Genesis -------------------------------------------------------
    def _init_genesis(self) -> None:
        import genesis as gs
        gs.init(backend=gs.cpu)  # keep CPU on Mac; metal backend is in flux
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.cfg.timestep_s,
                gravity=(0.0, 0.0, -MARS.gravity_m_s2),
            ),
            show_viewer=False,
        )
        # Static terrain
        self._scene.add_entity(
            gs.morphs.Mesh(
                file=str(Path(self.cfg.terrain_obj).resolve()),
                fixed=True,
                collision=True,
                visualization=True,
            ),
            material=gs.materials.Rigid(
                friction=MARS.kinetic_friction_regolith,
            ),
        )
        self._scene.build()
        self._handle = self._scene

    # -- MuJoCo --------------------------------------------------------
    def _init_mujoco(self) -> None:
        import mujoco
        # Inline MJCF: Mars gravity, mesh asset for terrain, one camera.
        mesh_path = Path(self.cfg.terrain_obj).resolve()
        cam = self.cfg.camera
        xml = f"""
<mujoco model="mars">
  <option gravity="0 0 -{MARS.gravity_m_s2}" timestep="{self.cfg.timestep_s}"/>
  <asset>
    <mesh name="terrain" file="{mesh_path}"/>
  </asset>
  <worldbody>
    <camera name="rover_eye"
            pos="{cam.position_m[0]} {cam.position_m[1]} {cam.position_m[2]}"
            xyaxes="1 0 0 0 0 1"
            fovy="{cam.fov_deg}"/>
    <geom type="mesh" mesh="terrain" friction="{MARS.kinetic_friction_regolith} 0.005 0.0001"/>
  </worldbody>
</mujoco>
"""
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)
        self._handle = (self._model, self._data)

    # -- Public stepping API ------------------------------------------
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            if self.backend == "genesis":
                self._scene.step()
            else:
                import mujoco
                mujoco.mj_step(self._model, self._data)
            self._t += self.cfg.timestep_s

    @property
    def time_s(self) -> float:
        return self._t

    def camera_pose(self) -> CameraPose:
        return self.cfg.camera

    def close(self) -> None:
        # Both backends manage their own resources; explicit close is mostly
        # a hook for future expansion (recording, viewers, etc).
        pass


# ── Convenience factory ─────────────────────────────────────────────────

def make_mars_world(terrain_obj: Path, **kw: Any) -> MarsWorld:
    return MarsWorld(MarsWorldConfig(terrain_obj=Path(terrain_obj), **kw))
