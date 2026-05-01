# Mars World Model

> A simulated Mars environment for pretraining humanoid robots before they ship to Mars.

[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![mujoco](https://img.shields.io/badge/physics-MuJoCo%203.8-orange)](https://mujoco.org/)
[![three](https://img.shields.io/badge/viewer-three.js%200.167-black)](https://threejs.org/)

Mars World Model is an end-to-end pipeline that builds a physically faithful Mars environment from procedural terrain and NASA DEM data, simulates it with MuJoCo at real Mars gravity, and exposes it through two interactive viewers — a native physics window and a browser-based PBR scene. The goal is a self-contained playground for training humanoid robot policies in Mars-relevant conditions, well before any humanoid actually goes there.

The whole stack runs on a 16 GB MacBook Air. Heavy generative-rendering work optionally offloads to free Colab T4 / Kaggle P100 GPUs.

---

## Why

Putting humanoid robots on Mars means training policies in conditions Earth can't reproduce: 38% gravity, ~0.6% atmospheric density, fine cohesive regolith, harsh low-angle sunlight, dust-scattered shadows. None of those map onto an Earth simulator's defaults. This repo is the substrate to fix that — an evolving Mars environment with the right physics constants baked in, real Mars topography from NASA, and a humanoid that walks under all of them.

It's also a research scaffold. The renderer is pluggable so you can iterate from MuJoCo's basic OpenGL up to Blender Cycles offline frames or AI-photoreal video via NVIDIA Cosmos / Wan 2.1, sharing the same physics + camera trajectories.

---

## Architecture

```
                       ┌─────────────────────────────────────┐
                       │  Procedural / NASA HiRISE terrain   │
                       │  (heightmap + metadata + textures)   │
                       └──────────────────┬──────────────────┘
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  │                       │                       │
        ┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
        │  MuJoCo physics   │  │  Three.js viewer  │  │  Offline render   │
        │  (mars play)      │  │  (mars serve)     │  │  Blender / Cosmos │
        │                   │  │                   │  │                   │
        │  • Mars gravity   │  │  • PBR materials  │  │  • Cinema-quality │
        │    3.721 m/s²     │  │  • ACES tone-map  │  │    photoreal      │
        │  • Heightfield    │  │  • Real shadows   │  │  • Same heightmap │
        │  • Atmospheric    │  │  • Atmospheric    │  │    + camera path  │
        │    drag           │  │    haze           │  │    as the sim     │
        │  • Position-PD    │  │  • 12k instanced  │  │                   │
        │    humanoid       │  │    rocks          │  │                   │
        │  • Walk cycle     │  │  • Third-person   │  │                   │
        │    controller     │  │    astronaut      │  │                   │
        └───────────────────┘  └───────────────────┘  └───────────────────┘
```

The procedural terrain stage produces three sibling files used by every downstream consumer:

| File | Consumer | Purpose |
|---|---|---|
| `terrain.png` | MuJoCo `<hfield>` | 8-bit heightfield for collision |
| `terrain.bin` | Three.js viewer | Float32 heightmap for browser |
| `terrain.json` | both | Metadata — extent, m/cell, z range |
| `terrain.npy` | Python | Float32 array for spawn lookups, AO baking |

---

## Quick start

```bash
git clone https://github.com/amandilippandit/mars-world-model.git
cd mars-world-model

# Editable install (so `mars` is on the path; mjpython needs it too)
python3 -m pip install -e .
```

### 1. Build the terrain

```bash
mars terrain --size 1024 --seed 7
```

Generates a 1 km × 1 km Mars surface — gentle walkable foreground, one prominent asymmetric mountain massif (Mt-Sharp-style) on the horizon, secondary distant mesas, sinusoidal dune fields with steep-lee/gentle-windward profiles, sparse eroded craters, scattered surface rocks. Writes the `terrain.{png,bin,json,npy,obj}` quintuple to `data/processed/`.

Other styles available:

```bash
mars terrain --style highland   # rolling rocky hills
mars terrain --style badlands   # dramatic Valles-Marineris-like canyons
mars terrain --site jezero      # real NASA HiRISE DEM of Perseverance landing site
```

### 2. Walk a humanoid on it (MuJoCo)

```bash
mjpython -m mars.cli play
```

Opens MuJoCo's native viewer with:
- Real Mars gravity (3.721 m/s²) and atmospheric density (0.020 kg/m³)
- 17-DoF humanoid driven by position-PD actuators
- A scripted walk-cycle controller (phase-driven hip/knee/ankle/arm coordination)
- ~680 visual rock decorations (boulders, rocks, pebbles) at varied scales
- Mars-warm sun, butterscotch sky, atmospheric haze
- Auto-tracking free camera

Controls:

| Key | Action |
|---|---|
| ↑ / ↓ | walk forward / back |
| ← / → | turn left / right |
| Shift | sprint (held) |
| R | reset humanoid |
| Mouse | orbit camera |

> **Note:** macOS requires `mjpython` (not `python3`) for `launch_passive`. It ships with the `mujoco` pip package — already on your `PATH` after install.

### 3. Browse the PBR web viewer

```bash
mars serve
```

Opens `http://localhost:8765/viewer/` automatically — a Three.js scene with:
- PBR `MeshStandardMaterial` with normal-mapped regolith
- ACES Filmic tone mapping + bloom + atmospheric fog
- 12,600 GPU-instanced rocks (boulders + rocks + pebbles)
- Vertex-baked ambient occlusion in crevices and crater rims
- PMREM environment map for realistic PBR reflections
- A third-person astronaut with biomechanically-driven gait — hip swing, knee flexion timed to swing phase, pelvis Trendelenburg drop, spine counter-twist, arm pump with elbow bend, idle breathing, sprint forward lean, airborne tuck pose

Controls:

| Key | Action |
|---|---|
| W A S D | walk |
| Shift | sprint |
| Space | jump (Mars 1.2 m hop, 2.6× higher than Earth would be) |
| Mouse | orbit camera around astronaut |
| Scroll | zoom |
| Esc | release pointer lock |

### 4. Bake assets

```bash
# Procedural sandy regolith texture + matching normal map
python scripts/bake_regolith_texture.py

# Or download a real NASA Curiosity rover photo and use it as the regolith
python scripts/download_nasa_regolith.py
```

---

## Mars physics constants

Encoded in [`src/mars/constants.py`](src/mars/constants.py) and used throughout:

| Quantity | Value | Source |
|---|---|---|
| Surface gravity | 3.721 m/s² | NASA Mars Fact Sheet |
| Atmospheric density (surface, mean) | 0.020 kg/m³ | Mars Climate Database |
| Surface pressure | 610 Pa (~0.6% of Earth) | Viking & MSL surface measurements |
| Mean surface temperature | 210 K | derived |
| Speed of sound (surface) | 240 m/s | thin CO₂ atmosphere |
| Solar irradiance (surface, mean) | 590 W/m² (~43% of Earth) | TOA × atmospheric attenuation |
| Sun angular diameter | 21 arcmin (~⅔ of Earth's) | mean orbital distance 1.524 AU |
| Sky color (zenith) | RGB (0.83, 0.50, 0.30) | Perseverance Mastcam-Z white-balanced |
| Regolith friction (kinetic) | 0.55 | Apollo-era / MER calibrations |
| Sol length | 88,775.244 s | 24h 39m 35.244s |

Jump physics work out to ~2.6× Earth's at the same leg-push impulse, which the walk controller uses to time foot lift.

---

## Project layout

```
mars-world-model/
├── pyproject.toml                project + dependencies
├── src/mars/
│   ├── constants.py              Mars physics + appearance constants
│   ├── terrain.py                Procedural Mars + NASA HiRISE DEM loader
│   ├── sim.py                    MuJoCo MJCF composer + walk controller
│   ├── scene.py                  Blender scene composer (offline Cycles)
│   ├── physics.py                Genesis/MuJoCo backend facade
│   ├── render.py                 Pluggable renderer interface
│   ├── cli.py                    `mars terrain | spec | play | serve | render`
│   └── assets/
│       ├── humanoid.xml          17-DoF MJCF humanoid (position actuators)
│       ├── regolith.png          Baked Mars regolith texture (2048²)
│       └── regolith_normal.png   Matching normal map
├── viewer/                       Three.js + PBR web viewer
├── scripts/                      One-off pipeline scripts
├── notebooks/                    Wan 2.1 Colab GPU pipeline (free)
├── tests/                        pytest unit tests
└── data/
    ├── raw/                      Downloaded NASA / Curiosity imagery
    ├── processed/                Built terrain heightmaps
    └── renders/                  Output frames + previews
```

---

## Terrain generator

The procedural terrain has three styles:

- **plain** *(default)* — Gentle rolling regolith with one prominent asymmetric mountain massif, 1–2 distant mesas, scattered eroded craters, and a real wind-aligned dune field. Modeled on Gale Crater rover panoramas. Walkable foreground (median slope 6°, 40%+ cells flat).
- **highland** — Rolling hills with rocky patches, moderate slopes, small craters. In-between option.
- **badlands** — Dramatic Valles-Marineris-style canyons and ridges with thermal erosion + valley flattening. Heavy relief.

All styles share the same noise primitives:

| Primitive | Purpose |
|---|---|
| `_fbm` | Fractal Brownian motion — base relief, smooth dunes |
| `_ridged_multifractal` | Musgrave's ridged noise — sharp canyon networks, mountain ridges |
| `_domain_warp` | Bend straight ridges into organic curves |
| `_thermal_erosion` | Move material from steep slopes downhill (rounds knife-edges) |
| `_natural_mountain` | Asymmetric envelope × FBM × ridged-noise — naturalistic peaks |
| `_dune_field` | Multi-wavelength sinusoidal dunes with steep-lee/gentle-windward profiles |
| `_add_craters` | Bowl + raised lip with eroded edges |

You can also load real Mars elevation by `mars terrain --site jezero` (or `gale`), which downloads HiRISE stereo-derived DTMs from the U Arizona PDS and uses the actual NASA topography.

---

## MuJoCo walk controller

The humanoid is 17-DoF with all joints driven by position actuators (built-in PD). The Python loop in [`sim.py`](src/mars/sim.py) computes a target joint configuration each timestep based on a single `walk_phase` variable that advances at 0.95 cycles/s (walking) or 1.4 cycles/s (sprinting):

| Joint | Target (radians) |
|---|---|
| Hip Y (sagittal) | `−0.55 · cos(phase)` (forward/back swing) |
| Knee | base flex + `1.55 · max(0, sin(phase))²` (bends only during swing) |
| Ankle | `−0.32 · sin(phase)` (heel-toe) |
| Pelvis Z | `−0.10 · sin(phase)` (Trendelenburg tilt) |
| Spine Y | counter-twist + `−0.10 · sin(phase) · amp` |
| Spine X | forward lean: `0` standing → `−0.20` walking → `−0.28` sprinting |
| Shoulder Y | arm pump opposite to legs |
| Elbow | base flex + extra during forward swing |

Forward propulsion uses the standard "scripted-root, simulated-limbs" pattern: the freejoint's `qpos[0:3]` is translated each frame at intent-determined velocity while `qvel[0:6]` is zeroed so the constraint solver doesn't try to project the root motion away. The limbs cycle through real PD physics — they push against the heightfield, contact normals resolve, the humanoid responds to terrain bumps.

Real PPO-trained walking is out of scope today but the scaffold is policy-compatible: every joint has a `<position>` actuator, the model exposes COM/qpos/qvel observations, and the heightfield surface is fully physics-traversable.

---

## Three.js viewer

Browser-based PBR renderer. Loads the same heightmap (`terrain.bin`) used by the physics sim, plus the baked regolith albedo + normal map.

Features:
- **`MeshStandardMaterial`** with normal mapping, env-map reflections, ACES tone mapping
- **PMREM** environment map generated from the sky-shader sphere — gives PBR materials realistic ambient lighting
- **Vertex AO** baked from heightmap (sample 24 surrounding ring positions per vertex; cells in cavities get darker)
- **Bloom + HDR pipeline** (RenderPass → UnrealBloomPass → OutputPass)
- **GPU-instanced rocks** — 12,600 rock meshes across 12 InstancedMesh batches → ~12 draw calls regardless of count
- **Atmospheric haze** via FogExp2 with horizon-color blend
- **Hierarchical astronaut** built from primitives: chest + abdomen + pelvis + neck + head + helmet visor + 17 articulated joints, animated with a multi-component biomechanical walk cycle

---

## Hardware target

Built and tuned for **MacBook M2 Air, 16 GB RAM, no discrete GPU**. Both viewers run at 60 fps. The MuJoCo sim runs at ~37× realtime even with the full rock decoration scene. Heavier generative rendering (Wan 2.1, NVIDIA Cosmos) optionally offloads to free Colab T4 / Kaggle P100 GPUs via the [`notebooks/wan_colab_render.ipynb`](notebooks/wan_colab_render.ipynb) pipeline.

---

## CLI reference

```
mars terrain   build a Mars heightmap (synthetic or NASA HiRISE)
mars play      open the MuJoCo physics sim with walk controller
mars serve     start the Three.js + PBR web viewer at localhost:8765
mars spec      write a default SceneSpec for offline rendering
mars render    render a SceneSpec via Blender Cycles or Wan 2.1 (Colab)
```

Run any command with `--help` for full options.

---

## Roadmap

- [x] Procedural terrain (3 styles) + NASA HiRISE loader
- [x] MuJoCo physics sim with Mars-correct constants
- [x] Position-actuator humanoid + scripted walk controller
- [x] Three.js + PBR third-person browser viewer
- [x] GPU-instanced rocks at varied scales
- [x] Vertex AO + PMREM env maps + ACES tone mapping
- [ ] PPO-trained walking policy (MJX + JAX)
- [ ] Action-conditioned generative world model (Wan 2.1 LoRA fine-tune on Mars rollouts)
- [ ] Mars rover physics (suspension, wheel slip)
- [ ] Multi-humanoid scenarios (construction, surveying)
- [ ] Real Jezero / Gale integration with rover-eye camera traversal
- [ ] Cosmos-Predict integration for photoreal video output

---

## Data sources

- **NASA HiRISE DTMs** — high-resolution stereo-derived elevation (Jezero, Gale, etc.) — https://www.uahirise.org/dtm/
- **MOLA global elevation** — 463 m/px topographic mosaic — https://pds-geosciences.wustl.edu/missions/mgs/megdr.html
- **NASA Mars Trek tile server** — global imagery + elevation tiles — https://trek.nasa.gov/mars/
- **NASA Image Library** — Curiosity / Perseverance rover photos — https://images.nasa.gov/

---

## License

MIT — see [LICENSE](LICENSE).

The repository includes baked assets (`src/mars/assets/regolith.png`, `regolith_normal.png`) generated procedurally from public-domain NASA imagery and Mars constants. NASA imagery is itself public domain and may be redistributed.
