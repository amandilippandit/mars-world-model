# MarsWorldModel

A Mars-environment world model for Physical AI and humanoid robot pretraining.

## Goal
Build the simulated Mars environment that humanoid robots will train in *before* they're deployed for civilization-building on Mars. Uses real NASA terrain data, Mars-accurate physics (gravity, thin atmosphere, dust), and a pluggable photoreal rendering layer.

## Architecture

```
   NASA HiRISE / MOLA DEMs
            ↓
   [terrain.py]  →  Mars heightmap mesh
            ↓
   [scene.py]    →  Blender scene (lighting, materials, atmosphere)
            ↓
   [physics.py]  →  Genesis sim (3.71 m/s² gravity, 600 Pa atmosphere)
            ↓
   [render.py]   →  Photoreal frames
                       ├─ local: Blender Cycles
                       ├─ local: Wan 2.1 1.3B (MPS)
                       └─ cloud: Colab/Kaggle/fal.ai/NVIDIA Cosmos API
```

## Hardware target
MacBook M2 Air, 16 GB RAM, no discrete GPU. The heavy AI rendering step runs on free cloud GPUs (Google Colab T4, Kaggle P100). Everything else runs locally.

## Status
Early scaffold. See [todos](#todos) below.

## Project layout

```
MarsWorldModel/
├── pyproject.toml          Python deps (uv-compatible)
├── src/mars/
│   ├── constants.py        Mars physics constants
│   ├── terrain.py          DEM → mesh pipeline
│   ├── scene.py            Blender scene composer
│   ├── physics.py          Genesis Mars world
│   └── render.py           Pluggable renderer interface
├── scripts/                Runnable entry points
├── notebooks/              Jupyter / Colab notebooks
├── data/
│   ├── raw/                Downloaded NASA DEMs
│   ├── processed/          Generated meshes
│   └── renders/            Output frames
└── tests/
```

## Setup

```bash
# Recommended: use uv (faster, simpler)
brew install uv
uv sync

# Or with pip
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Blender must be installed separately (the Python `bpy` module is bundled inside Blender).

```bash
brew install --cask blender
```

## Data sources
- **NASA HiRISE DTMs** — https://www.uahirise.org/dtm/
- **MOLA global elevation** — https://pds-geosciences.wustl.edu/missions/mgs/megdr.html
- **Mars Trek** (tile server) — https://trek.nasa.gov/mars/

## License
MIT
