"""Build a Mars terrain mesh.

Examples:
    # synthetic Mars terrain (no download, fast smoke test)
    python scripts/01_build_terrain.py

    # real Jezero Crater HiRISE DTM (large download)
    python scripts/01_build_terrain.py --site jezero
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from mars.terrain import HIRISE_SITES, build_terrain

app = typer.Typer(add_completion=False, help="Mars terrain builder")


@app.command()
def main(
    site: str = typer.Option(
        None,
        "--site",
        help=f"HiRISE site key. Options: {list(HIRISE_SITES)}. "
             "Omit for a synthetic Mars heightmap.",
    ),
    out: Path = typer.Option(
        Path("data/processed/terrain.obj"),
        "--out",
        help="Output OBJ path.",
    ),
    size: int = typer.Option(256, "--size", help="Synthetic heightmap size (px)."),
    seed: int = typer.Option(0, "--seed", help="Synthetic seed."),
    m_per_px: float = typer.Option(
        1.0, "--m-per-px", help="Horizontal meters per pixel."
    ),
):
    print(f"[bold]Building terrain → {out}[/bold]")
    if site:
        if site not in HIRISE_SITES:
            raise typer.BadParameter(f"unknown site '{site}'")
        info = HIRISE_SITES[site]
        print(f"  site = [cyan]{info.name}[/cyan]: {info.description}")
    else:
        print(f"  mode = [yellow]synthetic[/yellow] ({size}×{size}, seed={seed})")

    path = build_terrain(
        site=site,
        out_path=out,
        synthetic_size=size,
        synthetic_seed=seed,
        horizontal_m_per_px=m_per_px,
    )
    print(f"[green]✓ wrote {path}[/green]")


if __name__ == "__main__":
    app()
