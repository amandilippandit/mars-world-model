"""Top-level Mars CLI: glues terrain → scene → render together.

    mars terrain                    # build synthetic Mars terrain
    mars terrain --site jezero      # download Jezero HiRISE DTM
    mars spec                       # write a default SceneSpec
    mars render --renderer blender  # full local render
    mars render --renderer wan_colab  # package spec for Colab upload
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from mars.render import CameraKeyframe, SceneSpec, get_renderer
from mars.sim import SimConfig, launch_interactive
from mars.terrain import HIRISE_SITES, build_terrain

app = typer.Typer(add_completion=False, help="Mars World Model CLI")


@app.command()
def terrain(
    site: str = typer.Option(None, "--site", help=f"HiRISE site: {list(HIRISE_SITES)}"),
    out: Path = typer.Option(Path("data/processed/terrain.obj"), "--out"),
    size: int = typer.Option(1024, "--size",
        help="Heightmap resolution. With 1m/cell, 1024 = 1 km × 1 km terrain."),
    seed: int = typer.Option(0, "--seed"),
    style: str = typer.Option(
        "plain", "--style",
        help="Synthetic style: plain | highland | badlands. "
             "plain = realistic Jezero-style rolling regolith with craters and "
             "wind ripples (default — what rovers actually operate on). "
             "highland = rolling hills with rocky patches. "
             "badlands = dramatic Valles-Marineris-like canyons (rare on Mars).",
    ),
):
    """Build a Mars terrain mesh (HiRISE site or synthetic)."""
    print(f"[bold]building terrain → {out}[/bold]  style=[cyan]{style}[/cyan]")
    p = build_terrain(
        site=site, out_path=out,
        synthetic_size=size, synthetic_seed=seed, synthetic_style=style,
    )
    print(f"[green]✓[/green] {p}")


@app.command()
def spec(
    terrain_obj: Path = typer.Option(Path("data/processed/terrain.obj"), "--terrain"),
    out: Path = typer.Option(Path("data/renders/run01/spec.json"), "--out"),
    duration: float = typer.Option(4.0, "--duration"),
    prompt: str = typer.Option(
        "Mars surface, butterscotch sky, regolith terrain, low-angle sun, "
        "thin dusty atmosphere, slow forward dolly, photorealistic, cinematic, 4K",
        "--prompt",
    ),
):
    """Write a default SceneSpec for the given terrain."""
    s = SceneSpec(
        terrain_obj=terrain_obj,
        duration_s=duration,
        text_prompt=prompt,
        camera_keyframes=[
            CameraKeyframe(t_s=0.0, position_m=(0.0, 0.0, 1.7), look_at_m=(10.0, 0.0, 1.0)),
            CameraKeyframe(t_s=duration, position_m=(8.0, 0.0, 1.7), look_at_m=(20.0, 0.0, 1.0)),
        ],
    )
    s.save(out)
    print(f"[green]✓[/green] spec → {out}")


@app.command()
def render(
    spec_path: Path = typer.Option(Path("data/renders/run01/spec.json"), "--spec"),
    out: Path = typer.Option(Path("data/renders/run01"), "--out"),
    renderer: str = typer.Option("blender", "--renderer", help="blender | wan_colab"),
):
    """Render a SceneSpec with the chosen backend."""
    s = SceneSpec.load(spec_path)
    r = get_renderer(renderer)
    paths = r.render(s, out)
    print(f"[green]✓[/green] {len(paths)} output(s) in {out}")


@app.command()
def play(
    terrain_obj: Path = typer.Option(Path("data/processed/terrain.obj"), "--terrain"),
    no_humanoid: bool = typer.Option(False, "--no-humanoid", help="Empty Mars (no robot)."),
):
    """Open the interactive Mars sim in a native window (humanoid + physics)."""
    print("[bold]launching Mars sim[/bold]")
    launch_interactive(SimConfig(terrain_obj=terrain_obj, enable_humanoid=not no_humanoid))


@app.command()
def serve(
    port: int = typer.Option(8765, "--port"),
    no_open: bool = typer.Option(False, "--no-open", help="Don't auto-open the browser."),
):
    """Run the Three.js Mars viewer at http://localhost:<port>/viewer/.

    Serves the project directory over HTTP so the browser can load the
    terrain heightmap (.bin), metadata (.json), regolith textures, and
    the viewer JS. PBR materials, ACES tone mapping, normal-mapped
    surface, real shadows, FPS controls — runs on the browser's GPU.
    """
    import http.server
    import socketserver
    import threading
    import webbrowser
    from functools import partial

    repo_root = Path(__file__).resolve().parent.parent.parent
    handler_cls = partial(http.server.SimpleHTTPRequestHandler, directory=str(repo_root))

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler_cls) as httpd:
        url = f"http://localhost:{port}/viewer/"
        print(f"[bold]Mars viewer[/bold] → [cyan]{url}[/cyan]")
        print(f"  serving from {repo_root}")
        print("  Ctrl+C to stop")
        if not no_open:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  stopping…")


if __name__ == "__main__":
    app()
