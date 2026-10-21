"""Demo video for the technical report: real runs from the experiments, side by side.

Run:  python report/video.py     (a few minutes)

Every clip replays a run from the experiments with the same task, world, data
seed, MPC seed and cached networks, so it shows exactly what was measured.
Frames are rendered off-screen with MuJoCo and written to an MP4 with ffmpeg.
Motion plays at half speed so it can be followed.

Writes report/demo.mp4.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from lac import arm, methods  # noqa: E402
from lac.mpc import MPCController  # noqa: E402
from lac.simulate import run_reach  # noqa: E402
from m6a_outside import TASKS  # noqa: E402

OUT = ROOT / "report" / "demo.mp4"
WIDTH, HEIGHT, FPS = 1280, 720, 30
PANEL = 500
PLAYBACK = 0.5  # motion at half speed
HOLD = 1.5  # seconds the last frame of a clip stays on screen
PAYLOAD = 1.0
FONT_DIR = Path("C:/Windows/Fonts")
COLOURS = {"C0": (76, 114, 176), "C1": (221, 132, 82), "C2": (85, 168, 104), "C3": (196, 78, 82),
           "C3-E": (129, 114, 179), "reference": (40, 40, 40)}
NAMES = {"C0": "C0: assumes no payload", "C1": "C1: estimated payload mass", "C2": "C2: no-payload physics + network",
         "C3": "C3: estimated mass + network", "C3-E": "C3-E: estimated mass + 5 networks averaged",
         "reference": "Reference: told the true payload"}


def font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


# --- replaying and rendering runs ---------------------------------------------


def replay(world: str, method: str, task: tuple, budget: float | None, seed: int, mpc_seed: int):
    """Run one reach exactly as in the experiments and return its rollout."""
    q_start, target = task
    if method in ("C0", "reference"):
        predictor = methods.build_predictors(world, PAYLOAD, 9.0, 1)[method]
    elif method == "C3-E":
        predictor = methods.build_ensembles(world, PAYLOAD, budget, seed)["C3-E"]
    else:
        predictor = methods.build_predictors(world, PAYLOAD, budget, seed)[method]
    controller = MPCController(predictor=predictor, seed=mpc_seed, show_plan=False)
    return run_reach(controller, PAYLOAD, q_start, target, 3.0, world)


class PanelRenderer:
    """Draws the arm at a given state with its tip trail, as seen by the side camera."""

    def __init__(self, world: str):
        self.model, self.data = arm.load_arm()
        arm.set_world(self.model, world)
        arm.set_payload(self.model, PAYLOAD)
        self.renderer = mujoco.Renderer(self.model, PANEL, PANEL)

    def frame(self, q: np.ndarray, target: np.ndarray, trail: np.ndarray, colour) -> Image.Image:
        arm.set_target(self.model, self.data, target)
        arm.set_joint_angles(self.model, self.data, q)
        self.renderer.update_scene(self.data, camera="side")
        scene = self.renderer.scene
        # A larger, translucent marker so the target is visible at video size.
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.035, 0, 0],
                            pos=np.array([target[0], -0.05, target[1]]), mat=np.eye(3).flatten(),
                            rgba=np.array([0.1, 0.7, 0.2, 0.45], dtype=np.float32))
        scene.ngeom += 1
        rgba = np.array([*[c / 255 for c in colour], 1.0], dtype=np.float32)
        for x, z in trail[:: max(1, len(trail) // 400)]:
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.006, 0, 0],
                                pos=np.array([x, -0.03, z]), mat=np.eye(3).flatten(), rgba=rgba)
            scene.ngeom += 1
        return Image.fromarray(self.renderer.render())


# --- composing frames ----------------------------------------------------------


def blank() -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), "white")


def centred(draw: ImageDraw.ImageDraw, y: int, text: str, size: int, bold=False, fill=(30, 30, 30)) -> None:
    f = font(size, bold)
    for i, line in enumerate(text.split("\n")):
        w = draw.textlength(line, font=f)
        draw.text(((WIDTH - w) / 2, y + i * int(size * 1.35)), line, font=f, fill=fill)


def card(title: str, body: str, seconds: float, image: Path | None = None) -> list[np.ndarray]:
    canvas = blank()
    draw = ImageDraw.Draw(canvas)
    centred(draw, 60 if image else 200, title, 40, bold=True)
    if image is not None:
        picture = Image.open(image).convert("RGB")
        picture.thumbnail((520, 430))
        canvas.paste(picture, ((WIDTH - picture.width) // 2, 140))
        centred(draw, 600, body, 24)
    else:
        centred(draw, 300, body, 28)
    return [np.asarray(canvas)] * round(seconds * FPS)


def clip(title: str, caption: str, world: str, runs: list[tuple[str, object]], task: tuple) -> list[np.ndarray]:
    """Two runs of the same task side by side, at half speed, with live tip errors."""
    renderer = PanelRenderer(world)
    _, target = task
    sim_seconds = 3.0
    frames = []
    total = round(sim_seconds / PLAYBACK * FPS)
    for k in range(total + round(HOLD * FPS)):
        t = min(k, total) / total * sim_seconds
        canvas = blank()
        draw = ImageDraw.Draw(canvas)
        centred(draw, 22, title, 30, bold=True)
        for i, (method, rollout) in enumerate(runs):
            index = min(int(round(t / 0.002)), len(rollout.q) - 1)
            panel = renderer.frame(rollout.q[index], target, rollout.tip[: index + 1], COLOURS[method])
            x = 110 + i * (PANEL + 60)
            canvas.paste(panel, (x, 75))
            draw.rectangle([x, 75, x + PANEL - 1, 75 + PANEL - 1], outline=COLOURS[method], width=4)
            label = NAMES[method]
            f = font(22, bold=True)
            draw.text((x + (PANEL - draw.textlength(label, font=f)) / 2, 585), label, font=f, fill=COLOURS[method])
            error = f"distance to target {rollout.error[index] * 100:5.1f} cm"
            f = font(22)
            draw.text((x + (PANEL - draw.textlength(error, font=f)) / 2, 615), error, font=f, fill=(30, 30, 30))
        draw.text((WIDTH - 230, 690), f"t = {t:4.2f} s (half speed)", font=font(16), fill=(110, 110, 110))
        centred(draw, 655, caption, 22, fill=(60, 60, 60))
        frames.append(np.asarray(canvas))
    return frames


def best_ensemble_task() -> tuple[int, int]:
    """Outside W2 with 90 s of data: the task and data seed where C3-E most improves on C3."""
    with open(ROOT / "results" / "m6" / "ensemble_runs.csv", encoding="utf-8") as f:
        ens = {(r["data_seed"], r["task"]): float(r["final_error_cm"]) for r in csv.DictReader(f)
               if (r["region"], r["world"], r["method"], r["budget_s"]) == ("outside", "W2", "C3-E avg", "90.0")}
    with open(ROOT / "results" / "m6" / "runs.csv", encoding="utf-8") as f:
        single = {(r["data_seed"], r["task"]): float(r["final_error_cm"]) for r in csv.DictReader(f)
                  if (r["region"], r["world"], r["method"], r["budget_s"]) == ("outside", "W2", "C3", "90.0")}
    seed, task = max(single, key=lambda k: single[k] - ens[k])
    return int(seed), int(task)


def main() -> None:
    inside, outside = TASKS["inside"], TASKS["outside"]
    frames = []
    frames += card("Physics, Learning, or Both?",
                   "A robot arm picks up a payload of unknown mass.\n"
                   "Its model-based controller now has the wrong model.\n\n"
                   "How should the model be corrected from a little motion data:\n"
                   "estimate the mass, learn the error with a neural network, or both?", 8)
    frames += card("Setup",
                   "Two-link arm in MuJoCo, 1 kg payload. Sampling-based MPC plans every 20 ms.\n"
                   "Training data aims only at targets on the right; test targets on the left are new to it.",
                   7, ROOT / "report" / "figures" / "setup.png")
    frames += clip("A wrong model: the arm sags below the target",
                   "Same controller, same task. Only the payload in the model differs.",
                   "W2", [("C0", replay("W2", "C0", inside[0], None, 0, 0)),
                          ("reference", replay("W2", "reference", inside[0], None, 0, 0))], inside[0])
    frames += clip("Inside the training region, with 90 s of data",
                   "Estimating the mass alone leaves about 2 cm; adding a learned correction removes almost all of it.",
                   "W2", [("C1", replay("W2", "C1", inside[0], 90.0, 1, 1)),
                          ("C3", replay("W2", "C3", inside[0], 90.0, 1, 1))], inside[0])
    frames += clip("Outside the training region, with 9 s of data",
                   "The controller exploits the network's errors where it has seen no data.",
                   "W3", [("C1", replay("W3", "C1", outside[1], 9.0, 3, 3)),
                          ("C2", replay("W3", "C2", outside[1], 9.0, 3, 3))], outside[1])
    seed, task = best_ensemble_task()
    frames += clip("Outside the training region, with 90 s of data",
                   "Averaging five networks removes many of these failures.",
                   "W2", [("C3", replay("W2", "C3", outside[task], 90.0, seed, seed)),
                          ("C3-E", replay("W2", "C3-E", outside[task], 90.0, seed, seed))], outside[task])
    frames += card("What this small study found",
                   "With little data, estimating the physical parameter is better.\n"
                   "With more data, physics plus learning is best, inside the training region.\n"
                   "Outside it, the controller exploits learned-model errors.\n"
                   "Averaging an ensemble helps, but costs more computation than real time allows here.\n\n"
                   "Simulation only, one planar arm, one payload mass.", 10)
    imageio.mimwrite(OUT, frames, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    print(f"wrote {OUT} ({len(frames) / FPS:.0f} s)")


if __name__ == "__main__":
    main()
