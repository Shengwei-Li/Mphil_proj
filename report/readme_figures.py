"""Images for the repository's README: two replays of real Panda runs and a results chart.

Run:  python report/readme_figures.py   (needs the m12c caches: python reproduce.py m12c first)

Each replay shows one test task of experiments/m12c_handover.py twice, side by
side: the policy alone, and the policy with a monitor that hands over to the MPC.
The runs are re-simulated with the same seeds as in that experiment, so the
numbers on screen match results/m12c/handover_runs.csv. The tasks are typical
ones: the policy's error is close to the median of its group.

    readme_outside.gif   no payload, target outside the training region, consistency monitor
    readme_payload.gif   2.5 kg payload, target inside the region, physics-residual monitor
    readme_handover.png  share of runs not settled, pooled over all Panda test runs

Writes report/figures/.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import m12c_handover as ho  # noqa: E402
import m12c_monitors as m12  # noqa: E402

from lac import panda  # noqa: E402
from lac import panda_monitor as pm  # noqa: E402
from lac import panda_policy as pp  # noqa: E402

OUT = ROOT / "report" / "figures"
PANEL = (440, 330)  # width, height of one rendered view
CAPTION = 46  # height of the text strip above each view
FRAME_EVERY = 25  # simulator steps (0.05 s)
FONT = ["Segoe UI", "Arial", "DejaVu Sans"]
REPLAYS = {
    # name: (gains, world, split, payload, task, monitor), camera (look at, distance, azimuth, elevation)
    "outside": (("soft", "W1", "outside", 0.0, 4, "consistency"), ([0.3, -0.3, 0.4], 1.45, 152, -20)),
    "payload": (("soft", "W1", "inside", 2.5, 5, "physics"), ([0.38, 0.0, 0.3], 0.85, 105, -5)),
}


def font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def simulate(gains, world, split, payload, task, monitor):
    """The task under the policy alone and under the handover, as in m12c_handover.run_group."""
    q_start, target = m12.TASKS[split][task]
    seed = m12.noise_seed(gains, world, split, payload, task) + 500
    alone = pp.PandaPolicyController(m12.policy(world, gains), seed=task, gains=gains)
    detector = ho.detectors(gains, world)[monitor]
    handover = pm.PandaHandover(pp.PandaPolicyController(m12.policy(world, gains), seed=task, gains=gains),
                                (detector,), mpc_seed=task, noise_seed=seed, gains=gains)
    runs = [panda.run_reach(c, payload, q_start, target, m12.DURATION, world) for c in (alone, handover)]
    return q_start, target, runs, handover.handover_period


def camera(look_at, distance, azimuth, elevation) -> mujoco.MjvCamera:
    c = mujoco.MjvCamera()
    c.lookat[:], c.distance, c.azimuth, c.elevation = look_at, distance, azimuth, elevation
    return c


def scene_model(payload, world, gains, target) -> panda.Robot:
    """The robot for drawing only: softer light, a larger target marker, a fainter payload."""
    robot = panda.Robot(payload, world, panda.Belief(), gains)
    robot.set_target(target)
    m = robot.model
    m.vis.headlight.diffuse[:], m.vis.headlight.ambient[:] = 0.25, 0.25
    m.light_diffuse[:] *= 0.75
    m.site_size[m.site("target_site").id] = 0.02
    m.geom_rgba[m.geom("payload_geom").id, 3] = 0.2
    return robot


def add_path(scene: mujoco.MjvScene, points) -> None:
    for point in points:
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.006, 0, 0], point,
                            np.eye(3).ravel(), np.array([0.15, 0.15, 0.15, 0.8], dtype=np.float32))
        scene.ngeom += 1


def caption(width: int, title: str, detail: str) -> Image.Image:
    strip = Image.new("RGB", (width, CAPTION), "white")
    draw = ImageDraw.Draw(strip)
    draw.text((12, 5), title, fill="#1f2328", font=font(16))
    draw.text((12, 25), detail, fill="#59636e", font=font(13))
    return strip


def replay(name: str) -> None:
    (gains, world, split, payload, task, monitor), view = REPLAYS[name]
    q_start, target, runs, handed = simulate(gains, world, split, payload, task, monitor)
    robot = scene_model(payload, world, gains, target)
    cam = camera(*view)
    titles = ["Policy alone", f"Policy with the {monitor} monitor"]
    frames = []
    with mujoco.Renderer(robot.model, PANEL[1], PANEL[0]) as renderer:
        steps = list(range(0, len(runs[0].q), FRAME_EVERY)) + [len(runs[0].q) - 1] * 25  # hold the last frame
        for step in steps:
            views = []
            for k, run in enumerate(runs):
                robot.data.qpos[:] = run.q[step]
                mujoco.mj_forward(robot.model, robot.data)
                renderer.update_scene(robot.data, cam)
                add_path(renderer.scene, run.tip[: step + 1 : FRAME_EVERY])
                error = run.final_error() if step == len(run.q) - 1 else run.error[step]
                detail = f"t = {step * panda.TIMESTEP:.2f} s    tip to target {error * 100:.1f} cm"
                if k == 1 and handed is not None and step >= handed * panda.HOLD:
                    detail += f"    MPC in control since {handed * panda.PERIOD:.2f} s"
                panel = Image.new("RGB", (PANEL[0], PANEL[1] + CAPTION), "white")
                panel.paste(caption(PANEL[0], titles[k], detail), (0, 0))
                panel.paste(Image.fromarray(renderer.render()), (0, CAPTION))
                views.append(panel)
            frame = Image.new("RGB", (2 * PANEL[0] + 8, PANEL[1] + CAPTION), "white")
            frame.paste(views[0], (0, 0))
            frame.paste(views[1], (PANEL[0] + 8, 0))
            frames.append(np.asarray(frame))
    imageio.mimsave(OUT / f"readme_{name}.gif", frames, duration=FRAME_EVERY * panda.TIMESTEP * 1000 * 1.5, loop=0)


def handover_chart() -> None:
    """Share of Panda runs not settled within 1 cm, pooled over three worlds and two stiffnesses (96 runs a bar)."""
    with open(ROOT / "results" / "m12c" / "handover_runs.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scenarios = [("inside", "2.5", "payload 2.5 kg\ninside the training region"),
                 ("inside", "1.0", "payload 1 kg\ninside the training region"),
                 ("outside", "0.0", "no payload\noutside the training region")]
    controllers = [("policy", "policy alone", "#8d99a6"),
                   ("handover-physics", "physics monitor", "#3b6ea8"),
                   ("handover-learned", "learned monitor", "#2f9e73"),
                   ("handover-consistency", "consistency monitor", "#e0703a")]
    plt.rcParams.update({"font.family": FONT})
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    width = 0.2
    for k, (name, label, color) in enumerate(controllers):
        shares = []
        for split, payload, _ in scenarios:
            group = [r for r in rows if (r["split"], r["payload_kg"], r["controller"]) == (split, payload, name)]
            shares.append(sum(1 - int(r["settled"]) for r in group) / len(group))
        x = np.arange(len(scenarios)) + (k - 1.5) * width
        ax.bar(x, shares, width * 0.92, color=color, label=label)
        for xi, s in zip(x, shares):
            ax.text(xi, s + 0.02, f"{s:.0%}", ha="center", va="bottom", fontsize=7.5, color="#333333")
    ax.set_xticks(range(len(scenarios)), [s[2] for s in scenarios], fontsize=9)
    ax.set_ylabel("runs not settled\nwithin 1 cm of the target", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "readme_handover.png", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in REPLAYS:
        replay(name)
        print("drew", name)
    handover_chart()


if __name__ == "__main__":
    main()
