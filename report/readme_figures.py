"""Images for the repository's README: a banner and a results-at-a-glance chart.

Run:  python report/readme_figures.py

The banner's robot is a real run, not a drawing: the Panda with a 2.5 kg payload
reaching a target under the MPC told the true payload, rendered by MuJoCo, with
the tip's path. The chart is drawn from results/m12c/handover_runs.csv; nothing
else is re-run. Writes report/figures/readme_banner.png and readme_handover.png.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import FancyBboxPatch

from lac import panda
from lac.panda_control import make_controller
from lac.panda_view import add_trail

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
BACKGROUND, INK, MUTED, ACCENT = "#11161d", "#f2f4f7", "#9aa5b1", "#e0703a"
FONT = ["Segoe UI", "Arial", "DejaVu Sans"]


def render_reach(size=(760, 600)) -> np.ndarray:
    q_start = np.array(panda.HOME) + np.array([0.5, 0.25, 0.0, 0.35, 0.0, -0.3, 0.0])
    target = np.array([0.5, -0.12, 0.3])
    payload = 2.5
    controller = make_controller("reference", "W1", payload, seed=0)
    robot = panda.Robot(payload, "W1", controller.load)
    robot.place(q_start)
    robot.set_target(target)
    controller.start(target, robot.q)
    trail = []
    for step in range(int(1.6 / panda.TIMESTEP)):
        robot.configure_load(controller.load)
        robot.step(controller.setpoint(robot.q, robot.dq))
        if step % 20 == 0:
            trail.append(robot.tip())
    camera = mujoco.MjvCamera()
    camera.lookat[:], camera.distance, camera.azimuth, camera.elevation = [0.3, -0.05, 0.42], 1.12, 142, -10
    with mujoco.Renderer(robot.model, size[1], size[0]) as renderer:
        renderer.update_scene(robot.data, camera)
        add_trail(renderer.scene, trail)
        return renderer.render()


def banner() -> None:
    image = render_reach()
    fig = plt.figure(figsize=(16, 5), dpi=100, facecolor=BACKGROUND)
    text = fig.add_axes((0, 0, 1, 1))
    text.set_axis_off()
    text.set_xlim(0, 16)
    text.set_ylim(0, 5)
    text.text(0.75, 3.55, "Learning-Assisted Control", color=INK, fontsize=40, fontweight="bold", family=FONT)
    text.text(0.75, 2.85, "under Unknown Payloads", color=INK, fontsize=40, fontweight="bold", family=FONT)
    text.plot([0.78, 1.9], [2.45, 2.45], color=ACCENT, linewidth=4, solid_capstyle="round")
    text.text(0.75, 1.75, "A learned policy moves, a learned model watches, physics catches.", color=INK,
              fontsize=17, family=FONT)
    text.text(0.75, 1.05, "MuJoCo  ·  two-link arm and Franka Panda  ·  model correction, diffusion policy,"
              "\nfailure monitors and handover  ·  simulation study", color=MUTED, fontsize=12.5, family=FONT,
              linespacing=1.5, va="center")
    left, bottom, width, height = 10.85, 0.35, 4.75, 4.3
    card = fig.add_axes((left / 16, bottom / 5, width / 16, height / 5))
    card.imshow(image, aspect="auto")
    card.set_axis_off()
    clip = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.035", transform=card.transAxes,
                          facecolor="none", edgecolor="none")
    card.add_patch(clip)
    card.images[0].set_clip_path(clip)
    fig.savefig(OUT / "readme_banner.png", facecolor=BACKGROUND)
    plt.close(fig)


def handover_chart() -> None:
    """Share of Panda runs not settled within 1 cm, pooled over three worlds and two stiffnesses (96 runs a bar)."""
    with open(ROOT / "results" / "m12c" / "handover_runs.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scenarios = [("inside", "2.5", "payload 2.5 kg\ninside the training region"),
                 ("inside", "1.0", "payload 1 kg\ninside the training region"),
                 ("outside", "0.0", "no payload\noutside the training region")]
    controllers = [("policy", "diffusion policy alone", "#8d99a6"),
                   ("handover-physics", "+ physics-model monitor", "#3b6ea8"),
                   ("handover-learned", "+ learned-model monitor", "#2f9e73"),
                   ("handover-consistency", "+ action-consistency monitor", ACCENT)]
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
    ax.set_ylabel("runs that fail to settle\nwithin 1 cm of the target", fontsize=9)
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
    banner()
    handover_chart()


if __name__ == "__main__":
    main()
