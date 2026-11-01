"""Figures for the technical report, drawn from the saved experiment tables.

Run:  python report/figures.py

Reads results/m5 and results/m6. Nothing is re-run, except that the three
worst runs outside the training region are replayed (deterministically, with
the cached networks) to draw their tip paths.

Writes report/figures/*.pdf (vector, for the report) and *.png (for quick viewing).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from lac import arm, data, methods  # noqa: E402
from lac.view import RENDER_SIZE  # noqa: E402

OUT = ROOT / "report" / "figures"
WORLDS = ("W1", "W2", "W3")
BUDGETS = (9.0, 30.0, 90.0)
COLUMN, PAGE = 3.5, 7.16  # IEEE column and page widths, inches
COLOURS = {"C0": "#4C72B0", "C1": "#DD8452", "C2": "#55A868", "C3": "#C44E52", "C3-E": "#8172B3", "reference": "black"}
MARKERS = {"C1": "o", "C2": "s", "C3": "^", "C3-E": "D"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2,
    "lines.markersize": 4, "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def cell(rows, metric: str, **match) -> list[float]:
    return [r[metric] for r in rows if all(r[k] == v for k, v in match.items())]


def budget_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS, [f"{b:g}" for b in BUDGETS])
    ax.minorticks_off()
    ax.set_xlim(7.5, 110)


# --- Figure 1: the arm, the regions and the test tasks -------------------------


def figure_setup() -> None:
    import mujoco

    model, sim = arm.load_arm()
    arm.set_world(model, "W2")
    arm.set_payload(model, 1.0)
    pose = np.radians([-50.0, 110.0])
    arm.set_joint_angles(model, sim, pose)
    arm.set_target(model, sim, np.array([2.0, 2.0]))  # out of view
    renderer = mujoco.Renderer(model, RENDER_SIZE, RENDER_SIZE)
    renderer.update_scene(sim, camera="side")
    image = renderer.render()

    # Pixel coordinates of a point (x, z) on the arm's plane, for the fixed side camera.
    focal = (RENDER_SIZE / 2) / np.tan(np.radians(20.0))
    distance = 2.2

    def pixel(x, z):
        return RENDER_SIZE / 2 + focal * np.asarray(x) / distance, RENDER_SIZE / 2 - focal * np.asarray(z) / distance

    fig, ax = plt.subplots(figsize=(COLUMN, COLUMN * 0.82))
    ax.imshow(image)
    ax.grid(False)
    left, right = pixel(-0.62, 0.62)[0], pixel(0.62, 0.62)[0]
    top, bottom = pixel(0, 0.62)[1], pixel(0, -0.62)[1]
    ax.axvspan(pixel(0.05, 0)[0], right, ymin=0, ymax=1, color=COLOURS["C2"], alpha=0.08, zorder=2)
    ax.axvspan(left, pixel(-0.1, 0)[0], color=COLOURS["C3"], alpha=0.08, zorder=2)
    reach = np.linspace(0, 2 * np.pi, 200)
    ax.plot(*pixel(0.6 * np.cos(reach), 0.6 * np.sin(reach)), color="0.5", linewidth=0.6, linestyle=":")
    for region, colour, marker in (("train", COLOURS["C2"], "o"), ("outside", COLOURS["C3"], "s")):
        tasks = methods.make_tasks(8, seed=5000 if region == "train" else 6000, region=region)
        targets = np.array([t for _, t in tasks])
        ax.plot(*pixel(targets[:, 0], targets[:, 1]), linestyle="none", marker=marker, markersize=3.5,
                markerfacecolor="none", markeredgecolor=colour)
    band = 52  # pixels of blank space above the image for the region labels
    ax.text(pixel(0.33, 0)[0], top - band + 4, "training region\n(test targets: circles)", ha="center", va="top",
            color=COLOURS["C2"], fontsize=7)
    ax.text(pixel(-0.37, 0)[0], top - band + 4, "outside region\n(test targets: squares)", ha="center", va="top",
            color=COLOURS["C3"], fontsize=7)
    ax.annotate("payload, 1 kg\n(offset in W2, W3)", xy=pixel(*arm.forward_kinematics(pose)),
                xytext=pixel(0.02, -0.57), fontsize=7, ha="left",
                arrowprops={"arrowstyle": "->", "linewidth": 0.6})
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top - band)
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, "setup")


# --- Figure 2: prediction error against data -----------------------------------


def figure_prediction() -> None:
    rows = read(ROOT / "results" / "m5" / "prediction.csv")
    fig, axes = plt.subplots(1, 3, figsize=(PAGE, 2.1), sharey=True)
    for ax, world in zip(axes, WORLDS):
        for method in ("C1", "C2", "C3"):
            for region, style in (("inside", "-"), ("outside", "--")):
                values = [max(np.mean(cell(rows, "median_tip_error_at_0.2s_cm", world=world, method=method,
                                           budget_s=b, validation=region)), 1e-3) for b in BUDGETS]
                ax.plot(BUDGETS, values, color=COLOURS[method], marker=MARKERS[method], linestyle=style,
                        markerfacecolor="white" if region == "outside" else COLOURS[method])
        c0 = np.mean(cell(rows, "median_tip_error_at_0.2s_cm", world=world, method="C0", validation="inside"))
        ax.axhline(c0, color=COLOURS["C0"], linewidth=0.8, linestyle=":")
        if world == "W1":
            ax.text(30, 1.6e-3, "C1, C3: exact (0), drawn at $10^{-3}$", ha="center", fontsize=6.5, color="0.3")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 60)
        budget_axis(ax)
        ax.set_title(world)
        ax.set_xlabel("training data (s of motion)")
    axes[0].set_ylabel("tip prediction error after 0.2 s (cm)")
    handles = [plt.Line2D([], [], color=COLOURS[m], marker=MARKERS[m], label=m) for m in ("C1", "C2", "C3")]
    handles += [plt.Line2D([], [], color="0.3", linestyle="-", label="inside training region"),
                plt.Line2D([], [], color="0.3", linestyle="--", label="outside"),
                plt.Line2D([], [], color=COLOURS["C0"], linestyle=":", label="C0")]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.08))
    save(fig, "prediction")


# --- Figure 3: control error inside and outside --------------------------------


def control_rows() -> list[dict]:
    single = read(ROOT / "results" / "m6" / "runs.csv")
    ensemble = [r for r in read(ROOT / "results" / "m6" / "ensemble_runs.csv") if r["method"] == "C3-E avg"]
    for r in ensemble:
        r["method"] = "C3-E"
    return single + ensemble


def figure_control() -> None:
    rows = control_rows()
    fig, axes = plt.subplots(1, 3, figsize=(PAGE, 2.1), sharey=True)
    for ax, world in zip(axes, WORLDS):
        for method in ("C1", "C2", "C3", "C3-E"):
            for region, style in (("inside", "-"), ("outside", "--")):
                values = [np.median(cell(rows, "final_error_cm", world=world, method=method, budget_s=b, region=region))
                          for b in BUDGETS]
                ax.plot(BUDGETS, values, color=COLOURS[method], marker=MARKERS[method], linestyle=style,
                        markerfacecolor="white" if region == "outside" else COLOURS[method])
        for method, style in (("C0", ":"), ("reference", ":")):
            ax.axhline(np.median(cell(rows, "final_error_cm", world=world, method=method)), color=COLOURS[method],
                       linewidth=0.8, linestyle=style)
        ax.set_yscale("log")
        ax.set_ylim(0.02, 30)
        budget_axis(ax)
        ax.set_title(world)
        ax.set_xlabel("training data (s of motion)")
    axes[0].set_ylabel("median final tip error (cm)")
    handles = [plt.Line2D([], [], color=COLOURS[m], marker=MARKERS[m], label=m) for m in ("C1", "C2", "C3", "C3-E")]
    handles += [plt.Line2D([], [], color="0.3", linestyle="-", label="inside"),
                plt.Line2D([], [], color="0.3", linestyle="--", label="outside"),
                plt.Line2D([], [], color=COLOURS["C0"], linestyle=":", label="C0"),
                plt.Line2D([], [], color="black", linestyle=":", label="reference")]
    fig.legend(handles=handles, loc="upper center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 1.08))
    save(fig, "control")


# --- Figure 4: the MPC's expectations are optimistic ---------------------------


def figure_optimism() -> None:
    rows = read(ROOT / "results" / "m6" / "runs.csv")
    fig, ax = plt.subplots(figsize=(COLUMN, 2.4))
    groups = {}
    for r in rows:
        if r["method"] == "reference" or (r["world"] == "W1" and r["method"] in ("C1", "C3")):
            continue  # exact models: no error to be optimistic about
        groups.setdefault((r["method"], r["world"], r["budget_s"], r["region"]), []).append(r)
    for (method, _, _, region), g in groups.items():
        x = np.median([r["prediction_error_cm"] for r in g])
        y = np.median([r["optimism_cm"] for r in g])
        ax.plot(x, y, linestyle="none", marker=MARKERS.get(method, "v"), color=COLOURS[method], alpha=0.85,
                markerfacecolor="white" if region == "outside" else COLOURS[method])
    line = np.array([1e-4, 1])
    ax.plot(line, line, color="0.4", linewidth=0.8)
    ax.plot(line, 0.5 * line, color="0.4", linewidth=0.8, linestyle="--")
    ax.text(0.98, 0.04, "solid: optimism = error\ndashed: optimism = half the error", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color="0.3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2e-4, 0.8)
    ax.set_ylim(1e-4, 0.8)
    ax.set_xlabel("model's prediction error, 0.02 s ahead (cm)")
    ax.set_ylabel("optimism (cm)")
    handles = [plt.Line2D([], [], linestyle="none", color=COLOURS[m], marker=MARKERS.get(m, "v"), label=m)
               for m in ("C0", "C1", "C2", "C3")]
    handles += [plt.Line2D([], [], linestyle="none", color="0.3", marker="o", markerfacecolor="white", label="outside")]
    ax.legend(handles=handles, loc="upper left", frameon=False)
    save(fig, "optimism")


# --- Figure 5: worst runs outside ----------------------------------------------


def figure_worst() -> None:
    from m6a_outside import TASKS, run

    rows = read(ROOT / "results" / "m6" / "runs.csv")
    learned = [r for r in rows if r["region"] == "outside" and r["method"] == "C2"]
    worst = sorted(learned, key=lambda r: -r["final_error_cm"])[:3]
    fig, axes = plt.subplots(1, 3, figsize=(PAGE, 2.45), sharey=True)
    for ax, r in zip(axes, worst):
        world, budget, seed, task = r["world"], r["budget_s"], int(r["data_seed"]), int(r["task"])
        predictors = methods.build_predictors(world, 1.0, budget, seed)
        q_start, target = TASKS["outside"][task]
        errors = {}
        for name in ("C2", "C1"):
            rollout, row = run(world, "outside", name, predictors[name], budget, seed, seed, task)
            ax.plot(rollout.tip[:, 0], rollout.tip[:, 1], color=COLOURS[name])
            errors[name] = row["final_error_cm"]
        ax.plot(*arm.forward_kinematics(q_start), "o", color="black", markersize=3.5)
        ax.plot(*target, "*", color=COLOURS["C0"], markersize=9)
        ax.axvspan(-0.7, -0.1, color=COLOURS["C3"], alpha=0.06, zorder=0)
        ax.set_xlim(-0.65, 0.65)
        ax.set_ylim(-0.65, 0.65)
        ax.set_aspect("equal")
        ax.set_title(f"{world}, {budget:g} s of data: C2 {errors['C2']:.1f} cm, C1 {errors['C1']:.1f} cm", fontsize=8)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("z (m)")
    handles = [plt.Line2D([], [], color=COLOURS["C2"], label="C2 (no-payload physics + network)"),
               plt.Line2D([], [], color=COLOURS["C1"], label="C1 (estimated mass)"),
               plt.Line2D([], [], color="black", marker="o", linestyle="none", label="start"),
               plt.Line2D([], [], color=COLOURS["C0"], marker="*", markersize=8, linestyle="none", label="target")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.07))
    save(fig, "worst")


# --- Figure 6: single networks against ensembles, outside ----------------------


def figure_ensemble() -> None:
    rows = read(ROOT / "results" / "m6" / "ensemble_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 2.0), sharey=True)
    pairs = (("C2", "C2-E avg", COLOURS["C2"]), ("C3", "C3-E avg", COLOURS["C3"]))
    for ax, world in zip(axes, ("W2", "W3")):
        positions = np.arange(len(BUDGETS))
        for i, (single, ensemble, colour) in enumerate(pairs):
            for j, (name, hatch) in enumerate(((single, None), (ensemble, "////"))):
                values = [next(r["runs_not_settled"] for r in rows if (r["region"], r["world"], r["method"], r["budget_s"])
                               == ("outside", world, name, b)) for b in BUDGETS]
                ax.bar(positions + (2 * i + j - 1.5) * 0.19, values, width=0.19, color=colour,
                       alpha=0.55 if hatch else 0.95, hatch=hatch, edgecolor=colour, linewidth=0.5,
                       label=name.replace(" avg", ""))
        c1 = [next(r["runs_not_settled"] for r in rows if (r["region"], r["world"], r["method"], r["budget_s"])
                   == ("outside", world, "C1", b)) for b in BUDGETS]
        ax.plot(positions, c1, color=COLOURS["C1"], marker="o", linewidth=0.8, label="C1")
        ax.set_xticks(positions, [f"{b:g} s" for b in BUDGETS])
        ax.set_ylim(0, 25)
        ax.set_title(f"{world}, outside")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("runs not settled (of 24)")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(n) for n in ("C1", "C2", "C2-E", "C3", "C3-E")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.1), fontsize=6.5, handlelength=1.5, columnspacing=0.8)
    save(fig, "ensemble")


# --- Parts II and III: monitors and handover -----------------------------------

MONITOR_STYLE = {"physics": ("#3b6ea8", "o", "physics"), "learned": ("#2f9e73", "s", "learned"),
                 "oracle": ("black", "x", "oracle"), "consistency": ("#e0703a", "^", "consistency")}


def figure_detection() -> None:
    """M9: alarm rate against payload on the two-link arm, CFDP inside the training region."""
    rows = read(ROOT / "results" / "m9" / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.9), sharey=True)
    for ax, world in zip(axes, ("W1", "W3")):
        g = sorted((r for r in rows if (r["split"], r["world"], r["policy"], r["alpha"]) == ("inside", world, "CFDP", 0.05)),
                   key=lambda r: r["payload_kg"])
        for monitor, (colour, marker, label) in MONITOR_STYLE.items():
            ax.plot(range(len(g)), [r[f"alarm_rate_{monitor}"] for r in g], color=colour,
                    marker=marker, label=label, linestyle=":" if monitor == "oracle" else "-", markersize=3)
        ax.axhline(0.05, color="0.5", linewidth=0.6, linestyle="--")
        ax.set_xticks(range(len(g)), [f"{r['payload_kg']:g}" for r in g])
        ax.set_xlim(-0.3, len(g) - 0.7)
        ax.set_title(f"{world}" + (" (model exact without payload)" if world == "W1" else " (friction unknown)"),
                     fontsize=7)
        ax.set_xlabel("payload (kg)")
    axes[0].set_ylabel("alarm rate")
    axes[0].set_ylim(-0.03, 1.05)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.1))
    save(fig, "detection")


def figure_panda_handover() -> None:
    """M12c: share of Panda runs not settled, pooled over worlds and stiffnesses (96 runs a bar)."""
    rows = read(ROOT / "results" / "m12c" / "handover_runs.csv")
    scenarios = (("inside", 2.5, "2.5 kg, inside"), ("inside", 1.0, "1 kg, inside"), ("outside", 0.0, "0 kg, outside"))
    controllers = (("policy", "policy alone", "#8d99a6"), ("handover-physics", "+ physics", "#3b6ea8"),
                   ("handover-learned", "+ learned", "#2f9e73"), ("handover-consistency", "+ consistency", "#e0703a"))
    fig, ax = plt.subplots(figsize=(COLUMN, 1.9))
    width = 0.2
    for k, (name, label, colour) in enumerate(controllers):
        shares = []
        for split, payload, _ in scenarios:
            g = [r for r in rows if (r["split"], r["payload_kg"], r["controller"]) == (split, payload, name)]
            shares.append(sum(1 - r["settled"] for r in g) / len(g))
        x = np.arange(len(scenarios)) + (k - 1.5) * width
        ax.bar(x, shares, width * 0.92, color=colour, label=label)
    ax.set_xticks(range(len(scenarios)), [s[2] for s in scenarios])
    ax.set_ylabel("runs not settled")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.22), fontsize=6.5,
              handlelength=1.2, columnspacing=0.8)
    save(fig, "panda_handover")


def figure_panda_runs() -> None:
    """Last frame of the 2.5 kg replay of the README: the policy alone, and with the physics monitor."""
    import mujoco
    sys.path.insert(0, str(ROOT / "report"))
    import readme_figures as rf

    (gains, world, split, payload, task, monitor), view = rf.REPLAYS["payload"]
    _, target, runs, handed = rf.simulate(gains, world, split, payload, task, monitor)
    robot = rf.scene_model(payload, world, gains, target)
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.55))
    titles = (f"policy alone: {runs[0].final_error() * 100:.1f} cm",
              f"physics monitor, MPC from {handed * 0.02:.2f} s: {runs[1].final_error() * 100:.1f} cm")
    with mujoco.Renderer(robot.model, 360, 420) as renderer:
        for ax, run, title in zip(axes, runs, titles):
            robot.data.qpos[:] = run.q[-1]
            mujoco.mj_forward(robot.model, robot.data)
            renderer.update_scene(robot.data, rf.camera(*view))
            rf.add_path(renderer.scene, run.tip[:: rf.FRAME_EVERY])
            ax.imshow(renderer.render())
            ax.set_title(title, fontsize=6.5)
            ax.set_axis_off()
    save(fig, "panda_runs")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for make in (figure_setup, figure_prediction, figure_control, figure_optimism, figure_worst, figure_ensemble,
                 figure_detection, figure_panda_handover, figure_panda_runs):
        make()
        print("drew", make.__name__)


if __name__ == "__main__":
    main()
