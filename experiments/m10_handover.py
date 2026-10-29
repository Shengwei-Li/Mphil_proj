"""M10: does handing over to the MPC on a monitor's alarm rescue the policy?

Run:  python experiments/m10_handover.py              (all CPU cores, about 15 minutes)
      python experiments/m10_handover.py --plot-only  (redraw the figure from the saved tables)

Controllers, each run on the same tasks (lac.handover):

    policy        CFDP alone (M8's frozen settings)
    adaptive MPC  the MPC from the start, its payload estimated from the run's
                  own data every 10 periods (no policy)
    reference     the MPC told the true payload and world (an upper bound)
    handover-X    CFDP until detector X alarms, then the adaptive MPC; X is one of
                  physics, learned, consistency (alpha = 0.05), or physics or
                  consistency, learned or consistency (alpha / 2 each)

The detectors are M9's monitors with M9's recipe, refitted here on CFDP's clean
runs: standardisation on 20 dev tasks (seed 5500), thresholds on 100
calibration tasks (seed 7100; the first 40 are M9's). 100 rather than 40, because
at alpha / 2 = 0.025 a threshold from 40 runs is simply their maximum (M9).
The measurement noise is M9's.

Test tasks are M9's: 32 inside (seed 5000) with 0, 0.1 and 1 kg; 32 outside
(seed 6000) with 0 and 1 kg. Worlds W1-W3.

Writes results/m10/runs.csv, summary.csv and handover.png; the clean runs used
for fitting go to cache/m10_clean.pkl.
"""

from __future__ import annotations

import csv
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lac import arm, demos, methods, model, monitor
from lac.handover import HandoverController, consistency, fit_detector, residual
from lac.mpc import MPCController
from lac.policy import CFDP, PolicyController
from lac.predictors import PhysicsPredictor
from lac.simulate import run_reach

WORLDS = ("W1", "W2", "W3")
ALPHA = 0.05
TRIGGERS = {"physics": ("physics",), "learned": ("learned",), "consistency": ("consistency",),
            "physics|consistency": ("physics", "consistency"), "learned|consistency": ("learned", "consistency")}
CONTROLLERS = ("policy", "adaptive MPC", "reference") + tuple(f"handover-{t}" for t in TRIGGERS)
FIT_TASKS = {"dev": methods.make_tasks(20, seed=5500, region="train"),
             "calibration": methods.make_tasks(100, seed=7100, region="train")}
TASKS = {"inside": methods.make_tasks(32, seed=5000, region="train"),
         "outside": methods.make_tasks(32, seed=6000, region="outside")}
PAYLOADS = {"inside": (0.0, 0.1, 1.0), "outside": (0.0, 1.0)}
DURATION = 3.0
SETTLED = 0.01  # m
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m10"
CLEAN = ROOT / "cache" / "m10_clean.pkl"


def clean_runs(world: str) -> list[dict]:
    """CFDP alone on the unloaded arm: the material the detectors are fitted on."""
    policy = CFDP().fit(demos.load_or_record(world))
    runs = []
    for split, tasks in FIT_TASKS.items():
        for task, (q_start, target) in enumerate(tasks):
            c = PolicyController(policy, seed=task)
            r = run_reach(c, 0.0, q_start, target, DURATION, world)
            runs.append({"split": split, "periods": monitor.periods(r, c.hold), "chunks": c.chunks,
                         "chunk_periods": c.chunk_periods, "noise_seed": 10_000 + len(runs)})
    return runs


def detectors(world: str, clean: list[dict]) -> dict:
    """Every detector a trigger needs, at the alpha it is used with."""
    models = methods.build_predictors(world, 0.0, 90.0, 1)
    signals = {"physics": residual(models["C0"]), "learned": residual(models["C2"]), "consistency": consistency()}
    dev = [r for r in clean if r["split"] == "dev"]
    calibration = [r for r in clean if r["split"] == "calibration"]
    return {(name, alpha): fit_detector(name, signals[name], dev, calibration, alpha)
            for name in signals for alpha in (ALPHA, ALPHA / 2)}


def make_controller(name: str, policy, found: dict, world: str, payload: float, task: int, noise_seed: int):
    if name == "policy":
        return PolicyController(policy, seed=task)
    if name == "reference":
        return MPCController(predictor=PhysicsPredictor(model.Belief(payload, **arm.WORLDS[world])), seed=task,
                             show_plan=False)
    if name == "adaptive MPC":
        return HandoverController(None, mpc_seed=task, noise_seed=noise_seed)
    members = TRIGGERS[name.removeprefix("handover-")]
    alpha = ALPHA if len(members) == 1 else ALPHA / 2
    return HandoverController(PolicyController(policy, seed=task), tuple(found[(m, alpha)] for m in members),
                              mpc_seed=task, noise_seed=noise_seed)


def settle_time(rollout) -> float:
    above = np.nonzero(rollout.error > SETTLED)[0]
    if len(above) == 0:
        return 0.0
    return DURATION if above[-1] == len(rollout.error) - 1 else float(rollout.time[above[-1] + 1])


def run_group(job) -> list[dict]:
    world, name, split = job
    clean = pickle.loads(CLEAN.read_bytes())[world]
    found = detectors(world, clean) if name.startswith("handover") else {}
    policy = CFDP().fit(demos.load_or_record(world))
    rows = []
    for payload in PAYLOADS[split]:
        for task, (q_start, target) in enumerate(TASKS[split]):
            noise_seed = ((WORLDS.index(world) * 2 + list(TASKS).index(split)) * 20 + round(payload * 10)) * 100 + task
            c = make_controller(name, policy, found, world, payload, task, noise_seed)
            rollout = run_reach(c, payload, q_start, target, DURATION, world)
            handed = getattr(c, "handover_period", None)
            masses = getattr(c, "mass_estimates", [])
            rows.append({
                "world": world, "split": split, "controller": name, "payload_kg": payload, "task": task,
                "final_error_cm": round(rollout.final_error() * 100, 4),
                "settle_time_s": round(settle_time(rollout), 3),
                "handover_s": "" if handed is None or name == "adaptive MPC" else round(handed * 0.02, 2),
                "alarmed_by": "+".join(getattr(c, "alarmed_by", [])),
                "final_mass_estimate_kg": round(masses[-1][1], 3) if masses else "",
            })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--plot-only" not in sys.argv:
        for world in WORLDS:
            demos.load_or_record(world)
            methods.build_predictors(world, 0.0, 90.0, 1, train_missing=True)
        if not CLEAN.exists():
            with ProcessPoolExecutor() as pool:
                CLEAN.write_bytes(pickle.dumps(dict(zip(WORLDS, pool.map(clean_runs, WORLDS)))))
        jobs = [(w, n, s) for w in WORLDS for n in CONTROLLERS for s in TASKS]
        with ProcessPoolExecutor() as pool:
            rows = [r for g in pool.map(run_group, jobs) for r in g]
        write("runs.csv", rows)
        summary = summarise(rows)
        write("summary.csv", summary)
        for r in summary:
            print(r)
    plot(read("summary.csv"))


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for split in TASKS:
        for world in WORLDS:
            for payload in PAYLOADS[split]:
                for name in CONTROLLERS:
                    g = [r for r in rows if (r["split"], r["world"], r["payload_kg"], r["controller"])
                         == (split, world, payload, name)]
                    errors = [r["final_error_cm"] for r in g]
                    handed = [r["handover_s"] for r in g if r["handover_s"] != ""]
                    out.append({
                        "split": split, "world": world, "payload_kg": payload, "controller": name, "runs": len(g),
                        "median_final_error_cm": round(float(np.median(errors)), 3),
                        "p90_final_error_cm": round(float(np.percentile(errors, 90)), 3),
                        "worst_final_error_cm": round(max(errors), 3),
                        "runs_not_settled": sum(r["settle_time_s"] >= DURATION for r in g),
                        "handover_rate": round(len(handed) / len(g), 3) if name.startswith("handover") else "",
                        "median_handover_s": round(float(np.median(handed)), 2) if handed else "",
                    })
    return out


def write(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read(name: str) -> list[dict]:
    with open(OUT / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot(summary: list[dict]) -> None:
    """Runs not settled within 1 cm, per scenario and controller."""
    scenarios = [("inside", "0.0"), ("inside", "0.1"), ("inside", "1.0"), ("outside", "0.0"), ("outside", "1.0")]
    colors = ["#3182bd", "#bdbdbd", "#000000", "#6baed6", "#de2d26", "#31a354", "#9e9ac8", "#fd8d3c"]
    fig, axes = plt.subplots(1, len(WORLDS), figsize=(15, 4.6), sharey=True)
    width = 0.8 / len(CONTROLLERS)
    for ax, world in zip(axes, WORLDS):
        for k, (name, color) in enumerate(zip(CONTROLLERS, colors)):
            values = []
            for split, payload in scenarios:
                row = next(r for r in summary if (r["split"], r["world"], r["controller"]) == (split, world, name)
                           and float(r["payload_kg"]) == float(payload))
                values.append(int(row["runs_not_settled"]) / int(row["runs"]))
            ax.bar(np.arange(len(scenarios)) + (k - len(CONTROLLERS) / 2 + 0.5) * width, values, width,
                   color=color, label=name)
        ax.set_xticks(range(len(scenarios)), [f"{s}\n{float(p):g} kg" for s, p in scenarios], fontsize=8)
        ax.set_title(world, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("fraction of runs not settled within 1 cm")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CONTROLLERS), fontsize=8, frameon=False)
    fig.suptitle("Policy alone, MPC alone, and policy handing over to the adaptive MPC on a monitor's alarm",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "handover.png", dpi=130)


if __name__ == "__main__":
    main()
