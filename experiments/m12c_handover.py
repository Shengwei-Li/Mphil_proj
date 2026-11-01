"""M12c, step 3: does handing over to the MPC on an alarm rescue the policy on the Panda? (M10 on the Panda)

Run:  python experiments/m12c_monitors.py   (first: its dev and calibration runs fit the detectors)
      python experiments/m12c_handover.py    (all CPU cores)
      python experiments/m12c_handover.py --plot-only

Controllers, on the same tasks, at both joint stiffnesses:

    policy        CFDP alone
    adaptive MPC  the MPC from the start, payload re-estimated every 10 periods
                  (lac.panda_control.AdaptiveMPC; no policy)
    reference     the MPC told the true payload and world
    handover-X    CFDP until detector X alarms, then the MPC with the payload
                  estimated from every step so far (lac.panda_monitor.PandaHandover);
                  X is physics, learned or consistency, alpha = 0.05

M10's "or" triggers are left out: at alpha / 2 they lost most of the
consistency monitor's alarms (M10, H8). Detectors are fitted exactly as in
m12c_monitors.py, on its dev (20) and calibration (100) runs.

Test tasks: m12c_monitors.py's, 16 inside (0, 1 and 2.5 kg) and 16 outside (0 and
2.5 kg). The expectations H24-H26 were written in docs/PROGRESS.md before this run.
Writes results/m12c/handover_runs.csv, handover_summary.csv and handover.png.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m12c_monitors as m12  # noqa: E402

from lac import panda  # noqa: E402
from lac import panda_monitor as pm  # noqa: E402
from lac import panda_policy as pp  # noqa: E402
from lac.panda_control import make_controller  # noqa: E402

TRIGGERS = ("physics", "learned", "consistency")
CONTROLLERS = ("policy", "adaptive MPC", "reference") + tuple(f"handover-{t}" for t in TRIGGERS)
SPLITS = ("inside", "outside")
PAYLOADS = {"inside": (0.0, 1.0, 2.5), "outside": (0.0, 2.5)}
OUT = m12.OUT


def detectors(gains: str, world: str) -> dict:
    clean = [r for r in pickle.loads(m12.CLEAN.read_bytes()) if (r["gains"], r["world"]) == (gains, world)]
    models = pm.monitor_models(world, gains, pp.load_or_record(world, gains))
    signals = {m: pm.residual(models[m]) for m in ("physics", "learned")}
    signals["consistency"] = pm.consistency()
    dev = [r for r in clean if r["split"] == "dev"]
    calibration = [r for r in clean if r["split"] == "calibration"]
    return {m: pm.fit_detector(m, signals[m], dev, calibration, m12.ALPHA) for m in TRIGGERS}


def run_group(job) -> list[dict]:
    gains, world, name, split = job
    found = detectors(gains, world) if name.startswith("handover") else {}
    rows = []
    for payload in PAYLOADS[split]:
        for task, (q_start, target) in enumerate(m12.TASKS[split]):
            seed = m12.noise_seed(gains, world, split, payload, task) + 500  # not any monitor run's noise
            if name == "policy":
                c = pp.PandaPolicyController(m12.policy(world, gains), seed=task, gains=gains)
            elif name in ("adaptive MPC", "reference"):
                c = make_controller("C1" if name == "adaptive MPC" else "reference", world, payload, seed=task,
                                    gains=gains)
            else:
                c = pm.PandaHandover(pp.PandaPolicyController(m12.policy(world, gains), seed=task, gains=gains),
                                     (found[name.removeprefix("handover-")],), mpc_seed=task, noise_seed=seed,
                                     gains=gains)
            r = panda.run_reach(c, payload, q_start, target, m12.DURATION, world)
            above = np.nonzero(r.error > m12.SETTLED)[0]
            handed = getattr(c, "handover_period", None)
            masses = getattr(c, "mass_estimates", [])
            rows.append({"gains": gains, "world": world, "split": split, "controller": name, "payload_kg": payload,
                         "task": task, "final_error_cm": round(r.final_error() * 100, 4),
                         "settled": int(len(above) == 0 or above[-1] < len(r.error) - 1),
                         "handover_s": "" if handed is None else round(handed * panda.PERIOD, 2),
                         "final_mass_estimate_kg": round(masses[-1][1], 3) if masses else ""})
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for split in SPLITS:
        for gains in m12.GAINS:
            for world in m12.WORLDS:
                for payload in PAYLOADS[split]:
                    for name in CONTROLLERS:
                        g = [r for r in rows if (r["split"], r["gains"], r["world"], r["payload_kg"], r["controller"])
                             == (split, gains, world, payload, name)]
                        handed = [r["handover_s"] for r in g if r["handover_s"] != ""]
                        out.append({"split": split, "gains": gains, "world": world, "payload_kg": payload,
                                    "controller": name, "runs": len(g),
                                    "median_final_error_cm": round(float(np.median([r["final_error_cm"] for r in g])), 3),
                                    "worst_final_error_cm": round(max(r["final_error_cm"] for r in g), 3),
                                    "runs_not_settled": sum(1 - r["settled"] for r in g),
                                    "handover_rate": round(len(handed) / len(g), 3) if name.startswith("handover") else "",
                                    "median_handover_s": round(float(np.median(handed)), 2) if handed else ""})
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
    """Runs not settled within 1 cm, per scenario and controller; one row per stiffness."""
    scenarios = [("inside", 0.0), ("inside", 1.0), ("inside", 2.5), ("outside", 0.0), ("outside", 2.5)]
    colors = ["#3182bd", "#bdbdbd", "#000000", "#de2d26", "#31a354", "#fd8d3c"]
    fig, axes = plt.subplots(len(m12.GAINS), len(m12.WORLDS), figsize=(15, 8), sharey=True)
    width = 0.8 / len(CONTROLLERS)
    for i, gains in enumerate(m12.GAINS):
        for j, world in enumerate(m12.WORLDS):
            ax = axes[i, j]
            for k, (name, color) in enumerate(zip(CONTROLLERS, colors)):
                values = []
                for split, payload in scenarios:
                    row = next(r for r in summary if (r["split"], r["gains"], r["world"], r["controller"])
                               == (split, gains, world, name) and float(r["payload_kg"]) == payload)
                    values.append(int(row["runs_not_settled"]) / int(row["runs"]))
                ax.bar(np.arange(len(scenarios)) + (k - len(CONTROLLERS) / 2 + 0.5) * width, values, width,
                       color=color, label=name)
            ax.set_xticks(range(len(scenarios)), [f"{s}\n{p:g} kg" for s, p in scenarios], fontsize=8)
            ax.set_title(f"{world}, {gains} joints", fontsize=10)
            ax.grid(alpha=0.3, axis="y")
            if j == 0:
                ax.set_ylabel("fraction not settled within 1 cm")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CONTROLLERS), fontsize=8, frameon=False)
    fig.suptitle("Panda: CFDP alone, MPC alone, and CFDP handing over to the MPC on a monitor's alarm", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "handover.png", dpi=130)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--plot-only" not in sys.argv:
        jobs = [(g, w, n, s) for g in m12.GAINS for w in m12.WORLDS for n in CONTROLLERS for s in SPLITS]
        jobs.sort(key=lambda j: j[2] == "policy")  # the long MPC groups first
        with ProcessPoolExecutor() as pool:
            rows = [r for group in pool.map(run_group, jobs) for r in group]
        write("handover_runs.csv", rows)
        summary = summarise(rows)
        write("handover_summary.csv", summary)
        for r in summary:
            print(r)
    plot(read("handover_summary.csv"))


if __name__ == "__main__":
    main()
