"""M12c, step 2: CFDP on the Panda and the monitors that watch it (M8 and M9 on the Panda).

Run:  python experiments/m12_demos.py               (first: the demonstrations)
      python experiments/m12c_monitors.py --dev     (development check, unloaded, before any test run)
      python experiments/m12c_monitors.py           (all runs and the monitor analysis, all cores)
      python experiments/m12c_monitors.py --analyse-only

The policy is CFDP with M8's settings, on the Panda's observations and actions
(lac.panda_policy), one per world (W2 shares W1's demonstrations) and joint
stiffness. Monitors (lac.panda_monitor): physics, learned (network trained on the
same demonstrations, M5's frozen settings), oracle, consistency; M9's recipe
(standardisation on dev, split-conformal thresholds, measurement noise).

Splits, payload present from the start, MPC-free:

    dev          20 tasks (seed 5500, inside), no payload: standardisation
    calibration  100 tasks (seed 7100, inside), no payload: thresholds (100, as M10)
    inside       16 tasks (seed 5000), payloads 0, 0.1, 0.25, 1 and 2.5 kg
    outside      16 tasks (seed 6000), payloads 0 and 2.5 kg

The expectations H20-H23 were written in docs/PROGRESS.md before the test runs.
Writes results/m12c/runs.csv, summary.csv and detection.png; the runs go to
cache/m12c_runs.pkl (the dev and calibration runs are reused by m12c_handover.py).
"""

from __future__ import annotations

import csv
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lac import data, monitor, panda
from lac import panda_monitor as pm
from lac import panda_policy as pp
from lac.policy import CFDP

GAINS = ("stiff", "soft")
WORLDS = ("W1", "W2", "W3")
MONITORS = ("physics", "learned", "oracle", "consistency")
TASKS = {"dev": panda.make_tasks(20, seed=5500, region="train"),
         "calibration": panda.make_tasks(100, seed=7100, region="train"),
         "inside": panda.make_tasks(16, seed=5000, region="train"),
         "outside": panda.make_tasks(16, seed=6000, region="outside")}
PAYLOADS = {"dev": (0.0,), "calibration": (0.0,), "inside": (0.0, 0.1, 0.25, 1.0, 2.5), "outside": (0.0, 2.5)}
ALPHA = 0.05
DURATION = 3.0
SETTLED = 0.01
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m12c"
RUNS = data.CACHE / "m12c_runs.pkl"
CLEAN = data.CACHE / "m12c_clean.pkl"  # the dev and calibration runs alone, for m12c_handover.py


@lru_cache(maxsize=4)
def policy(world: str, gains: str) -> CFDP:
    return pp.fit_policy(CFDP(), pp.load_or_record(world, gains))


def noise_seed(gains, world, split, payload, task) -> int:
    return (((GAINS.index(gains) * 3 + WORLDS.index(world)) * 4 + list(TASKS).index(split)) * 1000
            + round(payload * 100)) * 1000 + task


def run_group(job) -> list[dict]:
    gains, world, split, payload = job
    runs = []
    for task, (q_start, target) in enumerate(TASKS[split]):
        c = pp.PandaPolicyController(policy(world, gains), seed=task, gains=gains)
        r = panda.run_reach(c, payload, q_start, target, DURATION, world)
        above = np.nonzero(r.error > SETTLED)[0]
        runs.append({"gains": gains, "world": world, "split": split, "payload_kg": payload, "task": task,
                     "final_error_cm": r.final_error() * 100,
                     "settled": len(above) == 0 or above[-1] < len(r.error) - 1,
                     "periods": pm.periods(r), "chunks": c.chunks, "chunk_periods": list(c.chunk_periods),
                     "noise_seed": noise_seed(gains, world, split, payload, task),
                     "solve_ms": float(np.mean(c.solve_seconds)) * 1000})
    return runs


def dev_check() -> None:
    jobs = [(g, w, "dev", 0.0) for g in GAINS for w in WORLDS]
    with ProcessPoolExecutor() as pool:
        groups = list(pool.map(run_group, jobs))
    rows = []
    for (g, w, _, _), runs in zip(jobs, groups):
        e = [r["final_error_cm"] for r in runs]
        rows.append({"gains": g, "world": w, "median_final_error_cm": round(float(np.median(e)), 3),
                     "worst_final_error_cm": round(max(e), 3), "runs_not_settled": sum(not r["settled"] for r in runs),
                     "mean_solve_ms": round(float(np.mean([r["solve_ms"] for r in runs])), 1)})
        print(rows[-1])
    write("dev.csv", rows)


def train_nets() -> None:
    """The learned monitor's networks, one per robot (W2 = W1) and stiffness, before the workers need them."""
    jobs = [(w, g) for g in GAINS for w in ("W1", "W3")]
    with ProcessPoolExecutor(len(jobs)) as pool:
        list(pool.map(_train, jobs))


def _train(job) -> None:
    world, gains = job
    pm.load_or_train_net(world, gains, pp.load_or_record(world, gains))


def analyse(runs: list[dict]) -> list[dict]:
    rows = []
    for gains in GAINS:
        for world in WORLDS:
            group = [r for r in runs if (r["gains"], r["world"]) == (gains, world)]
            models = pm.monitor_models(world, gains, pp.load_or_record(world, gains))
            signals = {m: pm.residual(models[m]) for m in ("physics", "learned", "oracle")}
            signals["consistency"] = pm.consistency()
            dev = [r for r in group if r["split"] == "dev"]
            calibration = [r for r in group if r["split"] == "calibration"]
            detectors = {m: pm.fit_detector(m, signals[m], dev, calibration, ALPHA) for m in MONITORS}
            for r in group:
                if r["split"] not in ("inside", "outside"):
                    continue
                p = pm.measured(r["periods"], np.random.default_rng(r["noise_seed"]))
                row = {k: r[k] for k in ("gains", "world", "split", "payload_kg", "task")}
                row["final_error_cm"] = round(r["final_error_cm"], 4)
                row["settled"] = int(r["settled"])
                for m, d in detectors.items():
                    period = monitor.first_alarm(d.scores(p, r["chunks"], r["chunk_periods"]), r["chunk_periods"],
                                                 d.threshold)
                    row[f"alarm_s_{m}"] = "" if period is None else round(period * panda.PERIOD, 3)
                rows.append(row)
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for split in ("inside", "outside"):
        for gains in GAINS:
            for world in WORLDS:
                for payload in PAYLOADS[split]:
                    g = [r for r in rows if (r["split"], r["gains"], r["world"], r["payload_kg"])
                         == (split, gains, world, payload)]
                    row = {"split": split, "gains": gains, "world": world, "payload_kg": payload, "runs": len(g),
                           "median_final_error_cm": round(float(np.median([r["final_error_cm"] for r in g])), 3),
                           "runs_not_settled": sum(1 - r["settled"] for r in g)}
                    for m in MONITORS:
                        times = [r[f"alarm_s_{m}"] for r in g if r[f"alarm_s_{m}"] != ""]
                        row[f"alarm_rate_{m}"] = round(len(times) / len(g), 3)
                        row[f"median_alarm_s_{m}"] = round(float(np.median(times)), 2) if times else ""
                    out.append(row)
    return out


def write(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary: list[dict]) -> None:
    styles = {"physics": ("tab:blue", "o"), "learned": ("tab:red", "s"), "oracle": ("k", "x"),
              "consistency": ("tab:green", "^")}
    fig, axes = plt.subplots(len(GAINS), len(WORLDS), figsize=(13, 7), sharex=True, sharey=True)
    for i, gains in enumerate(GAINS):
        for j, world in enumerate(WORLDS):
            ax = axes[i, j]
            g = sorted((r for r in summary if (r["split"], r["gains"], r["world"]) == ("inside", gains, world)),
                       key=lambda r: r["payload_kg"])
            for m, (color, marker) in styles.items():
                ax.plot([r["payload_kg"] for r in g], [r[f"alarm_rate_{m}"] for r in g], color=color, marker=marker,
                        label=m, linestyle=":" if m == "oracle" else "-")
            ax.axhline(ALPHA, color="grey", linewidth=0.8, linestyle="--")
            ax.set_xscale("symlog", linthresh=0.1)
            ax.set_xticks(PAYLOADS["inside"], [f"{p:g}" for p in PAYLOADS["inside"]])
            ax.set_title(f"{world}, {gains} joints", fontsize=10)
            ax.grid(alpha=0.3)
            if i == len(GAINS) - 1:
                ax.set_xlabel("payload (kg)")
            if j == 0:
                ax.set_ylabel("alarm rate")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Panda, CFDP inside the training region, thresholds for a 5% false-alarm rate (dashed): "
                 "at 0 kg the false-alarm rate, above it the detection rate", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "detection.png", dpi=130)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--dev" in sys.argv:
        dev_check()
        return
    if "--analyse-only" in sys.argv:
        runs = pickle.loads(RUNS.read_bytes())
    else:
        jobs = [(g, w, s, p) for g in GAINS for w in WORLDS for s in TASKS for p in PAYLOADS[s]]
        with ProcessPoolExecutor() as pool:
            runs = [r for group in pool.map(run_group, jobs) for r in group]
        RUNS.write_bytes(pickle.dumps(runs))
        CLEAN.write_bytes(pickle.dumps([r for r in runs if r["split"] in ("dev", "calibration")]))
    train_nets()
    rows = analyse(runs)
    write("runs.csv", rows)
    summary = summarise(rows)
    write("summary.csv", summary)
    plot(summary)
    for r in summary:
        print(r["split"], r["gains"], r["world"], r["payload_kg"], r["median_final_error_cm"], r["runs_not_settled"],
              {m: r[f"alarm_rate_{m}"] for m in MONITORS})


if __name__ == "__main__":
    main()
