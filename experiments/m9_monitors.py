"""M9: which monitor notices that the arm carries an unknown payload?

Run:  python experiments/m9_monitors.py                 (all CPU cores; runs, then analysis)
      python experiments/m9_monitors.py --analyse-only  (re-analyse the saved runs)
      python experiments/m9_monitors.py --analyse-only --noise-scale 3  (noisier sensors)

The policies of M8 (CFDP and NN, settings frozen) are watched by four monitors,
all accumulated the same way (lac.monitor.Cusum) and calibrated the same way:

    physics    residual of the physics model of the unloaded arm (no friction)
    learned    residual of the same model plus a residual network trained on 90 s
               of unloaded data in the training region (data seed 1; M5's frozen
               network settings)
    oracle     residual of a physics model told the world's friction; an upper
               bound for a physics model, not a method
    consistency  Project B's overlapping-chunk disagreement (B's M3)

Each monitor reads angles and speeds with measurement noise (lac.monitor.measured,
1e-4 rad and 0.01 rad/s, fixed in advance). The policy itself sees the clean
state, so the consistency monitor is noise-free; that favours it, not the
residual monitors. ``--noise-scale k`` multiplies both noise levels, for a
robustness check that needs no new simulation.

Four disjoint sets of runs, all with the payload present from the start:

    dev          20 tasks (seed 5500, inside), no payload: fit each monitor's
                 standardisation (median and MAD of log signal). kappa = 0.5 is
                 Project B's and is not tuned.
    calibration  40 tasks (seed 7100, inside), no payload: each run's maximum score
                 gives the split-conformal threshold for alpha = 0.05 and 0.10.
    inside       32 test tasks (seed 5000; the first 8 are M5b's), payloads 0,
                 0.05, 0.1, 0.25, 0.5 and 1 kg: false alarms at 0, detections
                 otherwise.
    outside      32 test tasks (seed 6000, x <= -0.1 m), payloads 0 and 1 kg. The
                 policies fail here even without payload (M8), so an alarm is the
                 useful answer; the question is which monitors give it.

Also reported: "residual or consistency", each at alpha / 2 so the pair keeps
the false-alarm target (union bound), as a candidate trigger for M10.

Writes results/m9/runs.csv, summary.csv, detection.png (with --noise-scale k
the files get a _noise{k} suffix); the simulated runs go to cache/m9_runs.pkl.
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

from lac import data, demos, methods, monitor
from lac.policy import CFDP, NearestNeighborPolicy, PolicyController
from lac.simulate import run_reach

WORLDS = ("W1", "W2", "W3")
POLICIES = {"CFDP": CFDP, "NN": NearestNeighborPolicy}
MONITORS = ("physics", "learned", "oracle", "consistency")
MODEL_OF = {"physics": "C0", "learned": "C2", "oracle": "reference"}  # built for payload 0
RESIDUAL_BUDGET, RESIDUAL_SEED = 90.0, 1
PAYLOADS = {"dev": (0.0,), "calibration": (0.0,), "inside": (0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
            "outside": (0.0, 1.0)}
TASKS = {"dev": methods.make_tasks(20, seed=5500, region="train"),
         "calibration": methods.make_tasks(40, seed=7100, region="train"),
         "inside": methods.make_tasks(32, seed=5000, region="train"),
         "outside": methods.make_tasks(32, seed=6000, region="outside")}
ALPHAS = (0.05, 0.10)
DURATION = 3.0
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m9"
RUNS = ROOT / "cache" / "m9_runs.pkl"


def run_group(job) -> list[dict]:
    """Every run of one world and policy: what a monitor can see, before any monitor looks at it."""
    world, name = job
    policy = POLICIES[name]().fit(demos.load_or_record(world))
    runs = []
    for split, tasks in TASKS.items():
        for payload in PAYLOADS[split]:
            for task, (q_start, target) in enumerate(tasks):
                controller = PolicyController(policy, seed=task)
                rollout = run_reach(controller, payload, q_start, target, DURATION, world)
                runs.append({"world": world, "policy": name, "split": split, "payload_kg": payload, "task": task,
                             "final_error_cm": rollout.final_error() * 100,
                             "periods": monitor.periods(rollout, controller.hold),
                             "chunks": controller.chunks, "chunk_periods": list(controller.chunk_periods)})
    return runs


def add_signals(runs: list[dict], noise_scale: float) -> None:
    """Each monitor's signal for each run; the measurement noise is seeded by the run."""
    for world in WORLDS:
        models = methods.build_predictors(world, 0.0, RESIDUAL_BUDGET, RESIDUAL_SEED)
        for r in (r for r in runs if r["world"] == world):
            key = (WORLDS.index(world), list(POLICIES).index(r["policy"]), list(TASKS).index(r["split"]),
                   round(r["payload_kg"] * 100), r["task"])
            p = monitor.measured(r["periods"], np.random.default_rng(key), 1e-4 * noise_scale, 0.01 * noise_scale)
            r["signals"] = {m: monitor.residual_signal(models[k], p, r["chunk_periods"]) for m, k in MODEL_OF.items()}
            r["signals"]["consistency"] = monitor.chunk_disagreement(r["chunks"], r["chunk_periods"])


def record() -> list[dict]:
    for world in WORLDS:  # demonstrations and residual networks, once, before the workers need them
        demos.load_or_record(world)
        methods.build_predictors(world, 0.0, RESIDUAL_BUDGET, RESIDUAL_SEED, train_missing=True)
    with ProcessPoolExecutor() as pool:
        runs = [r for g in pool.map(run_group, [(w, n) for w in WORLDS for n in POLICIES]) for r in g]
    RUNS.parent.mkdir(exist_ok=True)
    RUNS.write_bytes(pickle.dumps(runs))
    return runs


def analyse(runs: list[dict]) -> list[dict]:
    """Fit on dev, calibrate on calibration, then score the test runs. Returns one row per test run."""
    rows = []
    for world in WORLDS:
        for name in POLICIES:
            group = [r for r in runs if (r["world"], r["policy"]) == (world, name)]
            dev = [r for r in group if r["split"] == "dev"]
            calibration = [r for r in group if r["split"] == "calibration"]
            cusum = {m: monitor.Cusum().fit([r["signals"][m] for r in dev]) for m in MONITORS}

            def scores(r, m):
                return cusum[m].scores(r["signals"][m])

            thresholds = {(m, a): monitor.conformal_threshold([monitor.episode_max(scores(r, m)) for r in calibration], a)
                          for m in MONITORS for a in ALPHAS + tuple(x / 2 for x in ALPHAS)}
            for r in group:
                if r["split"] not in ("inside", "outside"):
                    continue
                row = {k: r[k] for k in ("world", "policy", "split", "payload_kg", "task")}
                row["final_error_cm"] = round(r["final_error_cm"], 4)
                for alpha in ALPHAS:
                    alarms = {}
                    for m in MONITORS:
                        alarms[m] = monitor.first_alarm(scores(r, m), r["chunk_periods"], thresholds[(m, alpha)])
                    for m in ("physics", "learned"):  # residual or consistency, each at alpha / 2
                        a = [monitor.first_alarm(scores(r, x), r["chunk_periods"], thresholds[(x, alpha / 2)])
                             for x in (m, "consistency")]
                        a = [x for x in a if x is not None]
                        alarms[f"{m}+consistency"] = min(a) if a else None
                    for m, period in alarms.items():
                        row[f"alarm_s_{m}_a{alpha:g}"] = "" if period is None else round(period * data.PERIOD, 3)
                rows.append(row)
    return rows


COMBINED = MONITORS + ("physics+consistency", "learned+consistency")


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["split"], r["world"], r["policy"], r["payload_kg"]) for r in rows},
                  key=lambda k: (k[0], k[1], k[2], k[3]))
    for split, world, name, payload in keys:
        g = [r for r in rows if (r["split"], r["world"], r["policy"], r["payload_kg"]) == (split, world, name, payload)]
        for alpha in ALPHAS:
            row = {"split": split, "world": world, "policy": name, "payload_kg": payload, "alpha": alpha,
                   "runs": len(g), "median_final_error_cm": round(float(np.median([r["final_error_cm"] for r in g])), 3)}
            for m in COMBINED:
                times = [r[f"alarm_s_{m}_a{alpha:g}"] for r in g if r[f"alarm_s_{m}_a{alpha:g}"] != ""]
                row[f"alarm_rate_{m}"] = round(len(times) / len(g), 3)
                row[f"median_alarm_s_{m}"] = round(float(np.median(times)), 3) if times else ""
            out.append(row)
    return out


SUFFIX = ""


def write(name: str, rows: list[dict]) -> None:
    name = name.replace(".", f"{SUFFIX}.")
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary: list[dict]) -> None:
    """Alarm rate against payload, inside the training region, alpha = 0.05."""
    styles = {"physics": ("tab:blue", "o"), "learned": ("tab:red", "s"), "oracle": ("k", "x"),
              "consistency": ("tab:green", "^")}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for i, name in enumerate(POLICIES):
        for j, world in enumerate(WORLDS):
            ax = axes[i, j]
            g = sorted((r for r in summary if (r["split"], r["world"], r["policy"], r["alpha"])
                        == ("inside", world, name, 0.05)), key=lambda r: r["payload_kg"])
            x = [r["payload_kg"] for r in g]
            for m, (color, marker) in styles.items():
                ax.plot(x, [r[f"alarm_rate_{m}"] for r in g], color=color, marker=marker, label=m,
                        linestyle=":" if m == "oracle" else "-")
            ax.axhline(0.05, color="grey", linewidth=0.8, linestyle="--")
            ax.set_xscale("symlog", linthresh=0.05)
            ax.set_xticks([0, 0.05, 0.1, 0.25, 0.5, 1.0], ["0", "0.05", "0.1", "0.25", "0.5", "1"])
            ax.set_title(f"{name}, {world}", fontsize=10)
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("payload (kg)")
            if j == 0:
                ax.set_ylabel("alarm rate")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Inside the training region, thresholds for a 5% false-alarm rate (dashed). "
                 "At 0 kg the rate is the false-alarm rate; above it, the detection rate", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"detection{SUFFIX}.png", dpi=130)


def main() -> None:
    global SUFFIX
    OUT.mkdir(parents=True, exist_ok=True)
    noise_scale = float(sys.argv[sys.argv.index("--noise-scale") + 1]) if "--noise-scale" in sys.argv else 1.0
    if noise_scale != 1.0:
        SUFFIX = f"_noise{noise_scale:g}"
    runs = pickle.loads(RUNS.read_bytes()) if "--analyse-only" in sys.argv else record()
    add_signals(runs, noise_scale)
    rows = analyse(runs)
    write("runs.csv", rows)
    summary = summarise(rows)
    write("summary.csv", summary)
    plot(summary)
    for r in summary:
        if r["alpha"] == 0.05:
            print(r["split"], r["world"], r["policy"], r["payload_kg"], r["median_final_error_cm"],
                  {m: r[f"alarm_rate_{m}"] for m in COMBINED})


if __name__ == "__main__":
    main()
