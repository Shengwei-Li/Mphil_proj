"""M11a: the Panda platform's basic numbers, before any controller is built on it.

Run:  python experiments/m11a_platform.py     (under a minute)

1. Sag. The robot holds a pose with its gravity compensation configured for no
   load while it really carries 0, 1 or 2.5 kg: how far does the tip sink, in
   each world? This is the size of the effect a controller with a wrong payload
   belief starts from (on the two-link arm, C0 ended about 3 cm off at 1 kg).
2. Exactness. A model told the truth predicts 0.5 s of jittered setpoints;
   largest tip difference from the real robot, per world.
3. Speed. Time to predict 128 candidate setpoint sequences of 0.3 s, with 1 and
   with 16 threads (the MPC's work per round).

Measured on 20 start poses (make_tasks seed 4000). Writes results/m11/platform.csv
and results/m11/panda.png (a snapshot of the scene).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import mujoco
import numpy as np

from lac import panda

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m11"
POSES = [q for q, _ in panda.make_tasks(20, seed=4000)]


def settle(robot: panda.Robot, q: np.ndarray, seconds: float = 2.0) -> None:
    robot.place(q)
    for _ in range(round(seconds / panda.TIMESTEP)):
        robot.step(q)


def sag_rows() -> list[dict]:
    rows = []
    for world in panda.WORLDS:
        for payload in (0.0, 1.0, 2.5):
            sag, comp = [], []
            for q in POSES:
                robot = panda.Robot(payload, world, panda.Belief(0.0))
                robot.place(q)
                start = robot.tip()
                settle(robot, q)
                sag.append(np.linalg.norm(robot.tip() - start))
                comp.append(np.abs(_placed(payload, world, q).gravity_compensation()).max())
            rows.append({"measure": "sag_cm", "world": world, "payload_kg": payload,
                         "median": round(float(np.median(sag)) * 100, 3), "max": round(float(np.max(sag)) * 100, 3),
                         "note": f"largest gravity compensation with the true load: {max(comp):.1f} N·m"})
    return rows


def _placed(payload, world, q) -> panda.Robot:
    robot = panda.Robot(payload, world, panda.world_belief(world, payload))
    robot.place(q)
    return robot


def exactness_rows() -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    for world in panda.WORLDS:
        truth = panda.world_belief(world, 1.5)
        worst = 0.0
        for q in POSES:
            setpoints = q + rng.normal(0.0, 0.2, size=(25, panda.JOINTS))
            robot = panda.Robot(1.5, world, truth)
            robot.place(q)
            for s in setpoints:
                for _ in range(panda.HOLD):
                    robot.step(s)
            tips, _ = panda.Predictor(truth).rollout(q, np.zeros(panda.JOINTS), setpoints[None])
            worst = max(worst, float(np.linalg.norm(tips[0, -1] - robot.tip())))
        rows.append({"measure": "prediction_error_m", "world": world, "payload_kg": 1.5, "median": "",
                     "max": f"{worst:.1e}", "note": "model told the truth, 0.5 s of jittered setpoints"})
    return rows


def speed_rows() -> list[dict]:
    rows = []
    setpoints = np.tile(panda.HOME, (128, 15, 1)) + np.random.default_rng(0).normal(0, 0.1, (128, 15, panda.JOINTS))
    for threads in (1, 16):
        p = panda.Predictor(threads=threads)
        p.rollout(panda.HOME, np.zeros(panda.JOINTS), setpoints)
        began = time.perf_counter()
        for _ in range(20):
            p.rollout(panda.HOME, np.zeros(panda.JOINTS), setpoints)
        ms = (time.perf_counter() - began) / 20 * 1000
        rows.append({"measure": "rollout_ms", "world": "W1", "payload_kg": 0.0, "median": round(ms, 2), "max": "",
                     "note": f"128 sequences x 0.3 s, {threads} thread(s)"})
    return rows


def snapshot() -> None:
    q, target = panda.make_tasks(1, seed=0)[0]
    robot = panda.Robot(2.5, "W1", panda.Belief(0.0))
    robot.place(panda.HOME)
    robot.set_target(target)
    mujoco.mj_forward(robot.model, robot.data)
    camera = mujoco.MjvCamera()
    camera.lookat[:], camera.distance, camera.azimuth, camera.elevation = [0.3, 0.0, 0.35], 1.6, 135, -20
    with mujoco.Renderer(robot.model, 720, 1280) as renderer:
        renderer.update_scene(robot.data, camera)
        import imageio.v3 as iio
        iio.imwrite(OUT / "panda.png", renderer.render())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sag_rows() + exactness_rows() + speed_rows()
    with open(OUT / "platform.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)
    snapshot()


if __name__ == "__main__":
    main()
