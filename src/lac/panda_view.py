"""Watch the Panda reach with C0, C1 or the reference, in MuJoCo's own 3D viewer.

Run:  python -m lac.panda_view                                  (C1, W1, 2.5 kg)
      python -m lac.panda_view --method C0 --world W3 --payload 1
      python -m lac.panda_view --region outside                 (targets off to the sides)
      python -m lac.panda_view --snapshot results/m11/c0_vs_c1.png   (no window; C0 and C1 side by side)

The robot reaches for one task after another (lac.panda.make_tasks), 3 s each.
The grey trail is where the tip has been. Keys, in the viewer window:

    ]  /  [      next / previous method (C0, C1, reference); restarts the task
    .  /  ,      payload up / down by 0.5 kg; restarts the task
    Enter        next task now

Mouse: left-drag rotates, right-drag moves, scroll zooms (MuJoCo's usual).
The top-left panel shows the method, what it believes, the real payload and
the tip error. When planning cannot keep up with real time the motion is
slowed down; the panel shows simulated time.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mujoco
import numpy as np

from lac import panda
from lac.panda_control import make_controller

METHODS = ("C0", "C1", "reference")
TASK_SECONDS = 3.0
TRAIL_EVERY = 10  # simulator steps between trail points
TRAIL_LENGTH = 150
KEY_ENTER = 257


class Session:
    """One reach at a time; the viewer only draws what this produces."""

    def __init__(self, method: str, world: str, payload: float, region: str, seed: int):
        self.method, self.world, self.payload = method, world, payload
        self.tasks = panda.make_tasks(50, seed=seed, region=region)
        self.task = 0
        self.threads = os.cpu_count() or 1
        self.restart()

    def restart(self) -> None:
        q, target = self.tasks[self.task % len(self.tasks)]
        self.controller = make_controller(self.method, self.world, self.payload, seed=self.task, threads=self.threads)
        if not hasattr(self, "robot"):
            self.robot = panda.Robot(self.payload, self.world, self.controller.load)
        panda.set_payload(self.robot.model, self.payload)  # same model, so an open viewer keeps working
        self.robot.data.time = 0.0
        self.robot.place(q)
        self.robot.set_target(target)
        self.target = target
        self.controller.start(target, self.robot.q)
        self.trail: list[np.ndarray] = []
        self.steps = 0

    def step(self) -> None:
        setpoint = self.controller.setpoint(self.robot.q, self.robot.dq)
        self.robot.configure_load(self.controller.load)
        tau = self.robot.step(setpoint)
        if hasattr(self.controller, "observe"):
            self.controller.observe(tau)
        self.steps += 1
        if self.steps % TRAIL_EVERY == 0:
            self.trail = (self.trail + [self.robot.tip()])[-TRAIL_LENGTH:]

    def error_cm(self) -> float:
        return float(np.linalg.norm(self.robot.tip() - self.target) * 100)

    def panel(self) -> tuple[str, str]:
        belief = self.controller.load.payload_mass
        left = "method\nbelieves\nreal payload\nworld\ntip error\nsim time\ntask"
        right = (f"{self.method}\n{belief:.2f} kg\n{self.payload:.2f} kg\n{self.world}\n{self.error_cm():.2f} cm\n"
                 f"{self.robot.data.time:.2f} s\n{self.task % len(self.tasks) + 1}")
        return left, right


def add_trail(scene: mujoco.MjvScene, trail: list[np.ndarray]) -> None:
    for point in trail:
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.004, 0, 0], point,
                            np.eye(3).ravel(), np.array([0.3, 0.3, 0.3, 0.6], dtype=np.float32))
        scene.ngeom += 1


def view(args) -> None:
    import mujoco.viewer

    session = Session(args.method, args.world, args.payload, args.region, args.seed)
    pending: list[int] = []
    with mujoco.viewer.launch_passive(session.robot.model, session.robot.data,
                                      key_callback=pending.append) as viewer:
        viewer.cam.lookat[:], viewer.cam.distance = [0.35, 0.0, 0.35], 1.8
        viewer.cam.azimuth, viewer.cam.elevation = 135, -20
        wall_start, sim_start = time.perf_counter(), 0.0
        while viewer.is_running():
            restart = False
            while pending:
                key = pending.pop(0)
                if key in (ord("]"), ord("[")):
                    i = METHODS.index(session.method) + (1 if key == ord("]") else -1)
                    session.method = METHODS[i % len(METHODS)]
                    restart = True
                elif key in (ord("."), ord(",")):
                    session.payload = float(np.clip(session.payload + (0.5 if key == ord(".") else -0.5), 0.0, 3.0))
                    restart = True
                elif key == KEY_ENTER:
                    session.task += 1
                    restart = True
            if session.robot.data.time >= TASK_SECONDS:
                session.task += 1
                restart = True
            if restart:
                with viewer.lock():
                    session.restart()
                wall_start, sim_start = time.perf_counter(), 0.0
            with viewer.lock():
                for _ in range(panda.HOLD):
                    session.step()
                viewer.user_scn.ngeom = 0
                add_trail(viewer.user_scn, session.trail)
                viewer.set_texts((None, None, *session.panel()))
            viewer.sync()
            ahead = (session.robot.data.time - sim_start) - (time.perf_counter() - wall_start)
            if ahead > 0:
                time.sleep(ahead)


def snapshot(args) -> None:
    """C0 and C1 on the same task after 2 s, side by side, saved to a PNG."""
    import matplotlib.pyplot as plt

    frames, titles = [], []
    for method in ("C0", "C1"):
        session = Session(method, args.world, args.payload, args.region, args.seed)
        while session.robot.data.time < 2.0:
            session.step()
        model, data = session.robot.model, session.robot.data
        model.geom("payload_geom").rgba[3] = 0.15  # faint, so the target inside it shows
        mujoco.mj_forward(model, data)
        camera = mujoco.MjvCamera()
        camera.lookat[:], camera.distance, camera.azimuth, camera.elevation = session.target, 0.45, 100, -5
        with mujoco.Renderer(model, 540, 640) as renderer:
            renderer.update_scene(data, camera)
            add_trail(renderer.scene, session.trail)
            frames.append(renderer.render())
        titles.append(f"{method}: believes {session.controller.load.payload_mass:.2f} kg, "
                      f"tip {session.error_cm():.2f} cm from the target")
        print(titles[-1])
    Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.9))
    for ax, frame, title in zip(axes, frames, titles):
        ax.imshow(frame)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{args.payload:g} kg payload, {args.world}, after 2 s (green: target; grey dots: path of the tip)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(args.snapshot, dpi=100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method", choices=METHODS, default="C1")
    parser.add_argument("--world", choices=tuple(panda.WORLDS), default="W1")
    parser.add_argument("--payload", type=float, default=2.5)
    parser.add_argument("--region", choices=tuple(panda.REGIONS), default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshot", help="save C0 and C1 side by side to this PNG instead of opening a window")
    args = parser.parse_args()
    snapshot(args) if args.snapshot else view(args)


if __name__ == "__main__":
    main()
