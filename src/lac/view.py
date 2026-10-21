"""Interactive panel: pose the arm, release it, or let a PD controller reach the target.

Run:  python -m lac.view
      python -m lac.view --q1 -60 --q2 60 --payload 0.5
      python -m lac.view --snapshot results/m0/pose.png     (no window, saves one image)
      python -m lac.view --native                           (MuJoCo's own 3D viewer, no controller)

In "pose" mode the arm is simply placed at the slider angles. Pressing run
starts the physics with the chosen control method. While it runs, the payload,
target and gains can be changed live; moving an angle slider goes back to pose mode.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import mujoco
import numpy as np

from lac import arm
from lac.control import PDController, nearest_solution
from lac.model import Belief
from lac.mpc import MPCController
from lac.predictors import PhysicsPredictor

RENDER_SIZE = 600  # pixels, square
FRAME_MS = 33  # requested screen refresh interval; the real one is measured
MAX_FRAME_GAP = 0.2  # seconds; a longer stall (e.g. dragging the window) is not replayed
TRAIL_EVERY = 10  # record the tip every this many physics steps
TRAIL_LENGTH = 400  # number of past tip positions drawn

# Control methods offered in the panel: label -> (kind, payload the controller assumes).
# For PD the assumed payload is used for gravity compensation; None switches it off.
METHODS = {
    "无（松手，电机关闭）": ("off", None),
    "PD": ("pd", None),
    "PD＋重力补偿（按空载算）": ("pd", 0.0),
    "MPC（模型按空载算）": ("mpc", 0.0),
    "MPC（告知真实负载，参考）": ("mpc", "true"),
}


class ArmPanel:
    def __init__(self, q_deg: tuple[float, float], payload: float, target: tuple[float, float],
                 method: int = 0, kp: float = 60.0, kd: float = 6.0, world: str = "W1"):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, Slider

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        self.model, self.data = arm.load_arm()
        self.world = world
        arm.set_world(self.model, world)
        self.renderer = mujoco.Renderer(self.model, RENDER_SIZE, RENDER_SIZE)
        self.initial = (q_deg, payload, target, kp, kd)
        self.running = False
        self.last_tick = 0.0
        self.frame_seconds = 0.0
        self.trail: list[np.ndarray] = []
        self.goal: np.ndarray | None = None  # joint-angle goal while a controller runs
        self.reachable = True
        self.torque = np.zeros(2)
        self.controller: PDController | MPCController | None = None

        self.fig = plt.figure(figsize=(12, 7.6))
        self.fig.canvas.manager.set_window_title("Project C · two-link arm")
        self.ax_img = self.fig.add_axes([0.01, 0.20, 0.48, 0.78])
        self.ax_img.axis("off")
        self.image = self.ax_img.imshow(np.zeros((RENDER_SIZE, RENDER_SIZE, 3), dtype=np.uint8))
        self.ax_text = self.fig.add_axes([0.53, 0.50, 0.46, 0.48])
        self.ax_text.axis("off")
        self.text = self.ax_text.text(0, 1, "", va="top", fontsize=10, linespacing=1.45)

        ax_radio = self.fig.add_axes([0.02, 0.01, 0.27, 0.175])
        ax_radio.set_title("按「运行」后使用的控制方法", fontsize=9, loc="left")
        self.radio = RadioButtons(ax_radio, list(METHODS), active=method)

        def slider(row: int, label: str, lo: float, hi: float, init: float, step: float) -> Slider:
            ax = self.fig.add_axes([0.70, 0.43 - 0.047 * row, 0.24, 0.028])
            return Slider(ax, label, lo, hi, valinit=init, valstep=step)

        self.s_q1 = slider(0, "起始肩关节角 q1 (°)", -180, 180, q_deg[0], 1)
        self.s_q2 = slider(1, "起始肘关节角 q2 (°)", -150, 150, q_deg[1], 1)
        self.s_payload = slider(2, "末端负载 (kg)", 0, 2, payload, 0.05)
        self.s_tx = slider(3, "目标 x (m)", -0.6, 0.6, target[0], 0.01)
        self.s_tz = slider(4, "目标 z (m)", -0.6, 0.6, target[1], 0.01)
        self.s_kp = slider(5, "Kp (N·m/rad)", 0, 300, kp, 5)
        self.s_kd = slider(6, "Kd (N·m·s/rad)", 0, 30, kd, 0.5)
        self.s_speed = slider(7, "播放速度 (倍)", 0.1, 1.0, 1.0, 0.1)
        # Angle sliders always return to pose mode. Payload and target act live while
        # running. Gains and speed are simply read every frame.
        for s in (self.s_q1, self.s_q2):
            s.on_changed(self._on_pose_slider)
        for s in (self.s_payload, self.s_tx, self.s_tz):
            s.on_changed(self._on_live_slider)

        self.b_run = Button(self.fig.add_axes([0.31, 0.11, 0.17, 0.06]), "运行")
        self.b_run.on_clicked(self._toggle_run)
        self.b_reset = Button(self.fig.add_axes([0.31, 0.03, 0.17, 0.06]), "重置")
        self.b_reset.on_clicked(self._reset)

        self.timer = self.fig.canvas.new_timer(interval=FRAME_MS)
        self.timer.add_callback(self._tick)
        self._apply_sliders()

    # --- state changes -------------------------------------------------

    @property
    def method(self) -> str:
        return self.radio.value_selected

    def _apply_payload_and_target(self) -> None:
        arm.set_payload(self.model, self.s_payload.val)
        arm.set_target(self.model, self.data, np.array([self.s_tx.val, self.s_tz.val]))

    def _update_goal(self) -> None:
        """Turn the target point into joint angles, keeping the elbow on its current side."""
        goal = nearest_solution(arm.target_position(self.data), self.data.qpos)
        self.reachable = goal is not None
        if goal is not None:
            self.goal = goal
        elif self.goal is None:
            self.goal = self.data.qpos.copy()
        if isinstance(self.controller, PDController):
            self.controller.set_goal(self.goal)
        elif isinstance(self.controller, MPCController):
            self.controller.retarget(arm.target_position(self.data))

    def _make_controller(self) -> PDController | MPCController | None:
        kind, assumed = METHODS[self.method]
        if kind == "off":
            return None
        if kind == "pd":
            controller = PDController(self.s_kp.val, self.s_kd.val, assumed)
            controller.set_goal(self.goal)
            return controller
        controller = MPCController(predictor=self._reference_predictor() if assumed == "true" else None,
                                   assumed_payload=0.0 if assumed == "true" else assumed)
        controller.start(arm.target_position(self.data), self.data.qpos)
        return controller

    def _reference_predictor(self) -> PhysicsPredictor:
        """The reference option: the panel hands the controller the real payload and world."""
        return PhysicsPredictor(Belief(payload_mass=self.s_payload.val, **arm.WORLDS[self.world]))

    def _apply_sliders(self) -> None:
        """Pose mode: put the arm exactly where the sliders say, at rest."""
        self._apply_payload_and_target()
        arm.set_joint_angles(self.model, self.data, np.radians([self.s_q1.val, self.s_q2.val]))
        self.data.time = 0.0
        self.trail.clear()
        self.goal = None
        self._update_goal()
        self._draw()

    def _on_pose_slider(self, _value: float) -> None:
        self._set_running(False)
        self._apply_sliders()

    def _on_live_slider(self, _value: float) -> None:
        if not self.running:
            self._apply_sliders()
            return
        self._apply_payload_and_target()
        self._update_goal()

    def _set_running(self, running: bool) -> None:
        self.running = running
        self.b_run.label.set_text("停住（回到起始姿势）" if running else "运行")
        if running:
            self.last_tick = 0.0
            self.controller = self._make_controller()
            self.timer.start()
        else:
            self.controller = None
            self.timer.stop()

    def _toggle_run(self, _event) -> None:
        self._set_running(not self.running)
        if not self.running:
            self._apply_sliders()

    def _reset(self, _event) -> None:
        self._set_running(False)
        q_deg, payload, target, kp, kd = self.initial
        for s, v in ((self.s_q1, q_deg[0]), (self.s_q2, q_deg[1]), (self.s_payload, payload),
                     (self.s_tx, target[0]), (self.s_tz, target[1]), (self.s_kp, kp), (self.s_kd, kd),
                     (self.s_speed, 1.0)):
            s.eventson = False
            s.set_val(v)
            s.eventson = True
        self._apply_sliders()

    def _tick(self) -> None:
        """Advance physics by the wall-clock time since the last frame.

        Drawing a frame takes longer than FRAME_MS on most machines, so stepping by
        measured time (not by a fixed amount per frame) keeps the playback speed honest.
        """
        now = time.perf_counter()
        elapsed = min(now - self.last_tick, MAX_FRAME_GAP) if self.last_tick else FRAME_MS / 1000.0
        self.last_tick = now
        self.frame_seconds = elapsed

        controller = self.controller
        if isinstance(controller, PDController):
            controller.kp, controller.kd = self.s_kp.val, self.s_kd.val
        elif isinstance(controller, MPCController) and METHODS[self.method][1] == "true":
            controller.predictor = self._reference_predictor()

        steps = max(1, round(elapsed * self.s_speed.val / self.model.opt.timestep))
        for i in range(steps):
            self.torque = controller.torque(self.data.qpos, self.data.qvel) if controller else np.zeros(2)
            self.data.ctrl[:] = self.torque
            arm.step(self.model, self.data)
            if i % TRAIL_EVERY == 0:
                x, z = arm.forward_kinematics(self.data.qpos)
                self.trail.append(np.array([x, 0.0, z]))
        mujoco.mj_forward(self.model, self.data)  # refresh positions for drawing and read-outs
        del self.trail[:-TRAIL_LENGTH]
        self._draw()

    # --- drawing -------------------------------------------------------

    def _render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="side")
        # Red: where the tip has been. Blue: where the MPC's model expects its plan to take it.
        dots = [(point, (0.9, 0.2, 0.2, 1.0), 0.005) for point in self.trail]
        if isinstance(self.controller, MPCController):
            dots += [(np.array([x, 0.0, z]), (0.1, 0.3, 0.9, 1.0), 0.007) for x, z in self.controller.planned_tip]
        scene = self.renderer.scene
        for point, rgba, radius in dots:
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[radius, 0, 0],
                pos=point - np.array([0.0, 0.03, 0.0]),  # slightly towards the camera
                mat=np.eye(3).flatten(),
                rgba=np.array(rgba, dtype=np.float32),
            )
            scene.ngeom += 1
        return self.renderer.render()

    def _mpc_status(self) -> str:
        """How long one MPC round takes to compute, against the time it has (its period)."""
        controller = self.controller
        if not isinstance(controller, MPCController) or not controller.solve_seconds:
            return "MPC 计算：—"
        solve_ms = 1000 * float(np.median(controller.solve_seconds[-50:]))
        return f"MPC 计算：每轮 {solve_ms:.1f} 毫秒（每 {1000 * controller.period:.0f} 毫秒算一轮）；蓝点是它预测的未来 {controller.steps * controller.period:.1f} 秒"

    def _status(self) -> str:
        q = np.degrees(self.data.qpos)
        dq = np.degrees(self.data.qvel)
        tip = arm.tip_position(self.data)
        target = arm.target_position(self.data)
        error = float(np.linalg.norm(tip - target))
        limits = self.model.actuator_ctrlrange[:, 1]
        goal = np.degrees(self.goal)
        lines = [
            f"模式：{'运行中' if self.running else '摆姿势，没有物理运动'}",
            f"控制方法：{self.method if self.running else '—（按「运行」后生效）'}",
            f"仿真时间：{self.data.time:6.2f} s    播放速度：{self.s_speed.val:.1f} 倍",
            f"画面刷新：{1.0 / self.frame_seconds:4.1f} 帧/秒" if self.running and self.frame_seconds else "画面刷新：—",
            self._mpc_status(),
            "",
            f"关节角度：q1 = {q[0]:7.1f}°   q2 = {q[1]:7.1f}°",
            f"关节角速度：{dq[0]:7.1f}°/s   {dq[1]:7.1f}°/s",
            f"目标对应的关节角：{goal[0]:7.1f}°   {goal[1]:7.1f}°" if self.reachable else "目标点够不着（离肩超过 0.6 m）",
            f"末端位置：x = {tip[0]:+.3f} m   z = {tip[1]:+.3f} m",
            f"目标位置：x = {target[0]:+.3f} m   z = {target[1]:+.3f} m",
            f"末端误差（到目标的距离）：{error * 100:.1f} cm",
            f"末端负载：{arm.get_payload(self.model):.2f} kg    真实世界：{self.world}",
            "",
        ]
        if self.running:
            lines.append("电机此刻出的力矩：")
            shown = self.torque
        else:
            lines.append("要让这个姿势静止不动，电机需要出的力矩：")
            shown = arm.gravity_torque(self.model, self.data)
        for name, t, limit in zip(("肩", "肘"), shown, limits):
            if self.running:
                flag = "   ← 已到电机上限" if abs(t) >= limit - 1e-9 else ""
            else:
                flag = "   ← 超过电机上限" if abs(t) > limit else ""
            lines.append(f"  {name}：{t:+6.2f} N·m（上限 ±{limit:.0f}）{flag}")
        return "\n".join(lines)

    def _draw(self) -> None:
        self.image.set_data(self._render())
        self.text.set_text(self._status())
        self.fig.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--q1", type=float, default=-60.0, help="starting shoulder angle in degrees")
    parser.add_argument("--q2", type=float, default=60.0, help="starting elbow angle in degrees")
    parser.add_argument("--payload", type=float, default=0.0, help="payload mass in kg")
    parser.add_argument("--target", type=float, nargs=2, default=(0.35, 0.25), metavar=("X", "Z"))
    parser.add_argument("--method", type=int, default=1, choices=range(len(METHODS)),
                        help="control method preselected in the panel: 0 off, 1 PD, 2 PD + gravity compensation, "
                             "3 MPC assuming no payload, 4 MPC told the real payload")
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=6.0)
    parser.add_argument("--world", default="W1", choices=sorted(arm.WORLDS),
                        help="how else the real arm differs from the controller's model (see lac.arm.WORLDS)")
    parser.add_argument("--snapshot", type=Path, help="save one image of the panel and exit")
    parser.add_argument("--native", action="store_true", help="open MuJoCo's built-in viewer instead")
    args = parser.parse_args()

    if args.native:
        import mujoco.viewer

        model, data = arm.load_arm()
        arm.set_world(model, args.world)
        arm.set_payload(model, args.payload)
        arm.set_target(model, data, np.array(args.target))
        arm.set_joint_angles(model, data, np.radians([args.q1, args.q2]))
        mujoco.viewer.launch(model, data)
        return

    if args.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel = ArmPanel((args.q1, args.q2), args.payload, tuple(args.target), args.method, args.kp, args.kd, args.world)
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        panel.fig.savefig(args.snapshot, dpi=110)
        print(f"saved {args.snapshot}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
