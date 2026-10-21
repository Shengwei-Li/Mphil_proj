"""The controller's own model of the arm: hand-written two-link dynamics.

This is what the controller *believes* about the arm. It never reads the
simulator. The unloaded arm is assumed to be known exactly; everything about
the payload (and, for reference models, friction) comes from a `Belief`
supplied by the caller, and it may be wrong.

Equation of motion (q = joint angles, dq = velocities, tau = motor torques):

    M(q) ddq + c(q, dq) + g(q) + damping * dq + coulomb(dq) = tau

    M        inertia matrix   how hard it is to accelerate each joint in this pose
    c        velocity terms   one joint's motion pushing on the other (Coriolis, centrifugal)
    g        gravity
    damping  friction proportional to speed
    coulomb  friction of roughly constant size that opposes motion (zero unless believed)

Wherever a function takes `payload`, a plain number means "a point mass of
this many kg exactly at the tip, no extra friction"; a Belief can say more.

All functions accept arrays whose last axis is the two joints, so many
states can be evaluated at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from lac.arm import FRICTION_SMOOTHING, LINK_LENGTHS

LINK_MASSES = (1.0, 1.0)  # kg
LINK_COM = (0.15, 0.15)  # m, distance from each link's joint to its centre of mass
LINK_INERTIAS = (0.0075, 0.0075)  # kg·m^2 about each link's centre of mass (uniform rod, m*l^2/12)
JOINT_DAMPING = 0.1  # N·m·s/rad
TORQUE_LIMITS = np.array([20.0, 10.0])  # N·m, shoulder and elbow
GRAVITY = 9.81
TIMESTEP = 0.002  # s, same as the simulator


@dataclass(frozen=True)
class Belief:
    """What the controller assumes about the parts of the arm it is unsure of."""

    payload_mass: float = 0.0  # kg
    payload_offset: tuple[float, float] = (0.0, 0.0)  # m from the tip: (along link 2, perpendicular)
    extra_damping: tuple[float, float] = (0.0, 0.0)  # N·m·s/rad on top of JOINT_DAMPING
    coulomb: tuple[float, float] = (0.0, 0.0)  # N·m


def as_belief(payload) -> Belief:
    return payload if isinstance(payload, Belief) else Belief(payload_mass=float(payload))


@lru_cache(maxsize=256)
def _forearm(payload):
    """Link 2 and the payload combined into one rigid body.

    Returns its mass, the distance from the elbow to its centre of mass, the
    angle by which that centre of mass sits off the link's axis, and its
    inertia about the centre of mass. With the payload exactly at the tip the
    angle is zero and this is the familiar uniform-rod-plus-point-mass case.

    The result depends only on the belief, so it is remembered (lru_cache)
    instead of being recomputed at every one of the MPC's many small steps.
    """
    b = as_belief(payload)
    l2 = LINK_LENGTHS[1]
    m2, lc2, i2 = LINK_MASSES[1], LINK_COM[1], LINK_INERTIAS[1]
    mp = b.payload_mass
    px, pz = l2 + b.payload_offset[0], b.payload_offset[1]  # payload position in the link's frame
    m = m2 + mp
    cx, cz = (m2 * lc2 + mp * px) / m, mp * pz / m
    inertia = i2 + m2 * ((lc2 - cx) ** 2 + cz**2) + mp * ((px - cx) ** 2 + (pz - cz) ** 2)
    return m, np.hypot(cx, cz), np.arctan2(cz, cx), inertia


def mass_matrix(q, payload=0.0):
    """Entries (M11, M12, M22) of the symmetric 2x2 inertia matrix."""
    l1 = LINK_LENGTHS[0]
    m1, lc1, i1 = LINK_MASSES[0], LINK_COM[0], LINK_INERTIAS[0]
    m2, lc2, angle, i2 = _forearm(payload)
    c2 = np.cos(q[..., 1] + angle)
    m11 = i1 + i2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2)
    m12 = i2 + m2 * (lc2**2 + l1 * lc2 * c2)
    m22 = i2 + m2 * lc2**2
    return m11, m12, m22


def velocity_terms(q, dq, payload=0.0):
    """Coriolis and centrifugal torques c(q, dq)."""
    l1 = LINK_LENGTHS[0]
    m2, lc2, angle, _ = _forearm(payload)
    h = m2 * l1 * lc2 * np.sin(q[..., 1] + angle)
    dq1, dq2 = dq[..., 0], dq[..., 1]
    return np.stack([-h * (2 * dq1 * dq2 + dq2**2), h * dq1**2], axis=-1)


def gravity(q, payload=0.0):
    """Gravity torques g(q): what the motors must supply to hold pose `q` still.

    Each joint carries the weight of everything beyond it, times the
    horizontal distance from the joint to that weight.
    """
    l1 = LINK_LENGTHS[0]
    m1, lc1 = LINK_MASSES[0], LINK_COM[0]
    m2, lc2, angle, _ = _forearm(payload)
    elbow = GRAVITY * m2 * lc2 * np.cos(q[..., 0] + q[..., 1] + angle)
    shoulder = GRAVITY * (m1 * lc1 + m2 * l1) * np.cos(q[..., 0]) + elbow
    return np.stack([shoulder, elbow], axis=-1)


@lru_cache(maxsize=256)
def _friction_coefficients(payload):
    """Viscous and Coulomb coefficients per joint, remembered per belief. Read-only arrays."""
    b = as_belief(payload)
    viscous = JOINT_DAMPING + np.asarray(b.extra_damping, dtype=float)
    coulomb = np.asarray(b.coulomb, dtype=float)
    viscous.flags.writeable = coulomb.flags.writeable = False
    return viscous, coulomb


def damping(payload=0.0) -> np.ndarray:
    """Viscous friction coefficient of each joint, N·m·s/rad."""
    return _friction_coefficients(payload)[0]


def coulomb_friction(dq, payload=0.0):
    """Coulomb friction torque at speed dq, smoothed near zero speed the same way as the real arm."""
    return -_friction_coefficients(payload)[1] * np.tanh(dq / FRICTION_SMOOTHING)


def _acceleration(q, dq, tau, payload, dt):
    """Solve (M + dt * D) ddq = tau - c - g - D dq + coulomb for ddq (2x2, closed form).

    D is the diagonal damping. With dt = 0 this is the equation of motion as
    written; dt > 0 evaluates the damping at the end of a step (see predict).
    """
    d = damping(payload)
    m11, m12, m22 = mass_matrix(q, payload)
    m11, m22 = m11 + dt * d[0], m22 + dt * d[1]
    rhs = tau - velocity_terms(q, dq, payload) - gravity(q, payload) - d * dq + coulomb_friction(dq, payload)
    r1, r2 = rhs[..., 0], rhs[..., 1]
    det = m11 * m22 - m12**2
    return np.stack([(m22 * r1 - m12 * r2) / det, (m11 * r2 - m12 * r1) / det], axis=-1)


def acceleration(q, dq, tau, payload=0.0):
    """Joint accelerations ddq from the equation of motion."""
    return _acceleration(q, dq, tau, payload, 0.0)


def predict(q, dq, tau, payload=0.0, dt=TIMESTEP):
    """State after one step of length `dt` with torque `tau` held constant.

    Update the velocity from the acceleration first, then the angle from the
    new velocity. Damping is evaluated at the new velocity rather than the old
    one, which amounts to adding dt * damping to the diagonal of M; the
    simulator integrates the same way, so the two agree step for step.
    Coulomb friction uses the old velocity, as the real arm does.
    """
    dq_next = dq + _acceleration(q, dq, tau, payload, dt) * dt
    return q + dq_next * dt, dq_next


def rollout(q, dq, torques, payload=0.0, dt=TIMESTEP):
    """Predict forward through a torque sequence without looking at the real arm.

    `torques` has shape (steps, ..., 2). Returns angles and velocities of shape
    (steps + 1, ..., 2), starting with the given state.
    """
    qs, dqs = [np.asarray(q, dtype=float)], [np.asarray(dq, dtype=float)]
    for tau in torques:
        q_next, dq_next = predict(qs[-1], dqs[-1], tau, payload, dt)
        qs.append(q_next)
        dqs.append(dq_next)
    return np.array(qs), np.array(dqs)
