"""C2 and C3: a neural network that learns what a physics model gets wrong (milestone M5).

The physics model (C0's empty-arm model for C2, C1's estimated-mass model for
C3) predicts the state one control period (0.02 s) ahead. The network is
trained on recorded motion to predict the difference between that prediction
and what the real arm actually did:

    corrected prediction = physics prediction + network(state, torque)

Learning only this difference (the residual) rather than the whole motion
means the network does not have to rediscover gravity or inertia, so it
needs far less data.

Training uses PyTorch. Afterwards the weights are copied into a plain NumPy
network, because the MPC calls it ten times per round on 256 candidates, and
for such small calls PyTorch's fixed overhead costs more than the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lac.data import PERIOD, Transitions
from lac.mpc import tip_of


def features(q: np.ndarray, dq: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Network input: sin and cos of both angles, both speeds, both torques (8 numbers).

    Sine and cosine are used instead of the angle itself because 179 and -179
    degrees are neighbours, which a raw angle would hide from the network.
    """
    return np.concatenate([np.sin(q), np.cos(q), dq, tau], axis=-1)


@dataclass
class ResidualNet:
    """A trained network in plain NumPy: standardise input, tanh hidden layers, un-standardise output."""

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    input_mean: np.ndarray
    input_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray
    history: dict = field(default_factory=dict)  # training and validation loss per epoch, for inspection

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = (x - self.input_mean) / self.input_std
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.tanh(h @ w + b)
        return (h @ self.weights[-1] + self.biases[-1]) * self.output_std + self.output_mean

    def save(self, path: Path) -> None:
        arrays = {f"w{i}": w for i, w in enumerate(self.weights)} | {f"b{i}": b for i, b in enumerate(self.biases)}
        np.savez(path, input_mean=self.input_mean, input_std=self.input_std, output_mean=self.output_mean,
                 output_std=self.output_std, train_loss=self.history.get("train", []),
                 validation_loss=self.history.get("validation", []), **arrays)

    @classmethod
    def load(cls, path: Path) -> "ResidualNet":
        with np.load(path) as f:
            layers = sum(name.startswith("w") for name in f.files)
            return cls(weights=[f[f"w{i}"] for i in range(layers)], biases=[f[f"b{i}"] for i in range(layers)],
                       input_mean=f["input_mean"], input_std=f["input_std"], output_mean=f["output_mean"],
                       output_std=f["output_std"],
                       history={"train": list(f["train_loss"]), "validation": list(f["validation_loss"])})


class ResidualPredictor:
    """A physics predictor corrected by a residual network. Same interface as PhysicsPredictor."""

    def __init__(self, base, net: ResidualNet):
        self.base = base
        self.net = net

    def step(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray, duration: float):
        if not np.isclose(duration, PERIOD):
            raise ValueError(f"the network was trained for steps of {PERIOD} s, not {duration} s")
        tau = np.broadcast_to(tau, np.shape(q))
        q_next, dq_next = self.base.step(q, dq, tau, duration)
        correction = self.net(features(q, dq, tau))
        return q_next + correction[..., :2], dq_next + correction[..., 2:]

    def hold_torque(self, q: np.ndarray) -> np.ndarray:
        # Only used to start the MPC's first plan; the physics part is good enough for that.
        return self.base.hold_torque(q)


def residual_targets(base, data: Transitions) -> np.ndarray:
    """What the base predictor got wrong on each transition: (angle error, speed error), 4 numbers."""
    q_pred, dq_pred = base.step(data.q, data.dq, data.tau, PERIOD)
    return np.concatenate([data.q_next - q_pred, data.dq_next - dq_pred], axis=-1)


def train_residual(
    base,
    train: Transitions,
    validation: Transitions,
    hidden: tuple[int, ...] = (64, 64),
    max_epochs: int = 3000,
    patience: int = 300,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 0,
    threads: int = 1,
) -> ResidualNet:
    """Fit a residual network with Adam on the full batch, stopping when validation stops improving.

    `validation` must be whole episodes not in `train`. The weights from the
    epoch with the lowest validation loss are kept.
    """
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(threads)  # small networks gain nothing from more threads; run trainings in parallel instead
    x_train, y_train = features(train.q, train.dq, train.tau), residual_targets(base, train)
    x_val, y_val = features(validation.q, validation.dq, validation.tau), residual_targets(base, validation)
    # Standardise with training statistics only; a tiny floor avoids dividing by zero
    # when an output never varies (e.g. W1 with the exact model).
    in_mean, in_std = x_train.mean(0), x_train.std(0) + 1e-8
    out_mean, out_std = y_train.mean(0), y_train.std(0) + 1e-12

    def tensor(a):
        return torch.as_tensor(a, dtype=torch.float32)

    xt, yt = tensor((x_train - in_mean) / in_std), tensor((y_train - out_mean) / out_std)
    xv, yv = tensor((x_val - in_mean) / in_std), tensor((y_val - out_mean) / out_std)

    sizes = [x_train.shape[1], *hidden, y_train.shape[1]]
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(torch.nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(torch.nn.Tanh())
    net = torch.nn.Sequential(*layers)
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best, best_state, since_best = np.inf, None, 0
    history = {"train": [], "validation": []}
    for _ in range(max_epochs):
        optimiser.zero_grad()
        loss = torch.mean((net(xt) - yt) ** 2)
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            val_loss = float(torch.mean((net(xv) - yv) ** 2))
        history["train"].append(float(loss.detach()))
        history["validation"].append(val_loss)
        if val_loss < best:
            best, best_state, since_best = val_loss, [p.detach().clone() for p in net.parameters()], 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    with torch.no_grad():
        for p, saved in zip(net.parameters(), best_state):
            p.copy_(saved)

    linear = [m for m in net if isinstance(m, torch.nn.Linear)]
    return ResidualNet(
        weights=[m.weight.detach().numpy().T.astype(float) for m in linear],
        biases=[m.bias.detach().numpy().astype(float) for m in linear],
        input_mean=in_mean, input_std=in_std, output_mean=out_mean, output_std=out_std,
        history=history,
    )


class EnsemblePredictor:
    """Several residual networks on the same physics model (milestone M6).

    Prediction uses the average of the members' corrections. Where the members
    were trained on plenty of similar data they agree; where they disagree, the
    data probably never covered that situation. After every step, `last_spread`
    holds that disagreement for each predicted state: the standard deviation,
    across members, of the predicted tip position (m). The MPC can be told to
    avoid actions that lead to a large spread.

    The members' weights are stacked so all of them run in one batched
    calculation rather than one after another.
    """

    def __init__(self, base, nets: list[ResidualNet]):
        self.base = base
        self.nets = nets
        stack = lambda arrays: np.stack(arrays)  # noqa: E731
        self.weights = [stack([n.weights[i] for n in nets]) for i in range(len(nets[0].weights))]
        self.biases = [stack([n.biases[i] for n in nets])[:, None, :] for i in range(len(nets[0].biases))]
        self.input_mean, self.input_std = stack([n.input_mean for n in nets])[:, None], stack([n.input_std for n in nets])[:, None]
        self.output_mean = stack([n.output_mean for n in nets])[:, None]
        self.output_std = stack([n.output_std for n in nets])[:, None]
        self.last_spread = None

    def corrections(self, x: np.ndarray) -> np.ndarray:
        """Each member's correction for inputs of shape (n, 8): returns (members, n, 4)."""
        h = (x[None] - self.input_mean) / self.input_std
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.tanh(h @ w + b)
        return (h @ self.weights[-1] + self.biases[-1]) * self.output_std + self.output_mean

    def step(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray, duration: float):
        if not np.isclose(duration, PERIOD):
            raise ValueError(f"the networks were trained for steps of {PERIOD} s, not {duration} s")
        q, dq = np.atleast_2d(q), np.atleast_2d(dq)
        tau = np.broadcast_to(tau, q.shape)
        q_next, dq_next = self.base.step(q, dq, tau, duration)
        each = self.corrections(features(q, dq, tau))
        tips = tip_of(q_next[None] + each[..., :2])
        self.last_spread = np.sqrt(np.sum(tips.var(axis=0), axis=-1))
        mean = each.mean(axis=0)
        return q_next + mean[..., :2], dq_next + mean[..., 2:]

    def hold_torque(self, q: np.ndarray) -> np.ndarray:
        return self.base.hold_torque(q)
