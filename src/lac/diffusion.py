"""A standard neural Diffusion Policy as the second imitation policy (milestone M9b).

CFDP (lac.policy) computes the denoising direction in closed form from the
demonstrations. The usual way is to *train a network* to do it: Diffusion
Policy (Chi et al., RSS 2023). This module implements its CNN version, with the
paper's main settings, so that M8's and M9's findings can be checked on a
policy that is not built out of the demonstrations themselves:

* network: a 1D convolutional U-Net over the action chunk's time axis; the
  observation and the noise level enter every residual block through FiLM
  (a per-channel scale and shift computed from them);
* training: add noise of a random level k (out of 100, squared-cosine schedule)
  to a demonstrated chunk, and train the network to predict that noise
  (mean squared error); AdamW, learning rate 1e-4 with warm-up and cosine
  decay, and an exponential moving average (EMA) of the weights, which is the
  network used for acting;
* acting: start from pure noise and remove it in 10 DDIM steps (deterministic
  given the starting noise), clipping the predicted clean chunk to [-1, 1].

Observations and actions are exactly CFDP's (lac.policy.training_pairs: one
frame of sin q, cos q, speeds and target minus tip; the next 8 joint positions
relative to the current ones), both scaled to [-1, 1] per dimension as in the
paper. CFDP's target weighting is not used: it only reshapes CFDP's distance
metric, and a network learns its own.

Training runs on the CPU (a few minutes per world, see TECHSTACK); the trained
network is cached in cache/dp/.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lac import data
from lac.demos import Demos
from lac.policy import training_pairs

CACHE = data.CACHE / "dp"


# --- the network -------------------------------------------------------------


class SinusoidalEmbedding(nn.Module):
    """The noise level k as a vector of sines and cosines of k at different frequencies."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / (half - 1))
        angle = k.float()[:, None] * freq[None, :]
        return torch.cat([angle.sin(), angle.cos()], dim=-1)


class ConvBlock(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, kernel: int, groups: int = 8):
        super().__init__(nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2), nn.GroupNorm(groups, c_out), nn.Mish())


class ResidualBlock(nn.Module):
    """Two convolutions; between them the condition scales and shifts every channel (FiLM)."""

    def __init__(self, c_in: int, c_out: int, cond_dim: int, kernel: int):
        super().__init__()
        self.first, self.second = ConvBlock(c_in, c_out, kernel), ConvBlock(c_out, c_out, kernel)
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * c_out))
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).unsqueeze(-1).chunk(2, dim=1)
        return self.second(scale * self.first(x) + shift) + self.skip(x)


class ConditionalUNet1D(nn.Module):
    """Noise prediction for a chunk of shape (batch, horizon, action_dim), given observation and noise level."""

    def __init__(self, action_dim: int, obs_dim: int, dims=(64, 128, 256), step_dim: int = 128, kernel: int = 5):
        super().__init__()
        self.step = nn.Sequential(SinusoidalEmbedding(step_dim), nn.Linear(step_dim, 4 * step_dim), nn.Mish(),
                                  nn.Linear(4 * step_dim, step_dim))
        cond = step_dim + obs_dim
        channels = (action_dim,) + tuple(dims)
        self.down = nn.ModuleList()
        for i, (a, b) in enumerate(zip(channels[:-1], channels[1:])):
            last = i == len(dims) - 1
            self.down.append(nn.ModuleList([ResidualBlock(a, b, cond, kernel), ResidualBlock(b, b, cond, kernel),
                                            nn.Identity() if last else nn.Conv1d(b, b, 3, stride=2, padding=1)]))
        self.middle = nn.ModuleList([ResidualBlock(dims[-1], dims[-1], cond, kernel) for _ in range(2)])
        self.up = nn.ModuleList()
        for i, (a, b) in enumerate(reversed(list(zip(dims[:-1], dims[1:])))):
            self.up.append(nn.ModuleList([ResidualBlock(2 * b, a, cond, kernel), ResidualBlock(a, a, cond, kernel),
                                          nn.ConvTranspose1d(a, a, 4, stride=2, padding=1)]))
        self.out = nn.Sequential(ConvBlock(dims[0], dims[0], kernel), nn.Conv1d(dims[0], action_dim, 1))

    def forward(self, x: torch.Tensor, k: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([self.step(k), obs], dim=-1)
        h = x.transpose(1, 2)  # convolve over time: (batch, channels, horizon)
        skips = []
        for first, second, down in self.down:
            h = second(first(h, cond), cond)
            skips.append(h)
            h = down(h)
        for block in self.middle:
            h = block(h, cond)
        for first, second, up in self.up:
            h = second(first(torch.cat([h, skips.pop()], dim=1), cond), cond)
            h = up(h)
        return self.out(h).transpose(1, 2)


# --- the noise schedule --------------------------------------------------------


def cosine_alpha_bar(steps: int, max_beta: float = 0.999) -> np.ndarray:
    """Fraction of the signal left after k = 1..steps noise steps (the "squaredcos_cap_v2" schedule)."""
    f = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa: E731
    betas = np.array([min(1 - f((i + 1) / steps) / f(i / steps), max_beta) for i in range(steps)])
    return np.cumprod(1.0 - betas)


@dataclass
class Scaler:
    """Per-dimension affine map of the data range onto [-1, 1]."""

    low: np.ndarray
    high: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        low, high = x.min(axis=0), x.max(axis=0)
        return cls(low, np.where(high - low < 1e-8, low + 1.0, high))

    def __call__(self, x):
        return 2.0 * (np.asarray(x, dtype=float) - self.low) / (self.high - self.low) - 1.0

    def inverse(self, z):
        return (np.asarray(z, dtype=float) + 1.0) / 2.0 * (self.high - self.low) + self.low


# --- the policy ----------------------------------------------------------------


@dataclass
class DiffusionPolicy:
    """Same interface as lac.policy.CFDP: ``fit(demos)``, ``predict(window, rng)``."""

    n_obs: int = 1
    n_act: int = 8
    train_steps: int = 100  # noise levels during training
    inference_steps: int = 10  # DDIM steps when acting
    epochs: int = 100
    batch: int = 256
    learning_rate: float = 1e-4
    warmup: int = 500  # optimiser steps
    ema_power: float = 0.75  # EMA decay grows towards 0.9999 as 1 - (1 + step)^-power, as in the paper's code
    seed: int = 0
    threads: int = 5  # three worlds train side by side on a 16-core machine
    net: ConditionalUNet1D = field(init=False, repr=False)
    losses: list[float] = field(init=False, repr=False, default_factory=list)

    def cache_path(self, demos: Demos) -> Path:
        n = len(np.unique(demos.episode))
        # The demonstrations are identified by size and a checksum of their angles (hashlib, not
        # Python's hash(), which differs between processes).
        tag = f"{n}ep_{len(demos.q)}p_{hashlib.sha1(np.round(demos.q, 6).tobytes()).hexdigest()[:8]}"
        return CACHE / f"dp_{tag}_e{self.epochs}_s{self.seed}.pt"

    def fit(self, demos: Demos, cache: bool = True) -> "DiffusionPolicy":
        obs, act = training_pairs(demos, self.n_obs, self.n_act)
        self.obs_scale, self.act_scale = Scaler.fit(obs), Scaler.fit(act)
        torch.manual_seed(self.seed)
        self.net = ConditionalUNet1D(action_dim=2, obs_dim=obs.shape[1])
        self.alpha_bar = cosine_alpha_bar(self.train_steps)
        path = self.cache_path(demos)
        if cache and path.exists():
            saved = torch.load(path, weights_only=False)
            self.net.load_state_dict(saved["net"])
            self.losses = saved["losses"]
        else:
            self._train(self.obs_scale(obs), self.act_scale(act).reshape(len(act), self.n_act, 2))
            if cache:
                CACHE.mkdir(parents=True, exist_ok=True)
                torch.save({"net": self.net.state_dict(), "losses": self.losses}, path)
        self.net.eval()
        return self

    def _train(self, obs: np.ndarray, act: np.ndarray) -> None:
        torch.set_num_threads(self.threads)
        rng = np.random.default_rng(self.seed)
        o, a = torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(act, dtype=torch.float32)
        alpha_bar = torch.as_tensor(self.alpha_bar, dtype=torch.float32)
        ema = ConditionalUNet1D(action_dim=2, obs_dim=obs.shape[1])
        ema.load_state_dict(self.net.state_dict())
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate, betas=(0.95, 0.999),
                                      weight_decay=1e-6)
        per_epoch = math.ceil(len(o) / self.batch)
        total = per_epoch * self.epochs
        schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: min(1.0, (s + 1) / self.warmup) * 0.5 *
                                                     (1 + math.cos(math.pi * min(s, total) / total)))
        step = 0
        self.losses = []
        for _ in range(self.epochs):
            order = torch.as_tensor(rng.permutation(len(o)))
            running = 0.0
            for i in range(per_epoch):
                idx = order[i * self.batch:(i + 1) * self.batch]
                x0, cond = a[idx], o[idx]
                k = torch.randint(0, self.train_steps, (len(idx),))
                noise = torch.randn_like(x0)
                ab = alpha_bar[k][:, None, None]
                noisy = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
                loss = torch.mean((self.net(noisy, k, cond) - noise) ** 2)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                schedule.step()
                step += 1
                decay = min(0.9999, max(0.0, 1 - (1 + step) ** -self.ema_power))
                with torch.no_grad():
                    for pe, p in zip(ema.parameters(), self.net.parameters()):
                        pe.mul_(decay).add_(p, alpha=1 - decay)
                    for be, b in zip(ema.buffers(), self.net.buffers()):
                        be.copy_(b)
                running += float(loss.detach()) * len(idx)
            self.losses.append(running / len(o))
        self.net = ema

    @torch.no_grad()
    def predict(self, window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One chunk (n_act, 2) of joint positions relative to the current ones."""
        obs = torch.as_tensor(self.obs_scale(window), dtype=torch.float32)[None]
        x = torch.as_tensor(rng.normal(size=(1, self.n_act, 2)), dtype=torch.float32)
        ks = np.linspace(self.train_steps - 1, 0, self.inference_steps).round().astype(int)
        for i, k in enumerate(ks):
            ab = float(self.alpha_bar[k])
            ab_next = float(self.alpha_bar[ks[i + 1]]) if i + 1 < len(ks) else 1.0
            eps = self.net(x, torch.tensor([k]), obs)
            x0 = ((x - math.sqrt(1 - ab) * eps) / math.sqrt(ab)).clamp(-1.0, 1.0)
            eps = (x - math.sqrt(ab) * x0) / math.sqrt(1 - ab)  # consistent with the clipped x0
            x = math.sqrt(ab_next) * x0 + math.sqrt(1 - ab_next) * eps  # DDIM, deterministic (eta = 0)
        return self.act_scale.inverse(x[0].numpy().reshape(-1)).reshape(self.n_act, 2)
