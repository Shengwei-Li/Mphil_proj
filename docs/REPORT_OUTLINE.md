# Report outline (M7a)

> 中文说明：这是英文技术报告的大纲，每节列出要讲的要点、用到的数字和图。请你重点看三处：标题、引言里的四条发现、讨论里的局限。措辞上觉得说过头或你讲不清的地方，直接标出来。报告定位是项目技术报告，不投稿，不提 AI 协助。所有数字都来自 `results/` 里保存的表：报告里的结果表由 `report/tables.py` 直接生成，正文里的数字是我逐个对照结果表核过的。（2026-10-20 从 `report/` 移到 `docs/`，因为它是内部工作文档。）

**Format.** IEEE two-column, about 6 pages plus references. Labelled as a project technical report, no venue.

**Working title.** *Physics, Learning, or Both? Correcting a Sampling-Based MPC after an Unknown Payload Change*

Alternative: *When Do Learned Residual Dynamics Help Sampling-Based MPC? A Controlled Study under Payload Uncertainty*

---

## Abstract (about 150 words)

- A robot arm picks up an unknown payload; its model-based controller is now wrong.
- Two ways to fix the model from a little motion data: estimate a physical parameter, or learn the residual with a neural network; also both.
- Controlled simulation study: planar two-link arm, sampling-based MPC, three "worlds" of increasing model mismatch, exact reference model, strict development/test separation.
- Findings in one sentence each: crossover with data; hybrid best inside the training region; outside it the MPC exploits learned-model errors and the advantage largely disappears; ensembles cut the worst failures but at a compute cost beyond real time.

## I. Introduction

- Motivation: model-based control needs a good model; payloads change the dynamics; learned models are popular but their failure modes inside a planner are less understood.
- Question: when does a learned correction help a sampling-based MPC, and when does it hurt?
- Approach: a deliberately small, fully controlled setting where every source of model error is known and an exact reference exists, so differences can be attributed.
- Findings (the four bullets that carry the report):
  1. **Data decides the winner.** With 9 s of motion, estimating the payload mass (C1) beats a learned residual (C2); with 30 s and more, learning wins inside the training region, and physics plus learning (C3) is best (W2, 90 s: 0.07 cm vs 1.95 cm for C1; exact reference 0.04 cm).
  2. **Outside the training region the MPC exploits learned-model errors.** One-step prediction errors along the executed path grow 8-13x for learned models and not for C1; the learned advantage largely disappears, with occasional failures of 50-66 cm.
  3. **Every wrong model is exploited.** The MPC's expected outcome is optimistic by 50-100% of the model's prediction error, for physics and learned models alike; learned models outside the data simply give it more to exploit.
  4. **Ensembles help with the tails, not with too little data.** Averaging five networks reduces failures; physics plus ensemble (C3-E) is the most robust method outside the training region with at least 30 s of data. Penalising ensemble disagreement added nothing measurable, and the ensemble planner needs about 28 ms per 20 ms control period.
- One sentence on scope: simulation only, one arm, one payload mass; conclusions are about this controlled setting.

## II. Related work (short, about 1/3 column)

- Payload / inertial parameter estimation: least squares on the rigid-body equations (Atkeson, An and Hollerbach, 1986).
- Learning residual dynamics on top of a physics model: Neural Lander (Shi et al., 2019); learning-based MPC for a robot arm with a Gaussian-process residual (Carron et al., 2019); review (Hewing et al., 2020).
- Learned dynamics inside MPC / model-based RL: Nagabandi et al. (2018); ensembles and uncertainty propagation, PETS (Chua et al., 2018).
- Parameter uncertainty in sampling-based control: Abraham et al. (2020).
- Every reference to be checked online before the draft is final.

## III. Setup

- **Arm and worlds** (Fig. 1). Two-link planar arm in MuJoCo, 0.3 m links, gravity, torque limits 20 / 10 N·m. Payload 1 kg at the tip. Three worlds: W1 mass only; W2 payload centre of mass offset (4 cm along, 6 cm across the link); W3 W2 plus joint friction (viscous and Coulomb) absent from the controller's model.
- **Controller.** Sampling-based MPC with path-integral (MPPI-style) weighting: 256 candidates, 0.2 s horizon, 0.02 s period, the same settings for all methods. Only the predictor changes.
- **Methods** (Table I): C0 no-payload physics; C1 physics with least-squares mass estimate; C2 no-payload physics + residual MLP (64x64, tanh) on one control period; C3 C1 physics + residual MLP; C2-E / C3-E five-network ensembles (average; optional disagreement penalty); reference = physics with the true parameters.
- **Data.** C0 MPC with exploration noise between random targets in the "training region" (x >= 0.05 m); 3 s episodes; budgets 9, 30, 90 s; split by whole episodes.

## IV. Protocol (the part that shows rigour)

- Hand-written model verified against the simulator term by term; the reference predicts the simulator exactly (Sec. III), so all remaining error is attributable.
- Development material (data seed 4, validation seed 500, tasks A, B, seeds 5500 / 6500) used for every choice: network size, training settings, penalty weight. Choices frozen before test runs.
- Test material: data seeds 1-3; 8 tasks inside (seed 5000) and 8 outside (seed 6000) the training region; 24 runs per cell.
- Metrics: final tip error (mean of the last 0.5 s); "not settled" (error above 1 cm at the end); 90th percentile; planning time per round measured separately with nothing else running.
- Reproducibility check: re-running the inside tasks in a later experiment reproduced all 792 earlier runs exactly.

## V. Results

- **A. Prediction** (Fig. 2). Data needed by C2; C3 most accurate; C1 flat because its error is structural; learned models degrade outside.
- **B. Control inside the training region** (Fig. 3, solid lines). Crossover; C3 near the reference at 90 s; learned models fail occasionally with little data (C2, W2, 9 s: worst 68 cm).
- **C. Outside the training region** (Fig. 3 dashed, Fig. 5). Advantage disappears for C2, shrinks for C3; trajectories of the worst runs.
- **D. Why: optimistic expectations** (Fig. 4). Optimism vs prediction error for every wrong model; learned models outside sit at the high-error end.
- **E. Ensembles** (Fig. 6, Table II). Fewer unsettled runs; C3-E most robust outside; penalty adds nothing; cost.

## VI. Discussion and limitations

- Practical reading: with little data prefer physics; with more data use physics plus learning and an ensemble; watch the tails, not only the median.
- Why the disagreement penalty did not help (one-step spread is small; errors compound over the horizon; propagating members separately, as in PETS, is the natural next step).
- Limitations: simulation only; planar arm; one payload mass; "outside" is one region of the workspace; ensembles differ only in initialisation; ensemble planner not real-time on the laptop used.

## VII. Conclusion (3-4 sentences)

---

## Figures and tables

| Item | File | Width |
|---|---|---|
| Fig. 1 arm, regions, test targets | `report/figures/setup.pdf` | one column |
| Table I methods | written in LaTeX | one column |
| Fig. 2 prediction error vs data | `figures/prediction.pdf` | full width |
| Fig. 3 control error vs data, inside and outside | `figures/control.pdf` | full width |
| Fig. 4 optimism vs prediction error | `figures/optimism.pdf` | one column |
| Fig. 5 worst runs outside | `figures/worst.pdf` | full width |
| Fig. 6 unsettled runs, single vs ensemble | `figures/ensemble.pdf` | one column |
| Table II key numbers | generated from `results/` by script | one column |
