EEG Intent-to-Speak Architecture (Report-Ready)

Assignment constraints: 2–4 electrodes, −500 ms to 0 ms window, binary classification.

PhysioNet proxy note (code behavior): cue-locked epochs with tmin=0.0, tmax=0.5
and bandpass 8–30 Hz. Use tmin=-0.5, tmax=0.0 if you want strict assignment
alignment for the proxy dataset.

---

## Full Pipeline

```
Raw EEG (BioSemi 128ch / OpenBCI 4ch)
          |
          v
  +-------------------------------------------+
  |              PREPROCESSING                |
  |  Bandpass 0.1–40 Hz (preserves BP < 1 Hz) |
  |  Notch 50 Hz  (power-line removal)         |
  |  Downsample  1024 -> 256 Hz                |
  |  Channel Selection: FCz, Cz, C3, F7        |
  |  Epoch: -500 ms -> 0 ms (128 samples)      |
  |  Baseline: first 100 ms of epoch           |
  |  Artifact Reject: +/-100 uV threshold      |
  |  Z-score: fit on train, apply to val/test  |
  +-------------------------------------------+
          |
          v  (n_trials x 4 channels x 128 samples)
  +-----------------------------------------------+
  |           AUGMENTATION (train only)            |
  |  +/-50 ms time-shift jitter  | Mixup (alpha=0.2)|
  |  Channel dropout 10%         | Online in Dataset|
  +-----------------------------------------------+
          |
          v
  +--------------------------------------------------------------+
  |                        MODEL SELECTION                        |
  |                                                              |
  |   +------------------+   +--------------------------+        |
  |   |  EEGNet [PRIMARY] |   | EEGNet + Attention [OPT] |        |
  |   +------------------+   +--------------------------+        |
  |                                                              |
  |   +------------------+   +------------------+                |
  |   | Riemannian + LR  |   | ShallowConvNet   |                |
  |   | [BASELINE]       |   | [BASELINE]       |                |
  |   +------------------+   +------------------+                |
  +--------------------------------------------------------------+
          |
          v
     Sigmoid -> Binary Decision (Intent=1 / No-Intent=0)

Note: EEG-Conformer is experimental and not used as a primary model due to
overfitting risk in low-channel, low-data settings.
```

---

## Primary Model: EEGNet

Input: (B, 1, 4, 128)

Block 1: Temporal Conv -> BatchNorm
Block 2: Depthwise Spatial Conv -> AvgPool -> Dropout
Block 3: Separable Conv -> AvgPool -> Dropout
Flatten -> Linear(1 logit) -> BCEWithLogitsLoss

---

## Optional Enhancement: EEGNet + Lightweight Temporal Attention

Same EEGNet backbone, then temporal self-attention over the separable-conv
features to emphasize the late BP component (around -100 ms).

---

## Baselines

Riemannian + Logistic Regression:
- Covariance (OAS) -> Tangent Space -> Logistic Regression

ShallowConvNet:
- Learns log band-power features via square + log transform

---

## Training Configuration (Aligned with Code)

- Loss: BCEWithLogitsLoss with label smoothing (eps = 0.05)
- Optimizer: AdamW, cosine LR schedule, early stopping
- Within-subject: block-wise k-fold CV
- Cross-subject: LOSO

---

## Expected Performance (Realistic Targets)

Within-subject (EEGNet, 4ch): 0.50–0.60 balanced accuracy
Cross-subject (LOSO): 0.48–0.56 balanced accuracy
With 30-trial fine-tune: +2–4 pp improvement

---

## Extra Analyses

- Channel ablation study: all 2–4 channel combinations
- Detection latency: evaluate windows ending at -100 to -500 ms
- Cross-subject fine-tuning: 30-trial calibration

---

## Realism vs Accuracy

High accuracy (>90%) on real data usually indicates leakage or contamination.
The pipeline prioritizes honest evaluation under realistic constraints.
