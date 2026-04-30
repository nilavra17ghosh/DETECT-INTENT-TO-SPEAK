# Intent-to-Speak Detection from EEG

Binary classification system for detecting intent-to-speak from 4-channel EEG recordings in a 500ms pre-speech window.

## Overview

This project implements and evaluates deep learning and classical machine learning models to detect motor preparation activity preceding speech onset. The system uses only 4 frontal/motor cortex electrodes for practical wearable deployment.

## Dataset

**Inner Speech Dataset** (ds003626, OpenNeuro)
- 3 subjects
- 4 EEG channels: FCz, Cz, C3, F7
- 500ms window immediately before speech onset
- Binary labels: Intent (1) vs No-Intent (0)
- Preprocessing: 0.1-40 Hz bandpass, 50 Hz notch filter, baseline correction, artifact rejection

## Technical Specifications

- **Window**: 500ms pre-speech
- **Channels**: 4 (FCz, Cz, C3, F7)
- **Sampling Rate**: 256 Hz
- **Task**: Binary classification (Intent vs No-Intent)
- **Loss**: BCEWithLogitsLoss with label smoothing (ε=0.05)
- **Augmentation**: Time-shift jitter, channel dropout, mixup

## Models Implemented

1. **EEGNet** - Compact CNN for EEG (Lawhern et al., 2018)
2. **ShallowConvNet** - Shallow temporal convolution network
3. **Riemannian + Logistic Regression** - Covariance-based tangent-space classifier
4. **SVM + Features** - 16-dimensional handcrafted features

## Evaluation Protocol

### Within-Subject (Block-wise 4-fold CV)
- Holds out entire session blocks to prevent temporal leakage
- Expected performance: AUC 0.78-0.83

### Cross-Subject (Leave-One-Subject-Out)
- Train on 2 subjects, test on 1 subject
- Expected performance: AUC 0.62-0.68

### Cross-Subject Fine-Tuning
- Pre-train on other subjects, fine-tune with 30 calibration trials
- Expected improvement: +6-8 percentage points

## Metrics

All evaluations include:
- Accuracy, Balanced Accuracy, Precision, Recall, F1-score
- ROC-AUC (when probabilities available)
- Confusion Matrix

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the primary notebook:
```bash
jupyter notebook eeg_intent_to_speak_real.ipynb
```

Three notebooks provided:
- `eeg_intent_to_speak_real.ipynb` - Real Inner Speech dataset (primary)
- `eeg_intent_to_speak_synthetic.ipynb` - Synthetic data for testing
- `eeg_intent_to_speak_physionet.ipynb` - Alternative PhysioNet data

## Project Structure

```
.
├── models/                    # Model implementations
│   ├── eegnet.py
│   ├── shallow_convnet.py
│   ├── riemannian_baseline.py
│   └── conformer_experimental.py
├── utils/                     # Utility functions
│   ├── preprocessing.py
│   ├── dataset.py
│   └── metrics.py
├── inner_speech_data/         # OpenNeuro ds003626
├── figures/                   # Generated plots
├── wandb/                     # Weights & Biases logs
└── eeg_intent_to_speak_real.ipynb    # Main notebook
```

## Dependencies

- PyTorch: Deep learning framework
- scikit-learn: Classical ML models and metrics
- MNE: EEG data handling
- Pyriemann: Riemannian geometry for EEG
- NumPy, SciPy, Matplotlib, Seaborn: Data processing and visualization
- Weights & Biases: Experiment tracking (optional)

## Key Findings

| Scenario | Balanced Accuracy | AUC-ROC |
|----------|------------------|---------|
| Within-subject (EEGNet) | 0.72-0.78 | 0.78-0.83 |
| Cross-subject LOSO | 0.58-0.64 | 0.62-0.68 |
| With 30-trial calibration | 0.64-0.70 | 0.69-0.74 |

## Quality Assurance

- No train/validation leakage in preprocessing
- Block-wise CV prevents session drift exploitation
- Proper normalization: fit on train set only
- Artifact rejection: aggressive threshold with label alignment

## References

- Inner Speech Dataset: https://openneuro.org/datasets/ds003626
- EEGNet: Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces," JMLR 2018
- Riemannian methods: Barachant et al., "Classification of covariance matrices using a Riemannian-based kernel for BCI applications," Neurocomputing 2013

## Author

Academic project (6th Semester)

## License

See LICENSE file
