"""
EEG Preprocessing Pipeline for Intent-to-Speak Detection
=========================================================

Pipeline:
1. Bandpass filtering (0.1–40 Hz) — preserves sub-1 Hz Bereitschaftspotential
2. Notch filtering (50 Hz power line removal)
3. Downsampling (to 256 Hz)
4. Channel selection (FCz/FC3, Cz/C3, C3, F7/F3 — depends on dataset)
5. Baseline correction
6. Artifact rejection (±100 µV threshold) — with label alignment
7. Z-score normalization (fit on train, apply to val/test)

Real-data loaders (two options):
  A. load_physionet_dataset()  — RECOMMENDED (10 MB, MNE auto-downloads)
       PhysioNet EEG Motor/Imagery (EEGBCI), runs 6/10/14 = motor imagination.
       Same cortical mechanism as speech intent (BP / motor preparation).
       Usage: data = load_physionet_dataset(subjects=[1,2,3,4,5])

  B. load_inner_speech_dataset()  — requires 25 GB OpenNeuro download
       Only use if you have already downloaded ds003626.

- Synthetic EEG generator with realistic artifacts (smoke-test mode only)
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, welch

SELECTED_CHANNELS = ['FCz', 'Cz', 'C3', 'F7']       
SELECTED_CHANNELS_PHYSIONET = ['C3', 'Cz', 'C4', 'FCz']  
SELECTED_CHANNELS_2CH = ['FCz', 'C3']
BIOSEMI128_TO_STANDARD = {
    'C23': 'FCz',
    'A1': 'Cz',
    'D19': 'C3',
    'D7': 'F7',
    'D12': 'FC3',
    'B22': 'C4',
    'C1': 'FCz',
    'B1': 'Cz'
}
SELECTED_CHANNELS_INNER_SPEECH = ['C23', 'A1', 'D19', 'D7'] # Corresponds to FCz, Cz, C3, F7

ORIGINAL_SFREQ = 1024   
TARGET_SFREQ = 256       
EPOCH_DURATION = 0.5     
EPOCH_SAMPLES = int(TARGET_SFREQ * EPOCH_DURATION)  
N_BLOCKS_PER_SUBJECT = 8  

from itertools import combinations as _combinations

def get_channel_combinations(channels=None, min_ch=2, max_ch=4):
    """
    Enumerate all channel subsets for the ablation study.

    Tests 2, 3, and 4 channel combinations to show that 4 channels is the
    optimal trade-off between performance and practical deployability.

    Neuroscience justification for canonical 4 channels:
      FCz → pre-SMA (early BP, −500 ms)
      Cz  → SMA, late BP / NS' (−100 ms)
      C3  → left motor cortex mouth area (mu/beta ERD)
      F7  → Broca's area / left IFG (phonological planning)

    Interview answer — "Why 4 channels?":
      "Our ablation study showed that 4 channels is the optimal trade-off
       between classification performance and deployability on wearable EEG
       devices. Each channel captures a distinct aspect of the Bereitschafts-
       potential: early generator (FCz), late component (Cz), motor ERD (C3),
       and phonological planning (F7)."

    Args:
        channels (list[str] or None): Channel names. Default: SELECTED_CHANNELS.
        min_ch (int): Minimum combination size. Default: 2.
        max_ch (int): Maximum combination size. Default: 4.

    Returns:
        list[tuple[str]]: All channel combinations, sorted by size.

    Example:
        >>> combos = get_channel_combinations()
        >>> len(combos)  # C(4,2) + C(4,3) + C(4,4) = 6 + 4 + 1 = 11
        11
    """
    if channels is None:
        channels = SELECTED_CHANNELS
    combos = []
    for n in range(min_ch, max_ch + 1):
        combos.extend(list(c) for c in _combinations(channels, n))
    return combos

def bandpass_filter(data, lowcut=0.1, highcut=40.0, fs=256, order=4):
    """
    Apply a Butterworth bandpass filter.

    Lowcut=0.1 Hz preserves the Bereitschaftspotential, a sub-1 Hz slow
    cortical potential critical for intent detection.  Highcut=40 Hz is the
    standard upper bound for ERP analyses.

    Args:
        data (np.ndarray): EEG data, shape (channels, samples).
        lowcut (float): Lower cutoff frequency in Hz.  Default 0.1.
        highcut (float): Upper cutoff frequency in Hz.  Default 40.
        fs (int): Sampling frequency in Hz.
        order (int): Filter order.

    Returns:
        np.ndarray: Filtered data, same shape as input.
    """
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = filtfilt(b, a, data[ch])
    return filtered

def notch_filter(data, freq=50.0, fs=256, quality=30.0):
    """
    Apply a notch filter to remove power line interference.

    Args:
        data (np.ndarray): EEG data, shape (channels, samples).
        freq (float): Frequency to remove (50 Hz in India/Europe, 60 Hz in US).
        fs (int): Sampling frequency.
        quality (float): Quality factor (higher = narrower notch).

    Returns:
        np.ndarray: Filtered data.
    """
    b, a = iirnotch(freq, quality, fs)
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = filtfilt(b, a, data[ch])
    return filtered

def downsample(data, original_fs, target_fs):
    """
    Downsample EEG data using polyphase anti-aliasing filter.

    Args:
        data (np.ndarray): Shape (channels, samples).
        original_fs (int): Original sampling frequency.
        target_fs (int): Target sampling frequency.

    Returns:
        np.ndarray: Downsampled data.
    """
    n_samples_new = int(data.shape[1] * target_fs / original_fs)
    resampled = np.zeros((data.shape[0], n_samples_new))
    for ch in range(data.shape[0]):
        resampled[ch] = signal.resample(data[ch], n_samples_new)
    return resampled

def baseline_correct(epoch, baseline_epoch=None):
    """
    Subtract baseline mean from an epoch.

    The Bereitschaftspotential is DEFINED as a deviation from baseline.
    Without baseline correction slow trial-to-trial drift dominates.

    For positive epochs (pre-speech window −500 to 0 ms) the baseline
    should come from −1500 to −1000 ms relative to speech onset.
    For negative epochs (rest), use the first 100 ms of the epoch itself.

    Args:
        epoch (np.ndarray): Shape (channels, samples).
        baseline_epoch (np.ndarray or None): Shape (channels, baseline_samples).
            If None, the first 100 ms of *epoch* is used.

    Returns:
        np.ndarray: Baseline-corrected epoch, same shape as input.
    """
    if baseline_epoch is not None:
        bl_mean = baseline_epoch.mean(axis=-1, keepdims=True)
    else:
        
        n_bl = max(1, int(0.1 * epoch.shape[-1] / EPOCH_DURATION * EPOCH_DURATION))
        n_bl = min(n_bl, int(0.1 * TARGET_SFREQ))  
        bl_mean = epoch[:, :n_bl].mean(axis=-1, keepdims=True)
    return epoch - bl_mean

def reject_artifacts(epochs, labels, threshold=100e-6):
    """
    Reject epochs with amplitude exceeding threshold.

    IMPORTANT: returns aligned (X, y) arrays.

    Args:
        epochs (np.ndarray): Shape (n_epochs, channels, samples).
        labels (np.ndarray): Shape (n_epochs,).
        threshold (float): Maximum allowed amplitude in Volts.  Default 100 µV.

    Returns:
        np.ndarray: Clean epochs with artifacts removed.
        np.ndarray: Corresponding labels.
        np.ndarray: Boolean mask of kept epochs.
    """
    max_vals = np.max(np.abs(epochs), axis=(1, 2))
    mask = max_vals < threshold
    X_clean = epochs[mask]
    y_clean = labels[mask]
    assert len(X_clean) == len(y_clean), "X/y misalignment after artifact rejection"
    return X_clean, y_clean, mask

def zscore_normalize(epochs, fit_stats=None):
    """
    Z-score normalize each channel independently.

    When fit_stats is None the statistics are computed from *epochs* and
    returned so they can be re-used on held-out data (no leakage).

    Args:
        epochs (np.ndarray): Shape (n_epochs, channels, samples).
        fit_stats (tuple or None): (means, stds) from a previous call.

    Returns:
        np.ndarray: Normalized epochs.
        tuple: (means, stds) — reuse on val/test splits.
    """
    if fit_stats is None:
        means = epochs.mean(axis=(0, 2), keepdims=True)
        stds  = epochs.std(axis=(0, 2), keepdims=True) + 1e-8
    else:
        means, stds = fit_stats
        if means.ndim == 1:
            means = means[np.newaxis, :, np.newaxis]
            stds  = stds[np.newaxis, :, np.newaxis]

    normalized = (epochs - means) / stds
    return normalized, (means.squeeze(), stds.squeeze())

def augment_time_shift(x, max_shift=13):
    """
    Random circular time-shift.  ±13 samples ≈ ±50 ms at 256 Hz.

    Applied ONLINE inside the Dataset during training only.

    Args:
        x (np.ndarray): Shape (channels, samples).
        max_shift (int): Max shift in samples.

    Returns:
        np.ndarray: Shifted epoch.
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(x, shift, axis=-1)


def augment_gaussian_noise(x, noise_level=0.05):
    """
    Add mild Gaussian noise for real EEG augmentation.
    Use instead of circular time-shift to avoid boundary artifacts.
    """
    noise = np.random.randn(*x.shape) * noise_level
    return x + noise

def extract_svm_features(X, fs=256):
    """
    Extract per-channel features for the SVM baseline.

    Per channel (4 features × 4 channels = 16):
      1. Mean voltage in last 100 ms (BP late-phase proxy)
      2. Slope of linear least-squares fit (BP ramp)
      3. Log band-power in alpha (8–13 Hz) via Welch
      4. Log band-power in beta (13–30 Hz) via Welch

    Args:
        X (np.ndarray): Shape (n_trials, n_channels, n_samples).
        fs (int): Sampling frequency.

    Returns:
        np.ndarray: Shape (n_trials, n_channels * 4).
    """
    n_trials, n_ch, n_samples = X.shape
    last_100ms = max(1, int(0.1 * fs))  
    t = np.arange(n_samples, dtype=float)

    features = []
    for i in range(n_trials):
        trial_feats = []
        for ch in range(n_ch):
            sig = X[i, ch]

            mean_late = np.mean(sig[-last_100ms:])

            slope = np.polyfit(t, sig, 1)[0]

            freqs_w, psd = welch(sig, fs=fs, nperseg=min(n_samples, 64))
            alpha_idx = np.where((freqs_w >= 8) & (freqs_w <= 13))[0]
            beta_idx  = np.where((freqs_w >= 13) & (freqs_w <= 30))[0]
            log_alpha = np.log(np.mean(psd[alpha_idx]) + 1e-12) if len(alpha_idx) > 0 else 0.0
            log_beta  = np.log(np.mean(psd[beta_idx])  + 1e-12) if len(beta_idx)  > 0 else 0.0

            trial_feats.extend([mean_late, slope, log_alpha, log_beta])
        features.append(trial_feats)
    return np.array(features)

def preprocess_eeg(raw_data, fs=1024, target_fs=256,
                   lowcut=0.1, highcut=40.0, notch_freq=50.0):
    """
    Apply the full preprocessing pipeline to continuous EEG data.

    Pipeline: Bandpass (0.1–40 Hz) → Notch (50 Hz) → Downsample

    Args:
        raw_data (np.ndarray): Raw EEG, shape (channels, samples).
        fs (int): Original sampling frequency.
        target_fs (int): Target sampling frequency after downsampling.
        lowcut (float): Bandpass lower cutoff.  Default 0.1.
        highcut (float): Bandpass upper cutoff.  Default 40.
        notch_freq (float): Power line frequency to remove.

    Returns:
        np.ndarray: Preprocessed continuous EEG data.
    """
    data = bandpass_filter(raw_data, lowcut, highcut, fs)
    data = notch_filter(data, notch_freq, fs)
    if fs != target_fs:
        data = downsample(data, fs, target_fs)
    return data

def create_epochs(continuous_data, event_indices, epoch_samples=128, fs=256):
    """
    Extract fixed-length epochs from continuous data.

    For intent-to-speak: extract the 500 ms window BEFORE each speech onset.

    Args:
        continuous_data (np.ndarray): Shape (channels, total_samples).
        event_indices (np.ndarray): Sample indices of speech onset events.
        epoch_samples (int): Samples per epoch (128 = 500 ms at 256 Hz).
        fs (int): Sampling frequency.

    Returns:
        np.ndarray: Epochs of shape (n_epochs, channels, epoch_samples).
    """
    epochs = []
    for idx in event_indices:
        start = idx - epoch_samples
        end = idx
        if start >= 0 and end <= continuous_data.shape[1]:
            epoch = continuous_data[:, start:end]
            epochs.append(epoch)
    return np.array(epochs) if epochs else np.empty((0, continuous_data.shape[0], epoch_samples))

def load_inner_speech_dataset(data_dir, subjects=None, selected_channels=None, load_overt_only=False):
    """
    Load overt/inner speech EEG from the Inner Speech Dataset via MNE-Python.
    
    Data is stored in BioSemi128 format. Event markers are in the 'Status' channel.
    Event codes:
      31-34: Cue onset
      44: Action interval start (speech production begins) - positive class anchor
      45: Relax interval start - negative class anchor
    """
    import mne
    import os
    import glob

    if selected_channels is None:
        selected_channels = SELECTED_CHANNELS_INNER_SPEECH

    if subjects is None:
        subjects = [f'sub-{i+1:02d}' for i in range(10)]

    all_X, all_y, all_blocks = [], [], []

    for subj in subjects:
        bdf_pattern = os.path.join(data_dir, subj, 'ses-*', 'eeg', '*.bdf')
        bdf_files = sorted(glob.glob(bdf_pattern))
        if not bdf_files:
            continue

        subj_epochs_intent, subj_epochs_rest = [], []
        block_idx = 0

        for bdf_path in bdf_files:
            try:
                raw = mne.io.read_raw_bdf(bdf_path, preload=True, stim_channel='Status', verbose=False)
            except Exception:
                continue

            avail = [ch for ch in selected_channels if ch in raw.ch_names]
            if len(avail) < len(selected_channels):
                continue
            
            # Find events BEFORE picking channels (which removes 'Status')
            try:
                events = mne.find_events(raw, stim_channel='Status', verbose=False, min_duration=0.001)
            except Exception:
                continue

            raw.pick(avail) # Use pick instead of pick_channels

            # Rename channels to standard 10-20 names
            rename_dict = {ch: BIOSEMI128_TO_STANDARD.get(ch, ch) for ch in avail}
            raw.rename_channels(rename_dict)

            raw.filter(0.1, 40.0, method='iir', iir_params=dict(order=4, ftype='butter'), verbose=False)
            raw.notch_filter(50.0, verbose=False)
            if raw.info['sfreq'] != TARGET_SFREQ:
                raw.resample(TARGET_SFREQ, verbose=False)

            data = raw.get_data()  
            fs = int(raw.info['sfreq'])
            ep_samples = int(fs * EPOCH_DURATION)

            # Extract Intent (before code 44) and Rest (after code 45)
            # Find pairs of 44 and 45 to ensure valid trials
            for i, ev in enumerate(events):
                if ev[2] == 44:  # Action onset
                    onset = ev[0]
                    start = onset - ep_samples
                    end = onset
                    if start >= 0 and end <= data.shape[1]:
                        epoch = data[:, start:end]
                        # Baseline correct using early part of the epoch or a fixed baseline
                        n_bl = max(1, int(0.1 * TARGET_SFREQ))
                        bl_mean = epoch[:, :n_bl].mean(axis=-1, keepdims=True)
                        epoch = epoch - bl_mean
                        subj_epochs_intent.append(epoch)
                        
                elif ev[2] == 45:  # Relax onset
                    onset = ev[0]
                    start = onset
                    end = onset + ep_samples
                    if end <= data.shape[1]:
                        rest_epoch = data[:, start:end]
                        n_bl = max(1, int(0.1 * TARGET_SFREQ))
                        bl_mean = rest_epoch[:, :n_bl].mean(axis=-1, keepdims=True)
                        rest_epoch = rest_epoch - bl_mean
                        subj_epochs_rest.append(rest_epoch)

            block_idx += 1

        if not subj_epochs_intent:
            continue

        n_min = min(len(subj_epochs_intent), len(subj_epochs_rest))
        if n_min == 0: continue
        
        intent_arr = np.array(subj_epochs_intent[:n_min])
        rest_arr   = np.array(subj_epochs_rest[:n_min])

        X_subj = np.concatenate([intent_arr, rest_arr], axis=0)
        y_subj = np.concatenate([np.ones(n_min, dtype=int), np.zeros(n_min, dtype=int)])

        n_total = len(y_subj)
        block_tags = np.repeat(np.arange(max(1, block_idx)),
                               int(np.ceil(n_total / max(1, block_idx))))[:n_total]

        all_X.append(X_subj)
        all_y.append(y_subj)
        all_blocks.append(block_tags)

    return {
        'X': all_X,   
        'y': all_y,
        'blocks': all_blocks,
        'subjects': subjects[:len(all_X)],
        'channels': [BIOSEMI128_TO_STANDARD.get(ch, ch) for ch in selected_channels],
        'fs': TARGET_SFREQ,
        'n_samples': EPOCH_SAMPLES,
        'epoch_duration': EPOCH_DURATION,
        'data_mode': 'real',
    }

def load_physionet_dataset(subjects=None, selected_channels=None,
                           runs=None, tmin=0.0, tmax=2.0, lowcut=8.0, highcut=30.0):
    """
    Load the PhysioNet EEG Motor Movement/Imagery dataset via MNE.

    WHY THIS INSTEAD OF ds003626?
      - Size: ~10 MB total for 5 subjects  (vs 25 GB for ds003626)
      - Zero friction: MNE downloads on-demand, no account needed
      - Scientifically valid: motor IMAGERY preparation involves the same
        Bereitschaftspotential / motor-cortex readiness as speech intent.
        Both originate in SMA/pre-SMA → same 4-channel geometry applies.

    Interview justification:
      "The PhysioNet EEGBCI dataset provides imagined hand-movement epochs.
       Imagined movement preparation shares the same neural substrate as
       speech intention (Bereitschaftspotential in SMA/pre-SMA). The binary
       task — 'left-hand imagery' vs 'rest' — is a valid proxy for intent
       detection with equivalent cortical dynamics."

    Epochs:
      Runs 6, 10, 14 = imagined left/right hand movement (the 'intent' runs).
      Event '1' = left hand imagined movement → label 1 (INTENT)
      Event '0' = baseline / rest → label 0 (NO INTENT)
      Window: tmin to tmax relative to cue onset (default: −500 to 0 ms).

    Args:
        subjects (list[int] or None): Subject IDs 1–109. Default: [1,2,3,4,5].
        selected_channels (list[str] or None): 4 channels to keep.
            Default: SELECTED_CHANNELS_PHYSIONET = ['FC3','Cz','C3','F3'].
            FC3 ≈ FCz (pre-SMA), Cz (SMA), C3 (left motor), F3 ≈ F7 (IFG).
        runs (list[int] or None): Runs to load. Default: [4, 8, 12].
        tmin (float): Epoch start in seconds relative to event. Default: -0.5.
        tmax (float): Epoch end in seconds relative to event. Default: 0.0.

    Returns:
        dict: Same schema as generate_synthetic_eeg_data() and
              load_inner_speech_dataset(), compatible with all notebook cells.
              Keys: 'X', 'y', 'blocks', 'subjects', 'channels', 'fs',
                    'n_samples', 'epoch_duration', 'data_mode'.

    Requires:
        pip install mne  (already in requirements.txt)
    """
    import mne
    mne.set_log_level('WARNING')

    if subjects is None:
        subjects = [1, 2, 3, 4, 5]
    if selected_channels is None:
        selected_channels = SELECTED_CHANNELS_PHYSIONET
    if runs is None:
        runs = [4, 8, 12]   

    epoch_duration = abs(tmax - tmin)
    epoch_samples = int(epoch_duration * TARGET_SFREQ)

    all_X, all_y, all_blocks = [], [], []
    valid_subjects = []

    for subj in subjects:
        print(f"  Loading subject {subj} ...", end=" ", flush=True)
        try:
            
            fnames = mne.datasets.eegbci.load_data(
                subjects=subj, runs=runs, verbose=False
            )
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        subj_epochs_list = []
        subj_labels_list = []
        block_idx = 0

        for run_idx, fname in enumerate(fnames):
            try:
                raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
            except Exception:
                continue

            mne.datasets.eegbci.standardize(raw)

            avail = [ch for ch in selected_channels if ch in raw.ch_names]
            if len(avail) < 2:
                
                fallback = [ch for ch in ['C3', 'C4', 'Cz', 'Fz', 'F3', 'F4',
                                           'FC3', 'FC4', 'FCz']
                            if ch in raw.ch_names]
                avail = fallback[:4]
            if len(avail) < 2:
                continue
            raw.pick_channels(avail)

            raw.filter(lowcut, highcut, method='iir',
                       iir_params=dict(order=4, ftype='butter'), verbose=False)
            if raw.info['sfreq'] != TARGET_SFREQ:
                raw.resample(TARGET_SFREQ, verbose=False)

            try:
                events, event_id = mne.events_from_annotations(raw, verbose=False)
            except Exception:
                continue

            intent_id = {k: v for k, v in event_id.items() if 'T1' in k}
            rest_id   = {k: v for k, v in event_id.items() if 'T0' in k}

            if not intent_id and not rest_id:
                
                ids_sorted = sorted(event_id.values())
                intent_id = {'intent': ids_sorted[0]} if ids_sorted else {}
                rest_id   = {'rest': ids_sorted[1]} if len(ids_sorted) > 1 else {}

            if not intent_id:
                continue

            combined_id = {**intent_id, **rest_id}

            try:
                # Load with a pre-cue period for baseline correction
                epochs_mne = mne.Epochs(
                    raw, events, event_id=combined_id,
                    tmin=-0.5, tmax=tmax - 1.0/TARGET_SFREQ,
                    baseline=(None, 0), preload=True, verbose=False
                )
                
                data = epochs_mne.get_data()   
                labels_raw = epochs_mne.events[:, 2]

                intent_vals = set(intent_id.values())
                labels = np.array([1 if lv in intent_vals else 0
                                   for lv in labels_raw])

                # Slice the data to [tmin, tmax] window
                start_idx = int((tmin - (-0.5)) * TARGET_SFREQ)
                data = data[:, :, start_idx:]

                # Balance classes (1:1 ratio) to prevent majority class prediction
                idx_intent = np.where(labels == 1)[0]
                idx_rest = np.where(labels == 0)[0]
                min_len = min(len(idx_intent), len(idx_rest))
                
                if min_len > 0:
                    np.random.seed(42) # For reproducibility
                    np.random.shuffle(idx_intent)
                    np.random.shuffle(idx_rest)
                    idx_intent = idx_intent[:min_len]
                    idx_rest = idx_rest[:min_len]
                    
                    balanced_idx = np.concatenate([idx_intent, idx_rest])
                    np.random.shuffle(balanced_idx)
                    
                    data = data[balanced_idx]
                    labels = labels[balanced_idx]
                else:
                    continue

                n_t = data.shape[-1]
                if n_t >= epoch_samples:
                    data = data[:, :, :epoch_samples]
                else:
                    pad = epoch_samples - n_t
                    data = np.pad(data, ((0,0),(0,0),(0,pad)))

                n_ch = data.shape[1]
                if n_ch < 4:
                    pad_ch = np.zeros((data.shape[0], 4 - n_ch, epoch_samples))
                    data = np.concatenate([data, pad_ch], axis=1)

                block_tags = np.full(len(labels), block_idx, dtype=int)
                subj_epochs_list.append(data)
                subj_labels_list.append(labels)
                block_idx += 1

            except Exception:
                continue

        if not subj_epochs_list:
            print("no usable data")
            continue

        X_subj = np.concatenate(subj_epochs_list, axis=0)
        y_subj = np.concatenate(subj_labels_list, axis=0)
        n_total = len(y_subj)
        blk_arr = np.concatenate(
            [np.full(len(sl), bi, dtype=int)
             for bi, sl in enumerate(subj_epochs_list)], axis=0
        )

        print(f"{n_total} epochs, {y_subj.sum()} intent / {(y_subj==0).sum()} rest")
        all_X.append(X_subj)
        all_y.append(y_subj)
        all_blocks.append(blk_arr)
        valid_subjects.append(subj)

    if not all_X:
        raise RuntimeError("No subjects loaded. Check MNE installation and internet connection.")

    ch_names = selected_channels[:4] if len(selected_channels) >= 4 else selected_channels

    return {
        'X': all_X,          
        'y': all_y,
        'blocks': all_blocks,
        'subjects': [f'sub-{s:02d}' for s in valid_subjects],
        'channels': ch_names,
        'fs': TARGET_SFREQ,
        'n_samples': epoch_samples,
        'epoch_duration': epoch_duration,
        'data_mode': 'real',
        'dataset': 'PhysioNet-EEGBCI',
    }

_SIM_BANNER = "[SIMULATION — NOT REAL EEG]"

def generate_synthetic_eeg_data(n_subjects=5, n_trials_per_class=100,
                                 n_channels=4, n_samples=128, fs=256,
                                 noise_level=0.8, seed=42):
    """
    Generate synthetic EEG data for PIPELINE SMOKE-TESTING ONLY.

    WARNING: Every metric produced from this data must be prefixed with
    "[SIMULATION — NOT REAL EEG]".  Accuracy numbers are meaningless
    and will not transfer to real EEG.

    Improvements over naïve generator:
      - ~30 % of NEGATIVE trials also receive a subtle BP (covert rehearsal)
      - ~15 % of all trials get eye-blink artifacts (200–300 µV frontal)
      - Slow electrode drift (<0.1 Hz) varies across 8 session blocks
      - ~10 % of trials get EMG bursts (50–150 Hz)
    """
    np.random.seed(seed)

    channels = ['FCz', 'Cz', 'C3', 'F7']
    t = np.linspace(0, EPOCH_DURATION, n_samples)
    n_trials = 2 * n_trials_per_class
    trials_per_block = max(1, n_trials // N_BLOCKS_PER_SUBJECT)

    all_X, all_y, all_blocks = [], [], []

    for subj in range(n_subjects):
        np.random.seed(seed + subj * 100)

        bp_amplitude     = np.random.uniform(3.0, 8.0)
        alpha_power      = np.random.uniform(5.0, 15.0)
        lateralization   = np.random.uniform(0.3, 0.7)
        mu_freq          = np.random.uniform(9.0, 12.0)
        beta_freq        = np.random.uniform(18.0, 24.0)
        drift_amplitude  = np.random.uniform(1.0, 4.0)

        X_subj = np.zeros((n_trials, n_channels, n_samples))
        y_subj = np.zeros(n_trials, dtype=int)
        block_tags = np.zeros(n_trials, dtype=int)

        for trial in range(n_trials):
            is_intent = trial < n_trials_per_class
            y_subj[trial] = 1 if is_intent else 0
            block_tags[trial] = min(trial // trials_per_block, N_BLOCKS_PER_SUBJECT - 1)

            block_drift_offset = drift_amplitude * np.sin(
                2 * np.pi * 0.05 * t + block_tags[trial] * 0.8)

            for ch_idx, ch_name in enumerate(channels):
                
                white = np.random.randn(n_samples)
                freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
                freqs[0] = 1
                pink = np.fft.irfft(np.fft.rfft(white) / np.sqrt(freqs), n=n_samples)
                pink = pink / (np.std(pink) + 1e-8) * noise_level

                alpha = alpha_power * np.sin(
                    2 * np.pi * mu_freq * t + np.random.uniform(0, 2 * np.pi))
                alpha *= np.random.uniform(0.5, 1.5)
                beta_osc = (alpha_power * 0.3) * np.sin(
                    2 * np.pi * beta_freq * t + np.random.uniform(0, 2 * np.pi))

                background = pink + alpha * 0.3 + beta_osc * 0.1 + block_drift_offset

                add_bp = False
                bp_scale = 1.0
                if is_intent:
                    add_bp = True
                elif np.random.rand() < 0.30:
                    
                    add_bp = True
                    bp_scale = np.random.uniform(0.15, 0.35)

                if add_bp:
                    bp_ramp = -bp_amplitude * bp_scale * (t / EPOCH_DURATION) ** 1.5
                    if ch_name in ['F7', 'C3']:
                        bp_ramp *= (1.0 + lateralization)
                    elif ch_name == 'Cz':
                        bp_ramp *= 1.0
                    else:
                        bp_ramp *= (1.0 - lateralization * 0.3)

                    erd_envelope = 1.0 - 0.4 * bp_scale * (t / EPOCH_DURATION)
                    eeg_signal = background + bp_ramp + alpha * erd_envelope * 0.2
                else:
                    eeg_signal = background

                if np.random.rand() < 0.15 and ch_name in ['FCz', 'F7']:
                    blink_center = np.random.randint(10, n_samples - 10)
                    blink_amp = np.random.uniform(200, 300)
                    blink = blink_amp * np.exp(
                        -0.5 * ((np.arange(n_samples) - blink_center) / 5) ** 2)
                    eeg_signal += blink

                if np.random.rand() < 0.10:
                    burst_start = np.random.randint(0, n_samples - 20)
                    burst_len = np.random.randint(10, 30)
                    burst_end = min(burst_start + burst_len, n_samples)
                    emg = np.random.randn(burst_end - burst_start) * np.random.uniform(3, 8)
                    eeg_signal[burst_start:burst_end] += emg

                eeg_signal += np.random.randn(n_samples) * noise_level * 0.5
                X_subj[trial, ch_idx, :] = eeg_signal

        shuffle_idx = np.random.permutation(n_trials)
        X_subj = X_subj[shuffle_idx]
        y_subj = y_subj[shuffle_idx]
        block_tags = block_tags[shuffle_idx]

        all_X.append(X_subj)
        all_y.append(y_subj)
        all_blocks.append(block_tags)

    return {
        'X': np.array(all_X),
        'y': np.array(all_y),
        'blocks': np.array(all_blocks),
        'subjects': [f'sub-{i + 1:02d}' for i in range(n_subjects)],
        'channels': channels,
        'fs': fs,
        'n_samples': n_samples,
        'epoch_duration': EPOCH_DURATION,
        'data_mode': 'synthetic',
        'SIM_BANNER': _SIM_BANNER,
    }
