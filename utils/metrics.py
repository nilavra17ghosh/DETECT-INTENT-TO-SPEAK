"""
Evaluation Metrics, Visualization, and Analysis for EEG Classification
=======================================================================

Includes:
  - compute_metrics            : accuracy, F1, precision, recall, AUC-ROC
  - plot_confusion_matrix      : heatmap
  - plot_roc_curve             : ROC with AUC
  - plot_training_history      : loss / accuracy curves
  - plot_model_comparison      : bar chart
  - compute_latency_accuracy   : performance at −100/−200/−300 ms windows
  - plot_latency_analysis      : visualise detection latency curve
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             confusion_matrix, balanced_accuracy_score)
import seaborn as sns

def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute all evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='binary'),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc_roc'] = 0.5
    return metrics

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Intent', 'Intent'],
                yticklabels=['No Intent', 'Intent'], ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

def plot_roc_curve(y_true, y_prob, title='ROC Curve', save_path=None):
    """Plot ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

def plot_training_history(train_losses, val_losses, train_accs, val_accs,
                          title='Training History', save_path=None):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

def plot_model_comparison(results_dict, metric='accuracy', save_path=None):
    """Bar plot comparing multiple models."""
    models = list(results_dict.keys())
    values = [results_dict[m][metric] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    bars = ax.bar(models, values, color=colors[:len(models)], edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Model Comparison — {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

def compute_latency_accuracy(X, y, model, device, fs=256,
                              window_ends_ms=None):
    """
    Evaluate classification accuracy at different detection horizons.

    Tests how early (before speech onset) the model can reliably detect
    intent.  This demonstrates real-world usability of the system.

    Each evaluation uses a sub-window of the full epoch:
      - window_end = −100 ms: uses the last 100 ms of the 500 ms epoch
        (most informative — late BP / NS')
      - window_end = −200 ms: uses the last 300 ms
      - window_end = −300 ms: uses the last 400 ms (mostly early BP)

    The sub-window always ENDS at the speech onset (t=0).

    Args:
        X (np.ndarray): Shape (n_trials, n_channels, n_samples).
            Full 500 ms epochs (−500 to 0 ms).
        y (np.ndarray): Shape (n_trials,). Binary labels.
        model: PyTorch model with .eval() and .forward() returning logits.
        device: torch.device.
        fs (int): Sampling frequency in Hz. Default: 256.
        window_ends_ms (list[int] or None): Detection latencies in ms before
            speech onset (negative values). Default: [-100, -200, -300, -400, -500].

    Returns:
        dict: Maps each window_end_ms → {'accuracy': float, 'n_samples': int}
    """
    import torch

    if window_ends_ms is None:
        window_ends_ms = [-100, -200, -300, -400, -500]

    total_samples = X.shape[-1]   
    epoch_duration_ms = 500        

    results = {}
    model.eval()

    for end_ms in window_ends_ms:
        
        n_window_samples = max(1, int(abs(end_ms) / 1000.0 * fs))
        start_idx = total_samples - n_window_samples

        X_sub = X[:, :, start_idx:]   

        pad_len = total_samples - n_window_samples
        X_padded = np.concatenate(
            [np.zeros((X.shape[0], X.shape[1], pad_len)), X_sub], axis=-1
        )

        X_tensor = torch.FloatTensor(X_padded).to(device)

        with torch.no_grad():
            logits = model(X_tensor)
            preds = (logits > 0).float().cpu().numpy()

        acc = accuracy_score(y, preds)
        bal_acc = balanced_accuracy_score(y, preds)
        results[end_ms] = {
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'n_samples': n_window_samples,
        }

    return results

def plot_latency_analysis(latency_results, title='Detection Latency Analysis',
                          save_path=None):
    """
    Plot accuracy vs. detection latency (how far before speech onset).

    Shows real-world usability: can the system detect intent early enough
    to be useful?  Key reference points:
      −100 ms: late BP / NS' — most discriminative
      −200 ms: mid BP — good performance
      −300 ms: early BP onset — degraded but still above chance

    Args:
        latency_results (dict): Output of compute_latency_accuracy().
            Keys are window_end_ms (negative ints).
        title (str): Plot title.
        save_path (str or None): Save path.

    Returns:
        matplotlib.figure.Figure
    """
    latencies = sorted(latency_results.keys(), reverse=True)  
    accs = [latency_results[l]['accuracy'] for l in latencies]
    bal_accs = [latency_results[l]['balanced_accuracy'] for l in latencies]
    x_labels = [f'{l} ms' for l in latencies]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_labels, accs, 'b-o', linewidth=2.5, markersize=8, label='Accuracy')
    ax.plot(x_labels, bal_accs, 'r--s', linewidth=2.5, markersize=8, label='Balanced Accuracy')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.7, label='Chance (50%)')

    for i, (l, a, ba) in enumerate(zip(x_labels, accs, bal_accs)):
        ax.annotate(f'{a:.2f}', xy=(i, a), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontsize=9, color='blue')

    ax.set_xlabel('Detection Window End (relative to speech onset)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0.35, 1.05)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    ax.axvspan(-0.5, 0.5, alpha=0.08, color='green', label='Most informative (late BP)')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
