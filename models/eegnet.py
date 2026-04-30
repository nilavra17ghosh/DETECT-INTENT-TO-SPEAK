"""
EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs
==================================================================

Reference: Lawhern et al., J. Neural Eng., 2018.

Output: single logit (scalar) for BCEWithLogitsLoss.

Two variants are provided:
  - EEGNet             : the canonical compact model (~1k params).  PRIMARY model.
  - EEGNetWithAttention: EEGNet + lightweight temporal self-attention.
                         Optional enhancement for capturing long-range BP dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    """
    EEGNet v4 — single-logit output for binary BCE loss.

    PRIMARY MODEL for intent-to-speak detection.
    Designed for EEG, low parameter count (~1k), works well with small data.
    Captures temporal + spatial features efficiently.

    Args:
        n_channels (int): Number of EEG channels (electrodes). Default: 4.
        n_samples (int): Number of time samples per epoch.  Default: 128.
        dropout_rate (float): Dropout probability.  Default: 0.5.
        F1 (int): Number of temporal filters.  Default: 8.
        D (int): Depth multiplier for depthwise conv.  Default: 2.
        F2 (int): Pointwise filters (= F1*D).  Default: 16.
        kernel_length (int): Temporal kernel length (≈ fs/2).  Default: 64.

    Channels:
        FCz → pre-SMA (early BP), Cz → SMA (late BP),
        C3  → motor cortex,       F7 → Broca's area
    """

    def __init__(self, n_channels=4, n_samples=128,
                 dropout_rate=0.5, F1=8, D=2, F2=16, kernel_length=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        self._feat = self._get_feat(n_channels, n_samples)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._feat, 1),
        )

    def _get_feat(self, C, T):
        x = torch.zeros(1, 1, C, T)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        """
        Args:
            x: (B, C, T) or (B, 1, C, T)
        Returns:
            logit: (B,)  — raw logit for BCEWithLogitsLoss
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        return self.classifier(x).squeeze(-1)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_proba(self, x):
        """Helper for sklearn compatibility: returns probability of class 1"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).cpu().numpy()

class TemporalAttention(nn.Module):
    """
    Lightweight temporal self-attention over the feature time-steps produced
    by EEGNet's separable conv block.

    Uses a simple scaled dot-product attention with a single head.
    Much lighter than a full Transformer — no feed-forward sublayer,
    no positional encoding overhead.  ~100–200 extra parameters.

    Why this (not full Transformer):
        Given 4 channels and ~500 trials per subject, a full Transformer
        overfits immediately.  This lightweight variant gives the model
        a way to upweight the late BP component (−100 ms) relative to
        early noise without dramatically increasing parameter count.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, E)  — batch × time-steps × embed_dim
        Returns:
            (B, T, E)  — attention-weighted output + residual
        """
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        attn = self.dropout(attn)
        out = attn @ V
        return self.norm(x + out)  

class EEGNetWithAttention(nn.Module):
    """
    EEGNet + Lightweight Temporal Attention.

    OPTIONAL ENHANCEMENT — use when you want to show a modern ML touch
    without full-Transformer overfitting risk.

    Architecture:
        EEG → [Temporal Conv] → [Spatial Conv] → [Separable Conv]
            → [Temporal Self-Attention] → [Flatten] → [Linear] → logit

    The attention layer operates on the T' time-steps output by EEGNet's
    separable conv, allowing the model to attend to the late BP component
    (the −100 ms window) which is most predictive of imminent speech.

    Args:
        n_channels (int): EEG channels. Default: 4.
        n_samples (int): Time samples. Default: 128.
        dropout_rate (float): Dropout in EEGNet blocks. Default: 0.5.
        attn_dropout (float): Attention dropout. Default: 0.1.
        F1 (int): Temporal filters. Default: 8.
        D (int): Depth multiplier. Default: 2.
        F2 (int): Pointwise filters. Default: 16.
        kernel_length (int): Temporal kernel. Default: 64.
    """

    def __init__(self, n_channels=4, n_samples=128,
                 dropout_rate=0.5, attn_dropout=0.1,
                 F1=8, D=2, F2=16, kernel_length=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        self.attention = TemporalAttention(embed_dim=F2, dropout=attn_dropout)

        self._feat = self._get_feat(n_channels, n_samples, F2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._feat, 1),
        )

    def _get_feat(self, C, T, F2):
        x = torch.zeros(1, 1, C, T)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        
        x = x.squeeze(2).permute(0, 2, 1)  
        x = self.attention(x)
        return x.reshape(1, -1).size(1)

    def forward(self, x):
        """
        Args:
            x: (B, C, T) or (B, 1, C, T)
        Returns:
            logit: (B,)  — raw logit for BCEWithLogitsLoss
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)

        B, F2, _, T_ = x.shape
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.attention(x)

        return self.classifier(x).squeeze(-1)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_proba(self, x):
        """Helper for sklearn compatibility: returns probability of class 1"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).cpu().numpy()

def init_weights(m):
    """Initialize weights for Conv2d and Linear layers"""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def build_eegnet(n_channels=4, n_samples=128, use_attention=False):
    """
    Factory function to build and initialize EEGNet.
    
    Args:
        n_channels: Number of EEG channels
        n_samples: Number of time samples
        use_attention: Whether to use the EEGNetWithAttention variant
    """
    if use_attention:
        model = EEGNetWithAttention(n_channels=n_channels, n_samples=n_samples)
    else:
        model = EEGNet(n_channels=n_channels, n_samples=n_samples)
    
    model.apply(init_weights)
    return model

if __name__ == "__main__":
    print("=== EEGNet (primary) ===")
    m = EEGNet(n_channels=4, n_samples=128)
    x = torch.randn(8, 4, 128)
    print(f"Input:  {x.shape}")
    print(f"Output: {m(x).shape}")        
    print(f"Params: {m.get_num_params():,}")

    print("\n=== EEGNetWithAttention (optional enhancement) ===")
    m2 = EEGNetWithAttention(n_channels=4, n_samples=128)
    print(f"Input:  {x.shape}")
    print(f"Output: {m2(x).shape}")       
    print(f"Params: {m2.get_num_params():,}")
