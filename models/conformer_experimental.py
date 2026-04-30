"""
EEG-Conformer: Convolutional Transformer for EEG Classification  [EXPERIMENTAL — DO NOT USE AS PRIMARY]
========================================================================================================

⚠️  WARNING: This model underperforms EEGNet at our scale.

Reasons NOT to use as primary model:
  1. ~60,000 parameters with ~500 trials and 4 channels → severe overfitting.
  2. Self-attention requires meaningful spatial structure across tokens.
     With only 4 channels, there is no rich spatial structure to exploit.
  3. The Bereitschaftspotential is a low-frequency (<1 Hz) slow cortical
     potential — Transformer attention over high-frequency patches adds
     complexity without signal gain.
  4. Even in the high-density EEG literature (64–256 channels), Conformers
     only match or marginally beat EEGNet.

✅  Recommended alternative: EEGNetWithAttention (models/eegnet.py)
  → Gives the "modern ML touch" with ~100–200 extra parameters.
  → Lightweight temporal attention over the separable conv output.
  → Suitable for 4-channel, low-data EEG.

Interview answer — "Why didn't you use Transformers?":
  "Given the low-channel, low-data EEG setting, Transformers tend to overfit
   and lack meaningful spatial structure to exploit. Instead, we use EEGNet
   with optional lightweight temporal attention, which is more suitable for
   this regime. The Bereitschaftspotential is a slow cortical potential —
   attention over local CNN features is sufficient."

Included for reference only. Do NOT report as a primary result.

Reference: Song et al., "EEG Conformer: Convolutional Transformer for EEG Decoding
and Visualization", IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2023.

Architecture Overview:
    1. Convolutional Tokenizer: temporal + spatial convolutions produce patch embeddings
    2. Positional Encoding: learnable position embeddings
    3. Transformer Encoder: multi-head self-attention + feed-forward blocks
    4. Classification Head: layer norm → mean pooling → linear
"""

import math
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Convolutional Patch Embedding for EEG.
    
    Converts raw EEG (C × T) into a sequence of patch tokens using 
    temporal and spatial convolutions, analogous to ViT's patch embedding 
    but adapted for the 2D structure of multi-channel EEG.
    """
    
    def __init__(self, n_channels=4, n_filters=40, kernel_length=25, 
                 pool_length=8, dropout_rate=0.5):
        super(PatchEmbedding, self).__init__()
        
        self.conv_block = nn.Sequential(
            
            nn.Conv2d(1, n_filters, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            
            nn.Conv2d(n_filters, n_filters, (n_channels, 1), groups=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            nn.AvgPool2d((1, pool_length)),
            nn.Dropout(dropout_rate),
        )
        
        self.embed_dim = n_filters
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, channels, time)
        Returns:
            tokens: (batch, seq_len, embed_dim)
        """
        x = self.conv_block(x)  
        x = x.squeeze(2)        
        x = x.permute(0, 2, 1)  
        return x

class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer encoder block with pre-norm architecture.
    
    Pre-norm is preferred for EEG because it provides more stable gradients
    during training with the typically small EEG datasets.
    """
    
    def __init__(self, embed_dim=40, num_heads=4, ff_dim=128, dropout=0.3):
        super(TransformerEncoderBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out
        
        return x

class EEGConformer(nn.Module):
    """
    EEG-Conformer: CNN + Transformer hybrid for EEG classification.
    
    Args:
        n_channels (int): Number of EEG channels. Default: 4.
        n_samples (int): Number of time samples. Default: 128.
        n_classes (int): Number of output classes. Default: 2.
        n_filters (int): Number of convolutional filters. Default: 40.
        kernel_length (int): Temporal conv kernel size. Default: 25.
        pool_length (int): Pooling window size. Default: 8.
        embed_dim (int): Transformer embedding dimension (= n_filters). Default: 40.
        num_heads (int): Number of attention heads. Default: 4.
        ff_dim (int): Feed-forward hidden dimension. Default: 128.
        num_layers (int): Number of Transformer encoder layers. Default: 3.
        dropout (float): Dropout rate. Default: 0.3.
    """
    
    def __init__(self, n_channels=4, n_samples=128, n_classes=2,
                 n_filters=40, kernel_length=25, pool_length=8,
                 embed_dim=40, num_heads=4, ff_dim=128,
                 num_layers=3, dropout=0.3):
        super(EEGConformer, self).__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        
        self.patch_embed = PatchEmbedding(
            n_channels=n_channels, 
            n_filters=n_filters,
            kernel_length=kernel_length, 
            pool_length=pool_length,
            dropout_rate=dropout
        )
        
        self._seq_len = self._get_seq_len(n_channels, n_samples)
        
        self.pos_embed = nn.Parameter(
            torch.randn(1, self._seq_len, embed_dim) * 0.02
        )
        self.pos_dropout = nn.Dropout(dropout)
        
        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_classes)
        )
    
    def _get_seq_len(self, n_channels, n_samples):
        """Compute the token sequence length after patch embedding."""
        x = torch.zeros(1, 1, n_channels, n_samples)
        x = self.patch_embed(x)
        return x.shape[1]
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input of shape (batch, channels, time) or (batch, 1, channels, time).
        
        Returns:
            torch.Tensor: Logits of shape (batch, n_classes).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        tokens = self.patch_embed(x)  
        
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        tokens = self.pos_dropout(tokens)
        
        tokens = self.transformer(tokens)  
        
        cls_token = tokens.mean(dim=1)  
        
        logits = self.head(cls_token)  
        return logits
    
    def freeze_encoder(self):
        """Freeze the convolutional and transformer encoder for fine-tuning."""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.pos_embed:
            self.pos_embed.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_num_params(self, only_trainable=True):
        """Return the number of parameters."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

if __name__ == "__main__":
    model = EEGConformer(n_channels=4, n_samples=128, n_classes=2)
    x = torch.randn(8, 4, 128)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters:   {model.get_num_params():,}")
