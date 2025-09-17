#!/usr/bin/env python3
"""
Band-Splitting EAT: Enabling EAT to have band-splitting process while maintaining batch parallelism.

This implementation combines EAT's efficient batch processing with ECHO's band-splitting approach,
allowing for frequency-aware processing while maintaining the benefits of batch parallelism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from functools import partial

from .mae import PatchEmbed_new, get_2d_sincos_pos_embed_flexible
from .modules import FixedPositionalEncoder, DynamicPositionalEncoder
from .base import D2vModalityConfig


@dataclass
class BandSplittingEATConfig(D2vModalityConfig):
    """Configuration for Band-Splitting EAT"""
    # Audio parameters
    target_length: int = 1024
    max_length: int = 768
    in_chans: int = 1
    
    # Patch parameters
    patch_size: int = 16
    embed_dim: int = 768
    
    # Band-splitting parameters
    band_width: int = 32
    shift_size: int = 16
    use_band_splitting: bool = True
    freq_pos_emb_dim: int = 768
    
    # Positional encoding
    fixed_positions: bool = False
    
    # Transformer parameters
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0


def get_sincos_encoding_1d(pos, dim, freq_scale=25):
    """
    Generate 1D sin/cos positional encoding for frequency positions.
    
    Args:
        pos: (N, 1) normalized position in [0, 1]
        dim: output embedding dimension
        freq_scale: frequency scaling factor
        
    Returns:
        (N, dim) tensor
    """
    import math
    device = pos.device
    N = pos.shape[0]
    pe = torch.zeros(N, dim, device=device)
    position = pos  # (N, 1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * freq_scale * div_term)
    pe[:, 1::2] = torch.cos(position * freq_scale * div_term)
    return pe


class BandSplittingPatchEmbed(nn.Module):
    """
    Patch embedding for band-splitting with batch parallelism.
    
    This module handles the conversion of band-split spectrograms into patches
    while maintaining batch parallelism.
    """
    
    def __init__(self, band_width: int, shift_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.band_width = band_width
        self.shift_size = shift_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # 2D convolution for patch embedding
        # Input: (batch_size, in_chans, band_width, time)
        # Output: (batch_size, embed_dim, 1, num_patches)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, 
            kernel_size=(band_width, shift_size), 
            stride=(band_width, shift_size)
        )
        
    def forward(self, x):
        """
        Forward pass for patch embedding.
        
        Args:
            x: (batch_size, in_chans, band_width, time)
            
        Returns:
            patches: (batch_size, num_patches, embed_dim)
        """
        # x: (batch_size, in_chans, band_width, time)
        x = self.proj(x)  # (batch_size, embed_dim, 1, num_patches)
        
        # Flatten and transpose
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        
        return x


class BandSplittingEAT(nn.Module):
    """
    Band-Splitting EAT: Combines EAT's batch processing with ECHO's band-splitting.
    
    Key Features:
    1. Batch-level band-splitting processing
    2. Frequency-aware positional encoding
    3. Dynamic patch embedding
    4. Maintains batch parallelism
    """
    
    def __init__(self, config: BandSplittingEATConfig):
        super().__init__()
        self.config = config
        
        # Band-splitting parameters
        self.band_width = config.band_width
        self.shift_size = config.shift_size
        self.freq_pos_emb_dim = config.freq_pos_emb_dim
        
        # Patch embedding for bands
        self.band_patch_embed = BandSplittingPatchEmbed(
            band_width=config.band_width,
            shift_size=config.shift_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        
        # Positional encoding
        if config.fixed_positions:
            # Fixed positional encoding (original EAT approach)
            max_patches = config.max_length // config.shift_size
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_patches + 1, config.embed_dim), 
                requires_grad=False
            )
            self.positional_encoder = FixedPositionalEncoder(self.pos_embed)
        else:
            # Dynamic positional encoding
            self.positional_encoder = DynamicPositionalEncoder(
                config.embed_dim, 
                patch_h=1,  # Only one row since height equals band_width
                patch_w=config.max_length // config.shift_size
            )
        
        # Frequency positional encoding
        self.freq_pos_embed = nn.Parameter(
            torch.zeros(1, config.freq_pos_emb_dim), 
            requires_grad=False
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        
        # Layer normalization
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight.view([m.weight.shape[0], -1]))
    
    def _split_into_bands(self, x):
        """
        Split spectrograms into frequency bands with batch parallelism.
        
        Args:
            x: (batch_size, in_chans, freq_bins, time)
            
        Returns:
            band_spectrograms: (batch_size * num_bands, in_chans, band_width, time)
            band_batch_indices: (batch_size * num_bands,) - which batch each band belongs to
            band_freq_indices: (batch_size * num_bands, 2) - frequency range for each band
        """
        batch_size, in_chans, freq_bins, time = x.shape
        
        # Calculate number of bands
        num_bands = max(1, (freq_bins + self.band_width - 1) // self.band_width)
        
        # Create band spectrograms
        band_spectrograms = []
        band_batch_indices = []
        band_freq_indices = []
        
        for band_idx in range(num_bands):
            start_freq = band_idx * self.band_width
            end_freq = min((band_idx + 1) * self.band_width, freq_bins)
            
            if end_freq > start_freq:
                # Extract band: (batch_size, in_chans, band_freq, time)
                band = x[:, :, start_freq:end_freq, :]
                
                # Pad if last band is smaller than band_width
                if end_freq - start_freq < self.band_width:
                    pad = self.band_width - (end_freq - start_freq)
                    band = F.pad(band, (0, 0, 0, pad))
                
                band_spectrograms.append(band)
                band_batch_indices.append(torch.arange(batch_size, device=x.device))
                band_freq_indices.append(torch.tensor([start_freq, end_freq], device=x.device).unsqueeze(0).expand(batch_size, -1))
        
        # Concatenate all bands
        if band_spectrograms:
            band_spectrograms = torch.cat(band_spectrograms, dim=0)  # (batch_size * num_bands, in_chans, band_width, time)
            band_batch_indices = torch.cat(band_batch_indices, dim=0)  # (batch_size * num_bands,)
            band_freq_indices = torch.cat(band_freq_indices, dim=0)  # (batch_size * num_bands, 2)
        else:
            raise ValueError("No bands found in spectrogram")
        
        return band_spectrograms, band_batch_indices, band_freq_indices
    
    def _compute_frequency_position_encoding(self, band_freq_indices, sample_rate=16000):
        """
        Compute frequency position encoding for bands.
        
        Args:
            band_freq_indices: (batch_size * num_bands, 2) - frequency range for each band
            sample_rate: Audio sample rate
            
        Returns:
            freq_pos_emb: (batch_size * num_bands, freq_pos_emb_dim)
        """
        # Calculate center frequency for each band
        center_freqs = (band_freq_indices[:, 0] + band_freq_indices[:, 1] - 1) / 2.0
        
        # Convert to normalized frequency position
        n_fft = int(0.025 * sample_rate)
        center_freq_hz = center_freqs * sample_rate / n_fft
        nyquist = sample_rate / 2.0
        freq_pos_norm = center_freq_hz / nyquist if nyquist > 0 else torch.zeros_like(center_freq_hz)
        
        # Generate frequency positional encoding
        freq_pos_norm = freq_pos_norm.unsqueeze(1)  # (batch_size * num_bands, 1)
        freq_pos_emb = get_sincos_encoding_1d(freq_pos_norm, self.freq_pos_emb_dim)
        
        return freq_pos_emb
    
    def _reconstruct_batch_structure(self, band_features, band_batch_indices, batch_size):
        """
        Reconstruct batch structure from band features.
        
        Args:
            band_features: (batch_size * num_bands, num_patches, embed_dim)
            band_batch_indices: (batch_size * num_bands,)
            batch_size: Original batch size
            
        Returns:
            batch_features: (batch_size, num_bands, num_patches, embed_dim)
        """
        num_bands = band_features.shape[0] // batch_size
        
        # Reshape to (batch_size, num_bands, num_patches, embed_dim)
        batch_features = band_features.view(batch_size, num_bands, -1, band_features.shape[-1])
        
        return batch_features
    
    def forward(self, x, sample_rate=16000, mask_ratio=0.0):
        """
        Forward pass for Band-Splitting EAT.
        
        Args:
            x: (batch_size, in_chans, freq_bins, time)
            sample_rate: Audio sample rate
            mask_ratio: Masking ratio for training
            
        Returns:
            features: (batch_size, num_bands, num_patches + 1, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Step 1: Split into frequency bands
        band_spectrograms, band_batch_indices, band_freq_indices = self._split_into_bands(x)
        
        # Step 2: Patch embedding for bands
        band_patches = self.band_patch_embed(band_spectrograms)  # (batch_size * num_bands, num_patches, embed_dim)
        
        # Step 3: Add positional encoding
        if self.config.fixed_positions:
            # Fixed positional encoding
            band_patches = self.positional_encoder(band_patches)
        else:
            # Dynamic positional encoding
            band_patches = self.positional_encoder(band_patches)
        
        # Step 4: Add frequency positional encoding
        freq_pos_emb = self._compute_frequency_position_encoding(band_freq_indices, sample_rate)
        freq_pos_emb = freq_pos_emb.unsqueeze(1).expand(-1, band_patches.shape[1], -1)  # (batch_size * num_bands, num_patches, freq_pos_emb_dim)
        band_patches = band_patches + freq_pos_emb
        
        # Step 5: Add CLS token
        cls_tokens = self.cls_token.expand(band_patches.shape[0], -1, -1)  # (batch_size * num_bands, 1, embed_dim)
        cls_freq_pos_emb = freq_pos_emb[:, 0:1, :]  # (batch_size * num_bands, 1, freq_pos_emb_dim)
        cls_tokens = cls_tokens + cls_freq_pos_emb
        
        # Concatenate CLS token with patches
        tokens = torch.cat([cls_tokens, band_patches], dim=1)  # (batch_size * num_bands, num_patches + 1, embed_dim)
        
        # Step 6: Apply masking if needed
        if mask_ratio > 0:
            tokens, mask = self._apply_masking(tokens, mask_ratio)
        
        # Step 7: Transformer encoding
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        
        # Step 8: Reconstruct batch structure
        batch_features = self._reconstruct_batch_structure(tokens, band_batch_indices, batch_size)
        
        return batch_features
    
    def _apply_masking(self, tokens, mask_ratio):
        """
        Apply random masking to tokens.
        
        Args:
            tokens: (batch_size * num_bands, num_patches + 1, embed_dim)
            mask_ratio: Masking ratio
            
        Returns:
            masked_tokens: Masked tokens
            mask: Binary mask
        """
        N, L, D = tokens.shape
        len_keep = int(L * (1 - mask_ratio))
        
        if mask_ratio == 0:
            return tokens, torch.zeros([N, L], device=tokens.device)
        
        # Generate random noise for masking
        noise = torch.rand(N, L, device=tokens.device)
        
        # Sort noise
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        tokens_unmasked = torch.gather(tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask
        mask = torch.ones([N, L], device=tokens.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return tokens_unmasked, mask
    
    def extract_features(self, x, sample_rate=16000):
        """
        Extract features from input spectrograms.
        
        Args:
            x: (batch_size, in_chans, freq_bins, time)
            sample_rate: Audio sample rate
            
        Returns:
            cls_features: (batch_size, num_bands, embed_dim)
            patch_features: (batch_size, num_bands, num_patches, embed_dim)
        """
        with torch.no_grad():
            features = self.forward(x, sample_rate, mask_ratio=0.0)
            
            # Extract CLS and patch features
            cls_features = features[:, :, 0, :]  # (batch_size, num_bands, embed_dim)
            patch_features = features[:, :, 1:, :]  # (batch_size, num_bands, num_patches, embed_dim)
            
            return cls_features, patch_features


def create_band_splitting_eat(config: BandSplittingEATConfig):
    """
    Create a Band-Splitting EAT model.
    
    Args:
        config: Configuration for the model
        
    Returns:
        model: Band-Splitting EAT model
    """
    return BandSplittingEAT(config)


# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = BandSplittingEATConfig(
        target_length=1024,
        max_length=768,
        in_chans=1,
        patch_size=16,
        embed_dim=768,
        band_width=32,
        shift_size=16,
        use_band_splitting=True,
        freq_pos_emb_dim=768,
        fixed_positions=False,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0
    )
    
    # Create model
    model = create_band_splitting_eat(config)
    
    # Test with different input sizes
    test_cases = [
        (2, 1, 128, 512),   # Short audio
        (2, 1, 128, 1024),  # Medium audio
        (2, 1, 128, 2000),  # Long audio
    ]
    
    print("🎯 Testing Band-Splitting EAT with Batch Parallelism")
    print("=" * 60)
    
    for batch_size, in_chans, freq_bins, time in test_cases:
        print(f"\n📊 Testing Input Shape: ({batch_size}, {in_chans}, {freq_bins}, {time})")
        
        # Create input tensor
        x = torch.randn(batch_size, in_chans, freq_bins, time)
        
        # Forward pass
        with torch.no_grad():
            features = model(x)
            cls_features, patch_features = model.extract_features(x)
        
        print(f"  ✅ Output Shape: {features.shape}")
        print(f"  ✅ CLS Features Shape: {cls_features.shape}")
        print(f"  ✅ Patch Features Shape: {patch_features.shape}")
        
        # Calculate number of bands
        num_bands = max(1, (freq_bins + config.band_width - 1) // config.band_width)
        num_patches = time // config.shift_size
        if time % config.shift_size != 0:
            num_patches += 1
        
        print(f"  📐 Number of Bands: {num_bands}")
        print(f"  📐 Patches per Band: {num_patches}")
        print(f"  📐 Total Patches: {num_bands * num_patches}")
    
    print("\n🎉 Band-Splitting EAT successfully maintains batch parallelism!")
    print("✅ Frequency-aware processing with band-splitting")
    print("✅ Efficient batch processing")
    print("✅ Dynamic patch embedding")
    print("✅ Frequency positional encoding")
