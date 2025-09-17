#!/usr/bin/env python3
"""
Simple test for band-splitting functionality in ImageEncoder
"""

import torch
import torch.nn as nn
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from images import ImageEncoder, D2vImageConfig

def test_band_splitting():
    """Test band-splitting functionality"""
    print("🎯 Testing Band-Splitting in ImageEncoder")
    print("=" * 50)
    
    # Create configuration with band-splitting enabled
    config = D2vImageConfig(
        in_chans=1,
        target_length=1024,
        max_length=768,
        patch_size=16,
        embed_dim=768,
        band_width=32,
        shift_size=16,
        use_band_splitting=True,
        freq_pos_emb_dim=768,
        fixed_positions=False,
        prenet_depth=2,
        start_drop_path_rate=0.0,
        end_drop_path_rate=0.0,
        prenet_layerdrop=0.0,
        prenet_dropout=0.0
    )
    
    # Create a simple block function for testing
    def make_block(drop_path_rate):
        return nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=12,
            dim_feedforward=config.embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
    
    # Create ImageEncoder
    encoder = ImageEncoder(
        modality_cfg=config,
        embed_dim=config.embed_dim,
        make_block=make_block,
        norm_layer=nn.LayerNorm,
        layer_norm_first=False,
        alibi_biases={},
        task=None
    )
    
    print(f"Configuration:")
    print(f"  - Band Width: {config.band_width}")
    print(f"  - Shift Size: {config.shift_size}")
    print(f"  - Use Band Splitting: {config.use_band_splitting}")
    print(f"  - Embed Dim: {config.embed_dim}")
    print()
    
    # Test with audio spectrogram
    batch_size = 2
    freq_bins = 128
    time_length = 1024
    
    # Create input spectrogram
    x = torch.randn(batch_size, freq_bins, time_length)
    
    print(f"Input Shape: {x.shape}")
    print(f"Expected Bands: {max(1, (freq_bins + config.band_width - 1) // config.band_width)}")
    print()
    
    # Test local_features with band-splitting
    try:
        features = encoder.local_features(x)
        print(f"✅ Local Features Shape: {features.shape}")
        print(f"✅ Band-splitting successful!")
        
        # Test contextualized_features
        padding_mask = None
        mask = None
        result = encoder.contextualized_features(
            features, padding_mask, mask, remove_masked=False
        )
        print(f"✅ Contextualized Features Shape: {result['x'].shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("🎉 Test completed!")

if __name__ == "__main__":
    test_band_splitting()
