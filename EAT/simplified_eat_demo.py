"""
Simplified EAT (Efficient Audio Transformer) Implementation
==========================================================

This is a simplified version that demonstrates how EAT predicts masked audio patches.
The code includes detailed annotations explaining each step of the process.

Key Concepts:
- Student-Teacher Architecture with EMA updates
- Masked Audio Patch Prediction
- Feature-level reconstruction (not pixel-level)
- Multi-layer target averaging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import math


class MaskInfo:
    """Container for masking information"""
    def __init__(self, mask, ids_keep, ids_restore):
        self.mask = mask              # Binary mask: 1=masked, 0=unmasked
        self.ids_keep = ids_keep      # Indices of unmasked tokens
        self.ids_restore = ids_restore # Indices to restore original order


class SimplePatchEmbedding(nn.Module):
    """
    Converts audio spectrogram patches to embeddings
    This simulates the CNN encoder in EAT
    """
    def __init__(self, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Simple linear projection (in real EAT, this is a CNN)
        self.proj = nn.Linear(patch_size * patch_size, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input spectrogram (batch_size, height, width)
        Returns:
            patches: Patch embeddings (batch_size, num_patches, embed_dim)
        """
        B, H, W = x.shape
        
        # Create patches (simplified version)
        # In real EAT: patches are created from spectrogram
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        
        # Simulate patch creation (in practice, this would be more complex)
        patches = torch.randn(B, num_patches, self.patch_size * self.patch_size)
        
        # Project to embedding dimension
        patches = self.proj(patches)  # (B, num_patches, embed_dim)
        
        return patches


class SimpleTransformerBlock(nn.Module):
    """
    Simplified Transformer block (single layer)
    In real EAT, this would be a full 12-layer Transformer
    """
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # Expand
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)   # Contract
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, return_layer_output=False):
        """
        Args:
            x: Input embeddings (batch_size, seq_len, embed_dim)
            return_layer_output: Whether to return intermediate output
        Returns:
            x: Output embeddings
            layer_output: Intermediate output (if requested)
        """
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)  # Residual + LayerNorm
        
        # Store intermediate output for teacher targets
        layer_output = x.clone() if return_layer_output else None
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)  # Residual + LayerNorm
        
        return x, layer_output


class SimpleDecoder(nn.Module):
    """
    Simplified decoder that reconstructs masked patches
    In real EAT, this is a 1D convolutional decoder
    """
    def __init__(self, embed_dim=768, output_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
        # Simple MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, output_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input embeddings (batch_size, seq_len, embed_dim)
        Returns:
            predictions: Reconstructed features (batch_size, seq_len, output_dim)
        """
        return self.decoder(x)


class SimpleEAT(nn.Module):
    """
    Simplified EAT model demonstrating masked audio patch prediction
    """
    def __init__(self, embed_dim=768, num_patches=64, mask_ratio=0.8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        
        # Student model components
        self.patch_embed = SimplePatchEmbedding(embed_dim=embed_dim)
        self.student_transformer = SimpleTransformerBlock(embed_dim)
        self.decoder = SimpleDecoder(embed_dim, output_dim=256)
        
        # Teacher model (EMA updated)
        self.teacher_transformer = SimpleTransformerBlock(embed_dim)
        
        # Initialize teacher with student weights
        self.teacher_transformer.load_state_dict(self.student_transformer.state_dict())
        
        # EMA decay rate
        self.ema_decay = 0.999
        
        # CLS token for utterance-level prediction
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
    def create_mask(self, x, mask_ratio):
        """
        Create random mask for input patches
        
        Args:
            x: Input patches (batch_size, num_patches, embed_dim)
            mask_ratio: Fraction of patches to mask
        Returns:
            mask_info: Masking information
        """
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))  # Number of patches to keep
        
        # Generate random noise for each patch
        noise = torch.rand(B, L, device=x.device)
        
        # Sort noise to determine which patches to keep
        ids_shuffle = noise.argsort(dim=1)  # Ascending order
        ids_restore = ids_shuffle.argsort(dim=1)  # Restore original order
        
        # Keep first subset (unmasked patches)
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Create binary mask: 0=keep, 1=mask
        mask = torch.ones(B, L, device=x.device)
        mask[:, :len_keep] = 0
        
        # Unshuffle to get final mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return MaskInfo(mask, ids_keep, ids_restore)
    
    def add_cls_token(self, x):
        """Add CLS token to input patches"""
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls_tokens, x], dim=1)
    
    def remove_cls_token(self, x):
        """Remove CLS token from output"""
        return x[:, 1:, :]  # Remove first token (CLS)
    
    def prepare_decoder_input(self, x, mask_info):
        """
        Prepare input for decoder by adding mask tokens back
        
        Args:
            x: Encoded features (batch_size, num_unmasked + 1, embed_dim)
            mask_info: Masking information
        Returns:
            decoder_input: Input with mask tokens restored (batch_size, num_patches + 1, embed_dim)
        """
        B, L, D = x.shape
        num_masked = mask_info.ids_restore.shape[1] - L + 1  # +1 for CLS token
        
        # Create mask tokens (random noise)
        mask_tokens = torch.randn(B, num_masked, D, device=x.device) * 0.01
        
        # Concatenate unmasked features with mask tokens
        x_with_masks = torch.cat([x[:, 1:], mask_tokens], dim=1)  # Remove CLS, add masks
        
        # Restore original order using ids_restore
        decoder_input = torch.gather(x_with_masks, dim=1, index=mask_info.ids_restore)
        
        # Add CLS token back at the beginning
        decoder_input = torch.cat([x[:, :1], decoder_input], dim=1)
        
        return decoder_input
    
    def update_teacher(self):
        """Update teacher model using EMA"""
        with torch.no_grad():
            for student_param, teacher_param in zip(
                self.student_transformer.parameters(),
                self.teacher_transformer.parameters()
            ):
                # EMA update: teacher = decay * teacher + (1 - decay) * student
                teacher_param.data = (
                    self.ema_decay * teacher_param.data + 
                    (1 - self.ema_decay) * student_param.data
                )
    
    def make_teacher_targets(self, x, mask_info):
        """
        Generate teacher targets by processing unmasked input
        
        Args:
            x: Original patches (batch_size, num_patches, embed_dim)
            mask_info: Masking information
        Returns:
            targets: Teacher targets for masked positions (num_masked, embed_dim)
        """
        with torch.no_grad():
            # Add CLS token
            x_with_cls = self.add_cls_token(x)
            
            # Process through teacher transformer
            teacher_out, teacher_layer_out = self.teacher_transformer(
                x_with_cls, return_layer_output=True
            )
            
            # Remove CLS token
            teacher_out = self.remove_cls_token(teacher_out)
            teacher_layer_out = self.remove_cls_token(teacher_layer_out)
            
            # In real EAT: average multiple layer outputs
            # Here we just use the single layer output
            targets = teacher_layer_out
            
            # Extract only masked positions
            mask = mask_info.mask.bool()
            targets = targets[mask]  # (num_masked_patches, embed_dim)
            
            return targets
    
    def forward(self, spectrogram):
        """
        Forward pass demonstrating EAT's masked prediction process
        
        Args:
            spectrogram: Input audio spectrogram (batch_size, height, width)
        Returns:
            loss: Prediction loss
            predictions: Reconstructed features
            targets: Teacher targets
        """
        # ========================================================================
        # STEP 1: CONVERT SPECTROGRAM TO PATCHES
        # ========================================================================
        print("Step 1: Converting spectrogram to patches...")
        patches = self.patch_embed(spectrogram)  # (B, num_patches, embed_dim)
        print(f"  Patches shape: {patches.shape}")
        
        # ========================================================================
        # STEP 2: CREATE MASK
        # ========================================================================
        print(f"Step 2: Creating mask with ratio {self.mask_ratio}...")
        mask_info = self.create_mask(patches, self.mask_ratio)
        print(f"  Mask shape: {mask_info.mask.shape}")
        print(f"  Masked patches: {mask_info.mask.sum().item()}")
        print(f"  Unmasked patches: {(~mask_info.mask.bool()).sum().item()}")
        
        # ========================================================================
        # STEP 3: STUDENT PROCESSING (MASKED INPUT)
        # ========================================================================
        print("Step 3: Student processing masked input...")
        
        # Add CLS token
        patches_with_cls = self.add_cls_token(patches)
        
        # Remove masked patches for student
        unmasked_patches = patches_with_cls[:, 1:][~mask_info.mask.bool()].view(
            patches_with_cls.shape[0], -1, patches_with_cls.shape[-1]
        )
        student_input = torch.cat([patches_with_cls[:, :1], unmasked_patches], dim=1)
        
        # Process through student transformer
        student_out, _ = self.student_transformer(student_input)
        print(f"  Student output shape: {student_out.shape}")
        
        # ========================================================================
        # STEP 4: DECODER INPUT PREPARATION
        # ========================================================================
        print("Step 4: Preparing decoder input...")
        decoder_input = self.prepare_decoder_input(student_out, mask_info)
        print(f"  Decoder input shape: {decoder_input.shape}")
        
        # ========================================================================
        # STEP 5: DECODER PREDICTION
        # ========================================================================
        print("Step 5: Decoder prediction...")
        predictions = self.decoder(decoder_input)  # (B, num_patches + 1, output_dim)
        predictions = self.remove_cls_token(predictions)  # (B, num_patches, output_dim)
        print(f"  Predictions shape: {predictions.shape}")
        
        # ========================================================================
        # STEP 6: TEACHER TARGET GENERATION
        # ========================================================================
        print("Step 6: Generating teacher targets...")
        targets = self.make_teacher_targets(patches, mask_info)
        print(f"  Teacher targets shape: {targets.shape}")
        
        # ========================================================================
        # STEP 7: LOSS COMPUTATION
        # ========================================================================
        print("Step 7: Computing loss...")
        
        # Extract predictions for masked positions only
        mask = mask_info.mask.bool()
        masked_predictions = predictions[mask]  # (num_masked, output_dim)
        
        # Project targets to same dimension as predictions
        target_proj = nn.Linear(self.embed_dim, predictions.shape[-1]).to(predictions.device)
        targets_projected = target_proj(targets)
        
        # Compute MSE loss
        loss = F.mse_loss(masked_predictions, targets_projected)
        print(f"  Loss: {loss.item():.4f}")
        
        # ========================================================================
        # STEP 8: UPDATE TEACHER (EMA)
        # ========================================================================
        print("Step 8: Updating teacher with EMA...")
        self.update_teacher()
        print("  Teacher updated!")
        
        return loss, masked_predictions, targets_projected


def demonstrate_eat_prediction():
    """
    Demonstrate how EAT predicts masked values
    """
    print("=" * 80)
    print("EAT MASKED PREDICTION DEMONSTRATION")
    print("=" * 80)
    
    # Create model
    model = SimpleEAT(embed_dim=768, num_patches=64, mask_ratio=0.8)
    
    # Create dummy spectrogram (batch_size=2, height=128, width=128)
    batch_size = 2
    spectrogram = torch.randn(batch_size, 128, 128)
    
    print(f"Input spectrogram shape: {spectrogram.shape}")
    print()
    
    # Forward pass
    loss, predictions, targets = model(spectrogram)
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Successfully predicted {predictions.shape[0]} masked patches")
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"✓ Targets shape: {targets.shape}")
    print()
    print("Key Insights:")
    print("1. EAT predicts FEATURE representations, not raw audio values")
    print("2. Teacher model provides rich semantic targets via EMA updates")
    print("3. Only MASKED positions contribute to the loss")
    print("4. CLS token enables utterance-level understanding")
    print("5. Decoder reconstructs all patches, but loss is only on masked ones")


if __name__ == "__main__":
    demonstrate_eat_prediction()
