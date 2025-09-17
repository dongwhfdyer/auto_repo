# Image Masking Implementation in EAT

This document provides a detailed analysis of how images (audio spectrograms) are masked in the Efficient Audio Transformer (EAT) model.

## Overview

EAT treats audio spectrograms as 2D images and applies **inverse block multi-masking** strategy. The masking process involves:
1. Converting audio waveforms to mel-spectrograms
2. Applying 2D block masking with inverse strategy
3. Using high masking rates (80%) for efficiency
4. Implementing multi-mask strategy for better data utilization

## 1. Audio to Image Conversion

### Waveform to Spectrogram Conversion

```python
# From EAT/data/raw_audio_dataset.py lines 398-430
# convert waveform to spectrogram
if self.wav2fbank:
    feats = feats.unsqueeze(dim=0)
    feats = torchaudio.compliance.kaldi.fbank(
        feats, 
        htk_compat=True, 
        sample_frequency=curr_sample_rate, 
        use_energy=False,
        window_type='hanning', 
        num_mel_bins=128,  # 128 mel frequency bins
        dither=0.0, 
        frame_shift=10      # 10ms frame shift
    ).unsqueeze(dim=0)
    
    # padding to target_length (1024 frames)
    n_frames = feats.shape[1]
    diff = self.target_length - n_frames
    if diff > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, diff)) 
        feats = m(feats)
    elif diff < 0:
        feats = feats[:,0:self.target_length,:]
    
    # Global normalization
    feats = (feats - self.norm_mean) / (self.norm_std * 2)
```

**Result**: Audio spectrogram of shape `(1, 1024, 128)` where:
- 1024 = time frames (10.24 seconds at 10ms frame shift)
- 128 = mel frequency bins

### Spectrogram to Patches

```python
# From EAT/models/images.py lines 195-210
def patchify(self, imgs):
    if self.modality_cfg.in_chans == 1:  # Audio spectrograms
        p = self.modality_cfg.patch_size  # patch_size = 16
        h = imgs.shape[2] // p  # 1024 // 16 = 64
        w = imgs.shape[3] // p  # 128 // 16 = 8
        
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))  # (B, 512, 256)
        
    return x
```

**Result**: Spectrogram divided into 512 patches (64×8) of size 16×16 each.

## 2. 2D Block Masking Implementation

### Main Masking Function

```python
# From EAT/utils/data_utils.py lines 211-223
def compute_block_mask_2d(      
    shape: Tuple[int, int],
    mask_prob: float,           # 0.8 (80% masking rate)
    mask_length: int,           # 5 (5×5 block size)
    mask_prob_adjust: float = 0,
    inverse_mask: bool = False, # True (inverse masking)
    require_same_masks: bool = True,
    expand_adjcent: bool = False,
    mask_dropout: float = 0,
    non_overlapping: bool = False,
    img_shape: tuple = None,    # (64, 8) for audio spectrograms
    flexible_mask: bool = False,
) -> torch.Tensor:
```

### Inverse Block Masking Algorithm

```python
# From EAT/utils/data_utils.py lines 224-350
def compute_block_mask_2d(...):
    B, L = shape  # B=batch_size, L=512 (total patches)
    d = (64, 8)   # img_shape for audio spectrograms
    
    if inverse_mask:
        mask_prob = 1 - mask_prob  # 1 - 0.8 = 0.2 (keep 20%)
    
    # 1. Initialize mask tensor
    mask = torch.zeros((B, d[0], d[1]))  # (B, 64, 8)
    
    # 2. Randomly select center points for blocks
    mask_inds = torch.randint(
        0, L,  # Random indices from 0 to 511
        size=(
            B,
            int(L * ((mask_prob + mask_prob_adjust) / mask_length**2) * (1 + mask_dropout))
        ),
    )
    mask.view(B, -1).scatter_(1, mask_inds, 1)  # Set center points to 1
    centers = mask.nonzero(as_tuple=True)
    
    # 3. Expand blocks around center points
    inds = ([], [], [])
    offset = mask_length // 2  # 2
    
    for i in range(mask_length):      # 0 to 4
        for j in range(mask_length):  # 0 to 4
            k1 = i - offset  # -2 to 2
            k2 = j - offset  # -2 to 2
            inds[0].append(centers[0])  # batch dimension
            inds[1].append(centers[1] + k1)  # height dimension
            inds[2].append(centers[2] + k2)  # width dimension
    
    # 4. Apply block masks
    i0 = torch.cat(inds[0])
    i1 = torch.cat(inds[1]).clamp_(min=0, max=d[0] - 1)  # Clamp to valid range
    i2 = torch.cat(inds[2]).clamp_(min=0, max=d[1] - 1)
    mask[(i0, i1, i2)] = 1  # Set entire blocks to 1
    
    # 5. Invert mask for inverse masking
    if inverse_mask:
        mask = 1 - mask  # 0 becomes 1, 1 becomes 0
    
    return mask.reshape(B, -1)  # Flatten to (B, 512)
```

## 3. Masking Configuration

### Configuration Parameters

```yaml
# From EAT/config/pretraining_AS2M.yaml
model:
  modalities:
    image:  # Audio treated as image modality
      inverse_mask: true      # Inverse block masking
      mask_prob: 0.8         # 80% masking rate
      mask_prob_adjust: 0.07 # Mask probability adjustment
      mask_length: 5         # 5×5 block size
      mask_noise_std: 0.01   # Mask noise standard deviation
      mask_dropout: 0.0      # No mask dropout
```

### Key Parameters Explained

1. **`inverse_mask: true`**: 
   - Normal masking: mask 80% of patches
   - Inverse masking: keep 20% of patches (mask 80%)
   - This means the student encoder only sees 20% of patches

2. **`mask_prob: 0.8`**: 
   - 80% of patches are masked
   - Only 20% of patches are visible to the student

3. **`mask_length: 5`**: 
   - Creates 5×5 block masks
   - Each mask covers 25 patches (5×5)

4. **`mask_prob_adjust: 0.07`**: 
   - Adjusts the number of mask blocks
   - Final probability = 0.8 + 0.07 = 0.87

## 4. Multi-Mask Strategy

### Clone Batch Implementation

```python
# From EAT/config/pretraining_AS2M.yaml
clone_batch: 16  # Create 16 different masks per sample

# From EAT/models/EAT_pretraining.py lines 610-615
# multiply the target value according to the number of clone batch
if self.cfg.clone_batch > 1:
    y = y.repeat_interleave(self.cfg.clone_batch, 0)
```

**Effect**: Each audio sample generates 16 different masked versions, increasing data utilization efficiency.

## 5. Mask Application

### How Masks Are Applied

```python
# From EAT/models/base.py lines 451-470
def apply_mask(self, x, mask_info):
    cfg = self.modality_cfg
    B, T, C = x.shape  # (B, 512, 768)

    if mask_info is not None:
        mask = mask_info.mask  # (B, 512) binary mask
        if cfg.encoder_zero_mask:
            # Zero out masked patches
            x = x * (1 - mask.type_as(x).unsqueeze(-1))
        else:
            # Replace masked patches with noise
            num_masks = mask.sum().item()
            masks = x.new_empty(num_masks, x.size(-1)).normal_(0, cfg.mask_noise_std)
            x = index_put(x, mask, masks)
    
    return x
```

### Mask Information Structure

```python
# From EAT/models/base.py lines 421-450
def make_maskinfo(self, x, mask, shape=None):
    if shape is None:
        B, T, D = x.shape
    else:
        B, T, D = shape

    mask = mask.to(torch.uint8)
    ids_shuffle = mask.argsort(dim=1)  # Sort by mask values
    ids_restore = ids_shuffle.argsort(dim=1).unsqueeze(-1).expand(-1, -1, D)

    len_keep = T - mask[0].sum()  # Number of unmasked patches
    ids_keep = ids_shuffle[:, :len_keep]  # Indices of unmasked patches

    mask_info = MaskInfo(
        x_unmasked=x_unmasked,  # Unmasked patch features
        mask=mask,              # Binary mask
        ids_restore=ids_restore, # Restoration indices
        ids_keep=ids_keep,      # Keep indices
    )
    return mask_info
```

## 6. Visual Representation

### Audio Spectrogram Structure
```
Time (1024 frames) → 64 patches
     ↓
Freq (128 bins) → 8 patches
     ↓
Total: 64 × 8 = 512 patches
```

### Block Masking Pattern
```
Original:  [P P P P P P P P]  (512 patches)
           [P P P P P P P P]
           [P P P P P P P P]
           ...

Masked:    [P M M M M P M M]  (M = masked, P = visible)
           [M M M M M M M M]
           [P M M M M P M M]
           ...

Where M represents 5×5 block masks
```

## 7. Key Features of EAT's Masking

### 1. **Inverse Block Masking**
- Keeps only 20% of patches visible
- Masks are 5×5 blocks (25 patches each)
- Creates structured, contiguous masked regions

### 2. **High Masking Rate (80%)**
- Significantly reduces computational cost
- Forces model to learn from limited information
- Enables faster training

### 3. **Multi-Mask Strategy**
- 16 different masks per sample
- Increases data utilization
- Improves learning efficiency

### 4. **2D Structure Preservation**
- Respects spectrogram's 2D structure
- Block masks maintain spatial relationships
- More realistic than random patch masking

### 5. **Efficient Processing**
- Student encoder only processes unmasked patches
- Teacher encoder processes full spectrogram
- Asymmetric architecture for efficiency

## 8. Comparison with Other Methods

| Method | Masking Rate | Mask Type | Structure |
|--------|-------------|-----------|-----------|
| **EAT** | 80% | 5×5 blocks | 2D structured |
| MAE | 75% | Random patches | 1D random |
| SimMIM | 60% | Random patches | 1D random |
| Data2Vec | 15% | Random tokens | 1D random |

## 9. Benefits of EAT's Masking Strategy

1. **Computational Efficiency**: 80% masking reduces computation by ~5x
2. **Structured Learning**: Block masks preserve spatial relationships
3. **High Challenge**: Limited visible information forces better representations
4. **Multi-Mask Efficiency**: 16 masks per sample improves data utilization
5. **Realistic Patterns**: Block masks simulate real-world occlusions

This masking strategy is a key component of EAT's efficiency and performance, enabling the model to learn robust audio representations while maintaining fast training speeds. 