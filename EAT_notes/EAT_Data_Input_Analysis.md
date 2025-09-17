# EAT Data Input Structure Analysis: 128-Dimensional Mel Spectrograms

This document provides a detailed analysis of how EAT processes 128-dimensional Mel spectrograms as input data.

## Overview

EAT treats audio data as **image modality** by converting audio waveforms into 2D Mel spectrograms and processing them through a vision Transformer architecture. The key insight is that audio spectrograms can be treated as 2D images for self-supervised learning.

## 1. Data Input Pipeline

### Raw Audio → Mel Spectrogram Conversion

```python
# From EAT/data/raw_audio_dataset.py lines 390-420
def __getitem__(self, index):
    # 1. Load audio file
    wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")
    
    # 2. Resample to 16kHz if needed
    if self.downsr_16hz:
        feats = torchaudio.functional.resample(feats, orig_freq=curr_sample_rate, new_freq=16000)
        curr_sample_rate = 16000
    
    # 3. Convert waveform to 128-dimensional Mel spectrogram
    if self.wav2fbank:
        feats = feats.unsqueeze(dim=0)
        feats = torchaudio.compliance.kaldi.fbank(
            feats, 
            htk_compat=True, 
            sample_frequency=curr_sample_rate, 
            use_energy=False,
            window_type='hanning', 
            num_mel_bins=128,  # 128 Mel frequency bins
            dither=0.0, 
            frame_shift=10      # 10ms frame shift
        ).unsqueeze(dim=0)
```

### Key Parameters for Mel Spectrogram Generation

```python
# Mel spectrogram configuration
num_mel_bins: 128        # 128 frequency bins
frame_shift: 10          # 10ms frame shift (100Hz frame rate)
window_type: 'hanning'   # Hanning window
sample_frequency: 16000  # 16kHz sample rate
use_energy: False        # No energy features
dither: 0.0             # No dithering
```

## 2. Data Structure and Dimensions

### Input Spectrogram Shape
```
Input Shape: (1, T, 128)
- 1: Batch dimension (single sample)
- T: Time frames (variable length, padded to target_length)
- 128: Mel frequency bins
```

### Padding and Length Standardization

```python
# From EAT/data/raw_audio_dataset.py lines 420-430
# Padding to target_length (1024 frames)
n_frames = feats.shape[1]
diff = self.target_length - n_frames
if diff > 0:
    m = torch.nn.ZeroPad2d((0, 0, 0, diff))  # Pad time dimension
    feats = m(feats)
elif diff < 0:
    feats = feats[:,0:self.target_length,:]  # Truncate if too long
```

### Final Input Shape
```
Standardized Shape: (1, 1024, 128)
- 1024: Fixed time frames (10.24 seconds at 100Hz frame rate)
- 128: Mel frequency bins
```

## 3. Normalization

### Global Normalization by Dataset

```python
# From EAT/data/raw_audio_dataset.py lines 432-445
# Global normalization for AudioSet
self.norm_mean = -4.268 
self.norm_std = 4.569

# Global normalization for ESC-50
if self.esc50_eval:
    self.norm_mean = -6.627
    self.norm_std = 5.359

# Global normalization for Speech Commands V2
if self.spcv2_eval:
    self.norm_mean = -6.846
    self.norm_std = 5.565

# Apply normalization
feats = (feats - self.norm_mean) / (self.norm_std * 2)
```

## 4. Image Treatment and Patching

### Spectrogram as 2D Image

```python
# From EAT/models/images.py lines 65-75
if modality_cfg.in_chans == 1:  
    img_size = (modality_cfg.target_length, 128)  # (1024, 128)
else:
    img_size = to_2tuple(modality_cfg.input_size)

patch_size = to_2tuple(modality_cfg.patch_size)  # (16, 16)
num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # 512 patches
self.H = img_size[0] // patch_size[0]  # 64 time patches
self.W = img_size[1] // patch_size[1]  # 8 frequency patches
```

### Patch Embedding Structure

```
Spectrogram: (1, 1024, 128)
↓
2D Image: (1, 1024, 128)  # Treated as 2D image
↓
Patches: 64 × 8 = 512 patches  # (16×16) patches
↓
Patch Embeddings: (1, 512, 768)  # 768-dimensional embeddings
```

## 5. Configuration Parameters

### Task Configuration

```yaml
# From EAT/config/pretraining_AS2M.yaml
task:
  audio_mae: true           # Treat audio as image modality
  downsr_16hz: true         # Resample to 16kHz
  target_length: 1024       # Fixed time length
  h5_format: false          # Use raw audio files
```

### Model Configuration

```yaml
# From EAT/config/pretraining_AS2M.yaml
model:
  modalities:
    image:  # Audio treated as image
      in_chans: 1           # Single channel (grayscale)
      patch_size: 16        # 16×16 patches
      embed_dim: 768        # Embedding dimension
```

## 6. Data Augmentation

### Training Augmentations

```python
# From EAT/data/raw_audio_dataset.py lines 445-450
if self.noise and self.train_mode == 'train': 
    # Add Gaussian noise
    feats = feats + torch.rand(feats.shape[1], feats.shape[2]) * np.random.rand() / 10
    # Time shifting
    feats = torch.roll(feats, np.random.randint(-10, 10), 1)

# Roll and magnitude augmentation
if self.roll_mag_aug and self.train_mode == 'train':
    feats = self._roll_mag_aug(feats)
```

## 7. Masking Strategy for Audio

### Inverse Block Multi-Masking

```python
# From EAT/data/mae_image_dataset.py lines 190-210
if self.mask_length == 1:
    mask = compute_block_mask_1d(
        shape=(self.clone_batch, self.patches),  # (16, 512)
        mask_prob=self.mask_prob,               # 0.8 (80% masking)
        mask_length=self.mask_length,           # 5 (block length)
        inverse_mask=self.inverse_mask,         # True (inverse masking)
        require_same_masks=True,
    )
else:
    mask = compute_block_mask_2d(           
        shape=(self.clone_batch, self.patches),
        mask_prob=self.mask_prob,
        mask_length=self.mask_length,
        inverse_mask=self.inverse_mask,
        require_same_masks=True,
        expand_adjcent=self.expand_adjacent,
        mask_dropout=self.mask_dropout,
        non_overlapping=self.non_overlapping,
        img_shape=self.img_shape,               # (64, 8)
        flexible_mask=self.flexible_mask
    )
```

## 8. Data Flow Summary

```
Raw Audio File (.wav)
↓
Load and Resample (16kHz)
↓
Convert to Mel Spectrogram (128 bins, 10ms frame shift)
↓
Pad/Truncate to 1024 frames
↓
Global Normalization (dataset-specific)
↓
Treat as 2D Image (1024 × 128)
↓
Extract 512 Patches (16 × 16 each)
↓
Patch Embeddings (512 × 768)
↓
Apply Inverse Block Masking (80% mask rate)
↓
Input to EAT Model
```

## 9. Key Characteristics

### Temporal Resolution
- **Frame Rate**: 100Hz (10ms frame shift)
- **Duration**: 10.24 seconds (1024 frames)
- **Time Patches**: 64 patches (16 frames each)

### Frequency Resolution
- **Mel Bins**: 128 frequency bins
- **Frequency Patches**: 8 patches (16 bins each)
- **Frequency Range**: 0-8000Hz (Mel scale)

### Spatial Structure
- **Total Patches**: 512 (64 × 8)
- **Patch Size**: 16 × 16 (time × frequency)
- **Embedding Dimension**: 768

## 10. Advantages of This Approach

1. **Unified Architecture**: Audio and vision use the same Transformer backbone
2. **Spatial Relationships**: Captures time-frequency correlations
3. **Efficient Processing**: 2D convolutions and attention mechanisms
4. **Scalable**: Same architecture for different audio tasks
5. **Transfer Learning**: Pre-trained vision models can be adapted

This data input structure allows EAT to effectively treat audio as a 2D spatial signal, enabling the application of vision Transformer techniques to audio representation learning. 