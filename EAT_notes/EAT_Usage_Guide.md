# EAT: Self-Supervised Pre-Training with Efficient Audio Transformer - Complete Usage Guide

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Demo](#quick-demo)
4. [Feature Extraction](#feature-extraction)
5. [Pre-training](#pre-training)
6. [Fine-tuning](#fine-tuning)
7. [Inference and Evaluation](#inference-and-evaluation)
8. [Data Preparation](#data-preparation)
9. [Configuration Files](#configuration-files)
10. [Troubleshooting](#troubleshooting)

## Overview

EAT (Efficient Audio Transformer) is a self-supervised learning model for audio processing that achieves state-of-the-art performance on audio classification tasks while being significantly more efficient than previous approaches. This guide will walk you through all aspects of using the EAT repository.

### Key Features
- **High Efficiency**: ~15x faster pre-training compared to BEATs
- **SOTA Performance**: Top results on AS-20K, AS-2M, ESC-50, and SPC-2 datasets
- **Dual Framework Support**: Both Fairseq and Hugging Face integration
- **Variable Length Support**: Handles different audio durations
- **Multiple Model Sizes**: Base and Large variants available

## Installation

### Prerequisites
- Python >= 3.8
- PyTorch >= 1.13
- pip < 24.1 (due to dependency issues)

### Option 1: For Feature Extraction and Inference Only (Recommended for Beginners)

If you only need to extract features or run inference, you can use the Hugging Face integration without installing Fairseq:

```bash
git clone https://github.com/cwx-worst-one/EAT
cd EAT
pip install -r requirements.txt
```

### Option 2: For Full Training (Pre-training and Fine-tuning)

For complete functionality including pre-training and fine-tuning, you need to install Fairseq:

```bash
# Install Fairseq first
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# Then install EAT
cd ..
git clone https://github.com/cwx-worst-one/EAT
cd EAT
pip install -r requirements.txt
```

### Verify Installation

Test your installation by running a simple feature extraction:

```bash
cd EAT
python feature_extract/feature_extract.py --help
```

## Quick Demo

### 1. Feature Extraction Demo

Extract audio features from a test file:

```bash
# Navigate to EAT directory
cd EAT

# Run feature extraction demo
bash scripts/feature_extract.sh
```

This will:
- Use the EAT-base model from Hugging Face
- Extract features from `feature_extract/test.wav`
- Save features to `feature_extract/test.npy`
- Use "all" granularity (frame-level + CLS token)

### 2. Inference Demo

Run inference on an audio file:

```bash
# Run inference demo
bash scripts/inference.sh
```

This will:
- Load the EAT-base model fine-tuned on AS-2M
- Process `inference/test.wav`
- Output top-12 acoustic event predictions

Expected output:
```
************ Acoustic Event Inference ************
LABEL                          PREDICTION
Percussion                     0.523
Drum kit                       0.437
Vibraphone                     0.420
...
```

## Feature Extraction

### Basic Usage

```bash
python feature_extract/feature_extract.py \
    --source_file='path/to/audio.wav' \
    --target_file='path/to/output.npy' \
    --model_dir='EAT' \
    --checkpoint_dir='worstchan/EAT-base_epoch30_finetune_AS2M' \
    --granularity='all' \
    --target_length=1024 \
    --mode='finetune' \
    --framework='huggingface'
```

### Parameters Explained

- `--source_file`: Input audio file path (.wav format)
- `--target_file`: Output feature file path (.npy format)
- `--checkpoint_dir`: Model checkpoint (Hugging Face repo or local path)
- `--granularity`: Feature type
  - `all`: Frame-level features + CLS token
  - `frame`: Frame-level features only
  - `utterance`: CLS token only (recommended for classification)
- `--target_length`: Mel-spectrogram length (1024 for 10s audio)
- `--mode`: Model type (`pretrain` or `finetune`)
- `--framework`: Framework to use (`huggingface` or `fairseq`)

### Advanced Configuration

For different audio lengths, adjust `target_length`:
- 10-second clips: `target_length=1024`
- 5-second clips: `target_length=512`
- Custom length: Must be multiple of 16 (due to CNN encoder constraints)

## Pre-training

### Prerequisites
- AudioSet data (or your own dataset)
- Fairseq installation
- Sufficient GPU resources (recommended: 4+ GPUs)

### Data Preparation

1. **Download AudioSet**: Get the AudioSet data manifest from the [official link](https://drive.google.com/file/d/1LH2C0q3d4zndoR3-oGkVdYYqDCIdxIsm/view?usp=drive_link)

2. **Organize Data**: Follow the wav2vec/data2vec format:
   ```
   data/
   ├── train.tsv          # Audio file paths
   ├── valid.tsv          # Validation file paths
   ├── train.lbl          # Training labels
   ├── valid.lbl          # Validation labels
   └── dict.ltr.txt       # Label dictionary
   ```

### Running Pre-training

#### Basic Pre-training (EAT-base)

```bash
# Modify the script with your paths
bash scripts/pretraining_AS2M.sh
```

#### Large Model Pre-training

```bash
# Use the large model configuration
python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name pretraining_AS2M_large \
    common.user_dir=EAT \
    checkpoint.save_dir=/path/to/save/checkpoints \
    dataset.batch_size=8 \
    distributed_training.distributed_world_size=4 \
    task.data=/path/to/your/dataset
```

### Configuration Customization

Edit `config/pretraining_AS2M.yaml` to customize:

```yaml
# Model configuration
model:
  depth: 12                    # Number of transformer layers
  embed_dim: 768              # Embedding dimension
  num_heads: 12               # Number of attention heads

# Training configuration
optimization:
  max_update: 100000          # Total training steps
  lr: [0.0001]               # Learning rate

# Data configuration
dataset:
  batch_size: 12             # Batch size per GPU
  max_tokens: 4096           # Maximum tokens per batch
```

### Monitoring Training

Training logs are saved to the checkpoint directory. You can monitor with:

```bash
# View training logs
tail -f /path/to/checkpoints/train.log

# Use TensorBoard (if configured)
tensorboard --logdir=/path/to/checkpoints
```

## Fine-tuning

### Supported Datasets

EAT supports fine-tuning on multiple datasets:
- **AS-20K**: AudioSet 20K subset
- **AS-2M**: AudioSet 2M subset  
- **ESC-50**: Environmental Sound Classification
- **SPCv2**: Speech Commands V2

### Basic Fine-tuning

#### AudioSet Fine-tuning

```bash
# Fine-tune on AS-2M
bash scripts/finetuning_AS2M.sh

# Fine-tune on AS-20K
bash scripts/finetuning_AS20K.sh
```

#### ESC-50 Fine-tuning

```bash
bash scripts/finetuning_ESC50.sh
```

#### Speech Commands V2 Fine-tuning

```bash
bash scripts/finetuning_SPCv2.sh
```

### Fine-tuning Configuration

Key parameters in `config/finetuning.yaml`:

```yaml
model:
  model_path: /path/to/pretrained/checkpoint.pt
  num_classes: 527           # Number of classes (527 for AudioSet)
  mixup: 0.8                # Mixup augmentation
  mask_ratio: 0.2           # SpecAugment masking ratio
  prediction_mode: PredictionMode.CLS_TOKEN

task:
  target_length: 1024       # Audio length
  roll_aug: true           # Roll augmentation
```

### Custom Fine-tuning

For custom datasets, modify the configuration:

```bash
python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name finetuning \
    common.user_dir=EAT \
    checkpoint.save_dir=/path/to/save \
    model.model_path=/path/to/pretrained/model.pt \
    model.num_classes=YOUR_NUM_CLASSES \
    task.data=/path/to/your/dataset \
    task.target_length=1024
```

## Inference and Evaluation

### Single File Inference

```bash
python inference/inference.py \
    --source_file='path/to/audio.wav' \
    --label_file='inference/labels.csv' \
    --model_dir='EAT' \
    --checkpoint_dir='worstchan/EAT-base_epoch30_finetune_AS2M' \
    --target_length=1024 \
    --top_k_prediction=12 \
    --framework='huggingface'
```

### Batch Evaluation

```bash
# Evaluate on full test set
bash scripts/eval.sh
```

This will:
- Load the fine-tuned model
- Evaluate on the test set
- Report mAP scores
- Save per-class AP scores to `ap_log.txt`

### Custom Evaluation

For custom evaluation:

```bash
python evaluation/eval.py \
    --checkpoint_path=/path/to/model.pt \
    --data_path=/path/to/test/data \
    --label_file=/path/to/labels.csv \
    --output_file=results.txt
```

## Data Preparation

### AudioSet Format

The repository expects data in the following format:

```
dataset/
├── train.tsv          # Tab-separated: ID, path, length
├── valid.tsv          # Validation set
├── test.tsv           # Test set
├── train.lbl          # Labels: ID, label_indices
├── valid.lbl          # Validation labels
├── test.lbl           # Test labels
└── dict.ltr.txt       # Label dictionary
```

### Converting Your Data

For custom datasets, create the required files:

1. **Create TSV files**:
   ```bash
   # Example train.tsv format
   echo -e "1\t/path/to/audio1.wav\t10.0" > train.tsv
   echo -e "2\t/path/to/audio2.wav\t8.5" >> train.tsv
   ```

2. **Create label files**:
   ```bash
   # Example train.lbl format (multi-label)
   echo -e "1\t0,5,12" > train.lbl
   echo -e "2\t3,8" >> train.lbl
   ```

3. **Create dictionary**:
   ```bash
   # dict.ltr.txt - one label per line
   echo "class_0" > dict.ltr.txt
   echo "class_1" >> dict.ltr.txt
   ```

### Audio Preprocessing

Ensure your audio files are:
- **Format**: WAV files
- **Sample Rate**: 16kHz (will be resampled if needed)
- **Channels**: Mono (single channel)
- **Duration**: Variable length (will be padded/truncated)

## Configuration Files

### Pre-training Configuration (`config/pretraining_AS2M.yaml`)

Key settings:
```yaml
model:
  depth: 12                    # Transformer depth
  embed_dim: 768              # Embedding dimension
  num_heads: 12               # Attention heads
  drop_path_rate: 0.1         # Drop path rate

optimization:
  max_update: 100000          # Training steps
  lr: [0.0001]               # Learning rate
  warmup_updates: 10000       # Warmup steps

dataset:
  batch_size: 12             # Batch size per GPU
  max_tokens: 4096           # Max tokens per batch
```

### Fine-tuning Configuration (`config/finetuning.yaml`)

Key settings:
```yaml
model:
  model_path: MISSING         # Path to pre-trained model
  num_classes: 1000           # Number of classes
  mixup: 0.0                 # Mixup alpha
  cutmix: 0.0                # CutMix alpha
  label_smoothing: 0.0       # Label smoothing

task:
  target_length: 1024        # Audio length
  roll_aug: false           # Roll augmentation
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   dataset.batch_size=4
   
   # Reduce max tokens
   dataset.max_tokens=2048
   ```

2. **Import Errors**
   ```bash
   # Ensure Fairseq is installed correctly
   pip install --editable ./fairseq
   
   # Check Python path
   export PYTHONPATH=$PYTHONPATH:/path/to/fairseq
   ```

3. **Audio Loading Issues**
   ```bash
   # Install soundfile
   pip install soundfile==0.12.1
   
   # Check audio format
   file your_audio.wav
   ```

4. **Checkpoint Loading Errors**
   ```bash
   # Verify checkpoint path
   ls -la /path/to/checkpoint.pt
   
   # Check model compatibility
   python -c "import torch; print(torch.load('/path/to/checkpoint.pt').keys())"
   ```

### Performance Optimization

1. **Multi-GPU Training**
   ```bash
   # Set number of GPUs
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   distributed_training.distributed_world_size=4
   ```

2. **Mixed Precision**
   ```bash
   # Enable mixed precision
   optimization.fp16=true
   ```

3. **Gradient Accumulation**
   ```bash
   # Accumulate gradients
   optimization.update_freq=[4]
   ```

### Debugging

1. **Enable Debug Logging**
   ```bash
   # Set log level
   export LOG_LEVEL=DEBUG
   ```

2. **Check Data Loading**
   ```bash
   # Test data loading
   python -c "
   from fairseq.tasks import setup_task
   task = setup_task({'data': '/path/to/data'})
   print('Data loaded successfully')
   "
   ```

## Model Checkpoints

### Available Models

| Model | Type | Epochs | Performance | Download |
|-------|------|--------|-------------|----------|
| EAT-base | Pre-trained | 10 | AS-20K: 40.3% | [Google Drive](https://drive.google.com/file/d/10pklbY_fKraQUIBizSg1kv4lJXNWxpxl/view?usp=sharing) |
| EAT-base | Pre-trained | 30 | AS-20K: 41.3% | [Google Drive](https://drive.google.com/file/d/19hfzLgHCkyqTOYmHt8dqVa9nm-weBq4f/view?usp=sharing) |
| EAT-large | Pre-trained | 20 | AS-20K: 42.0% | [Google Drive](https://drive.google.com/file/d/1PEgriRvHsqrtLzlA478VemX7Q0ZGl889/view?usp=sharing) |

### Hugging Face Models

For easier access, use Hugging Face models:
```python
from transformers import AutoModel

# Load pre-trained model
model = AutoModel.from_pretrained("worstchan/EAT-base_epoch30_pretrain")

# Load fine-tuned model
model = AutoModel.from_pretrained("worstchan/EAT-base_epoch30_finetune_AS2M")
```

## Advanced Usage

### Custom Model Architecture

To modify the model architecture, edit the model files in `models/`:

```python
# In models/EAT_audio_classification.py
class MaeImageClassificationModel(BaseFairseqModel):
    def __init__(self, cfg):
        # Modify architecture here
        super().__init__()
        # Your custom modifications
```

### Custom Training Loop

For custom training procedures:

```python
# Create custom training script
import torch
from fairseq import checkpoint_utils

# Load model and data
model, criterion, task = checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])

# Custom training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Your custom training logic
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

### Export to ONNX

To export models for deployment:

```python
import torch

# Load model
model = AutoModel.from_pretrained("worstchan/EAT-base_epoch30_finetune_AS2M")
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 1, 1024, 128)

# Export
torch.onnx.export(model, dummy_input, "eat_model.onnx", 
                 input_names=['input'], output_names=['output'])
```

## Support and Resources

### Documentation
- [Original Paper](https://arxiv.org/abs/2401.03497)
- [Hugging Face Collection](https://huggingface.co/collections/worstchan/eat-6815b4f1034f5214f9063948)
- [GitHub Repository](https://github.com/cwx-worst-one/EAT)

### Community
- Report issues on GitHub
- Check existing issues for solutions
- Join discussions in the repository

### Citation

If you use EAT in your research, please cite:

```bibtex
@inproceedings{ijcai2024p421,
  title     = {EAT: Self-Supervised Pre-Training with Efficient Audio Transformer},
  author    = {Chen, Wenxi and Liang, Yuzhe and Ma, Ziyang and Zheng, Zhisheng and Chen, Xie},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {3807--3815},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/421},
  url       = {https://doi.org/10.24963/ijcai.2024/421},
}
```

---

This guide covers all major aspects of using the EAT repository. For specific questions or advanced usage, refer to the original documentation or create an issue on the GitHub repository. 