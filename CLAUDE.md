# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an audio machine learning research repository containing multiple audio encoder frameworks, evaluation tools, and research implementations. The main components are:

- **X-ARES**: eXtensive Audio Representation and Evaluation Suite - a comprehensive benchmark for audio encoders
- **EAT**: Efficient Audio Transformer - self-supervised pre-training model with Fairseq integration
- **DASHENG**: Deep Audio-Signal Holistic Embeddings - scalable masked audio encoder
- **CED**: Audio classification encoder implementation
- **Binary Classification Task**: Evaluation tools for binary audio classification
- **DCASE Repositories**: Various DCASE (Detection and Classification of Acoustic Scenes and Events) challenge implementations

## Development Environment Setup

### Core Dependencies
The repository uses Python 3.8+ with PyTorch-based implementations. Main package managers:
- `pip install -e .` for local development packages
- Individual component installations vary (see component-specific instructions)

### Key Python Packages
- PyTorch >= 1.13 (some components require >= 2.1)
- torchaudio >= 0.13
- transformers >= 4.47.1
- webdataset >= 0.2.100
- scikit-learn >= 1.6.0
- pytorch-ignite
- loguru for logging

## Common Development Commands

### X-ARES Benchmark
```bash
# Install X-ARES
cd xares && pip install -e .[examples]

# Run benchmark with baseline encoder (8 parallel jobs)
python -m xares.run --max-jobs 8 example/dasheng/dasheng_encoder.py src/tasks/*.py

# Check encoder implementation
python -c "from xares.audio_encoder_checker import check_audio_encoder; check_audio_encoder(your_encoder)"

# Run single task
python -c "from example.dasheng.dasheng_encoder import DashengEncoder; from tasks.asvspoof_task import asvspoof2015_config; from xares.task import XaresTask; task = XaresTask(config=asvspoof2015_config(encoder=DashengEncoder())); task.run()"
```

### EAT (Efficient Audio Transformer)
```bash
# Feature extraction only (no Fairseq needed)
cd EAT && pip install -r requirements.txt
bash scripts/feature_extract.sh

# For pre-training/fine-tuning (requires Fairseq)
bash scripts/pretraining_AS2M.sh
bash scripts/finetuning_AS20K.sh
bash scripts/inference.sh
bash scripts/eval.sh
```

### DASHENG
```bash
# Install for inference only
pip install dasheng

# Install for training
pip install dasheng[train]

# Basic usage
python -c "from dasheng import dasheng_base; import torch; model = dasheng_base().eval(); features=model(torch.randn(1, 16000))"

# Training
accelerate launch --mixed_precision='bf16' dasheng/train/train.py dasheng/train/config/dasheng_base.yaml
```

### Binary Classification Evaluation
```bash
# Run binary classification metrics
cd binary_classification_task
python binary_classification_evaluator.py --gt_file ground_truth.csv --pred_file predictions.csv
```

## Architecture Overview

### X-ARES Framework Structure
- **`xares/src/xares/`**: Core evaluation framework
  - `task.py`: Main task execution engine with support for MLP and kNN evaluation methods
  - `trainer.py`: Training utilities for downstream tasks
  - `metrics.py`: Comprehensive audio classification metrics
  - `audiowebdataset.py`: WebDataset integration for audio data
  - `models/`: Model implementations (MLP, retrieval, ASR)
- **`xares/src/tasks/`**: 25+ audio task implementations covering speech, environment, and music domains
- **`xares/example/`**: Encoder wrapper examples (Dasheng, Data2Vec, Wav2Vec2, Whisper)
- **`xares/tools/`**: Dataset preparation utilities

### Key Evaluation Methods
1. **MLP (Linear Fine-Tuning)**: Trains linear layer on fixed representations with task-specific hyperparameters
2. **kNN (Unparameterized)**: Direct k-nearest neighbor classification without training to test representational quality

### Audio Encoder Integration
All audio encoders must implement:
- `sampling_rate`: Target sampling rate (typically 16000)
- `output_dim`: Embedding dimension
- `hop_size_in_ms`: Temporal resolution
- `forward(audio: torch.Tensor)`: Returns embeddings for variable-length audio (up to 10 minutes)

### Cross-Component Integration
- X-ARES integrates binary classification evaluator from `binary_classification_task/`
- EAT provides both Fairseq and HuggingFace interfaces
- DASHENG models can be used as X-ARES encoders via wrapper classes

## Data Formats

### Audio Data
- **WebDataset Format**: Primary format for large-scale training (`.tar` files with audio samples)
- **TSV Format**: Simple file lists following wav2vec/data2vec conventions
- **Standard Formats**: Support for WAV, FLAC, MP3 audio files

### Task Configuration
Tasks use dataclass-based configuration with:
- Dataset paths and splits
- Evaluation metrics specification
- Preprocessing parameters
- Model-specific hyperparameters

## Testing and Evaluation

### Automated Dataset Download
X-ARES automatically downloads evaluation datasets from Zenodo. Manual download available via `tools/download_manually.sh`.

### Supported Audio Tasks
**Speech**: ASV2015, CREMA-D, LibriSpeech variants, VoxCeleb1, Speech Commands, etc.
**Environment**: ESC-50, UrbanSound8k, FSD50k, DESED, etc.
**Music**: GTZAN, MAESTRO, NSynth, FMA, etc.

### Evaluation Metrics
- Classification accuracy, F1-score, AUP, mAP
- DCASE-specific metrics via binary classification evaluator
- Cross-validation support with k-fold evaluation

## Important Implementation Notes

- **Variable-Length Audio**: All components support variable-length audio up to 10 minutes
- **Chunking Strategy**: Implement audio chunking in encoders to handle memory constraints
- **Multiprocessing**: X-ARES supports parallel task execution with `--max-jobs` parameter
- **GPU Memory**: Monitor memory usage during evaluation, especially with large models like EAT-large (309M params)
- **WebDataset Integration**: Use webdataset format for efficient I/O with large audio datasets

## Documentation and Resources

- X-ARES: [arXiv:2505.16369](https://arxiv.org/abs/2505.16369)
- EAT: [arXiv:2401.03497](https://arxiv.org/abs/2401.03497)
- DASHENG: [arXiv:2406.06992](https://arxiv.org/abs/2406.06992)
- HEAR Benchmark: [hearbenchmark.com](https://hearbenchmark.com/)