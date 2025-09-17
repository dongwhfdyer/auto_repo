# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import sys
import argparse

from typing import Optional, List
from dataclasses import dataclass, field
from omegaconf import MISSING, II

from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

try:
    from ..data import MultiAudioDataset
except:
    sys.path.append("..")
    from data import MultiAudioDataset

logger = logging.getLogger(__name__)


@dataclass
class ImageMaskingConfig:
    patch_size: int = II("model.modalities.image.patch_size")
    mask_prob: float = II("model.modalities.image.mask_prob")
    mask_prob_adjust: float = II("model.modalities.image.mask_prob_adjust")
    mask_length: int = II("model.modalities.image.mask_length")
    inverse_mask: bool = II("model.modalities.image.inverse_mask")
    mask_dropout: float = II("model.modalities.image.mask_dropout")
    clone_batch: int = II("model.clone_batch")
    expand_adjacent: bool = False
    non_overlapping: bool = False


@dataclass
class MultiAudioPretrainingConfig(FairseqDataclass):
    # Data paths
    old_data: str = field(default=MISSING, metadata={"help": "path to old data directory"})
    new_data: str = field(default=MISSING, metadata={"help": "path to new data directory"})
    
    # Dataset parameters
    input_size: int = 224
    key: str = "imgs"
    rebuild_batches: bool = True
    precompute_mask_config: Optional[ImageMaskingConfig] = None
    subsample: float = 1
    seed: int = II("common.seed")
    dataset_type: str = "imagefolder"
    
    # Audio-specific parameters
    audio_mae: bool = field(default=False, metadata={"help": "if set, we use image_mae way to deal with audio files."})
    h5_format: bool = field(default=False, metadata={"help": "if set, dataset will read data file in h5df format."})
    downsr_16hz: bool = field(default=False, metadata={"help": "if set, wav file's sample rate will be reduced to 16kHz."})
    target_length: int = field(default=1024, metadata={"help": "This setting will pad the audio spectrogram with zeros."})
    flexible_mask: bool = field(default=False, metadata={"help": "if true, we will using flexible inverse block mask method."})
    # Augmentations
    roll_aug: bool = field(default=False, metadata={"help": "if true, use roll augmentation during pre-training."})
    noise: bool = field(default=False, metadata={"help": "if true, add noise augmentation during pre-training."})


@register_task("multi_audio_pretraining", dataclass=MultiAudioPretrainingConfig)
class MultiAudioPretrainingTask(FairseqTask):
    """
    Multi-data audio pre-training task that mixes old and new datasets
    with epoch-based ratio progression.
    """

    cfg: MultiAudioPretrainingConfig

    @classmethod
    def setup_task(cls, cfg: MultiAudioPretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (MultiAudioPretrainingConfig): configuration of this task
        """
        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        """Load dataset for the given split."""
        cfg = task_cfg or self.cfg
        
        # Validate data paths
        if not cfg.old_data or not cfg.new_data:
            raise ValueError("Both old_data and new_data paths must be provided")
        
        logger.info(f"Loading multi-audio dataset for split: {split}")
        logger.info(f"Old data path: {cfg.old_data}")
        logger.info(f"New data path: {cfg.new_data}")
        
        # Prepare dataset arguments
        compute_mask = cfg.precompute_mask_config is not None
        mask_args = {}
        if compute_mask:
            mask_args = cfg.precompute_mask_config
        
        # Create multi-audio dataset
        self.datasets[split] = MultiAudioDataset(
            old_data_path=cfg.old_data,
            new_data_path=cfg.new_data,
            split=split,
            input_size=cfg.input_size,
            key=cfg.key,
            compute_mask=compute_mask,
            dataset_type=cfg.dataset_type,
            audio_mae=cfg.audio_mae,
            downsr_16hz=cfg.downsr_16hz,
            h5_format=cfg.h5_format,
            target_length=cfg.target_length,
            flexible_mask=cfg.flexible_mask,
            roll_aug=cfg.roll_aug,
            noise=cfg.noise,
            **mask_args,
        )
        
        logger.info(f"Multi-audio dataset loaded successfully for split: {split}")

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize

    def get_sampling_stats(self, split: str = "train") -> dict:
        """Get current sampling statistics for monitoring."""
        if split in self.datasets:
            return self.datasets[split].get_sampling_stats()
        return {}


def add_multi_audio_args(parser: argparse.ArgumentParser):
    """Add multi-audio specific arguments to the parser."""
    group = parser.add_argument_group("Multi-Audio Pre-training")
    group.add_argument(
        "--old-data",
        type=str,
        required=True,
        help="Path to old data directory"
    )
    group.add_argument(
        "--new-data", 
        type=str,
        required=True,
        help="Path to new data directory"
    )
    return parser 