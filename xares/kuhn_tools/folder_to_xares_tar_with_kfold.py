import argparse
import random
import soundfile as sf
from pathlib import Path
import pandas as pd
from loguru import logger
from xares.audiowebdataset import write_audio_tar
from xares.common import XaresSettings
from xares.utils import mkdir_if_not_exists

def make_audio_tar(
    dataset_root: str,
    env_root: None | str = None,
    force_generate_audio_tar=True,
    num_shards=4,
    seed=42,
    k_folds: int | None = None,
    stratify: bool = True,
):
    settings = XaresSettings()
    target_dir = Path(env_root)
    audio_tar_ready_file_path = target_dir / settings.audio_ready_filename
    if not force_generate_audio_tar and audio_tar_ready_file_path.exists():
        logger.info(f"Audio tar ready file path {audio_tar_ready_file_path} already exists.")
        return

    # Read data list
    data_list = []
    dataset_path = Path(dataset_root)

    # Process normal files (label 0)
    normal_dir = dataset_path / "normal"
    if normal_dir.exists():
        for wav_file in normal_dir.glob("*.wav"):
            try:
                info = sf.info(str(wav_file))
                data_list.append({
                    "wav_path": str(wav_file),
                    "label": 0,
                    "reserve": 0,
                    "audio_length": info.duration,
                    "start_time": 0.0
                })
            except Exception as e:
                logger.warning(f"Error reading {wav_file}: {e}")

    # Process abnormal files (label 1)
    abnormal_dir = dataset_path / "abnormal"
    if abnormal_dir.exists():
        for wav_file in abnormal_dir.glob("*.wav"):
            try:
                info = sf.info(str(wav_file))
                data_list.append({
                    "wav_path": str(wav_file),
                    "label": 1,
                    "reserve": 0,
                    "audio_length": info.duration,
                    "start_time": 0.0
                })
            except Exception as e:
                logger.warning(f"Error reading {wav_file}: {e}")

    if not data_list:
        logger.error("No valid audio files found!")
        return

    # K-fold or simple train/valid/test split
    random.seed(seed)  # For reproducibility
    if k_folds and k_folds > 1:
        # Build folds (optionally stratified by label)
        logger.info(f"Creating {k_folds}-fold tars (stratify={stratify})...")

        # Group by label for stratification
        if stratify:
            by_label = {}
            for item in data_list:
                by_label.setdefault(item["label"], []).append(item)
            for label_items in by_label.values():
                random.shuffle(label_items)

            # Partition each label group into k chunks, then merge per-fold
            folds = {fold_idx: [] for fold_idx in range(1, k_folds + 1)}
            for label, label_items in by_label.items():
                n = len(label_items)
                # Compute split sizes as evenly as possible
                base = n // k_folds
                remainder = n % k_folds
                start = 0
                for i in range(k_folds):
                    size = base + (1 if i < remainder else 0)
                    end = start + size
                    if size > 0:
                        folds[i + 1].extend(label_items[start:end])
                    start = end
        else:
            # Non-stratified: shuffle all and split into contiguous chunks
            random.shuffle(data_list)
            n = len(data_list)
            base = n // k_folds
            remainder = n % k_folds
            folds = {}
            start = 0
            for i in range(k_folds):
                size = base + (1 if i < remainder else 0)
                end = start + size
                folds[i + 1] = data_list[start:end] if size > 0 else []
                start = end

        # Write a tar per fold using xares default naming expected by k-fold tasks
        for fold_idx, split_data in folds.items():
            if not split_data:
                logger.warning(f"No data for fold {fold_idx}")
                continue

            tar_path = target_dir / f"wds-audio-fold-{fold_idx}-*.tar"

            audio_paths = [item["wav_path"] for item in split_data]
            labels = [{"label": item["label"]} for item in split_data]

            write_audio_tar(
                audio_paths=audio_paths,
                labels=labels,
                tar_path=tar_path.as_posix(),
                force=force_generate_audio_tar,
                num_shards=num_shards,
            )
    else:
        # Default: simple 80/10/10 split after shuffle
        random.shuffle(data_list)
        total_samples = len(data_list)
        test_size = int(total_samples * 0.1)

        split_data_dic = {
            "test": data_list[:test_size],
            "valid": data_list[test_size:2 * test_size],
            "train": data_list[2 * test_size:],
        }

        for split_name, split_data in split_data_dic.items():
            if not split_data:
                logger.warning(f"No data for {split_name} split")
                continue

            tar_path = target_dir / f"wds-audio-split-{split_name}-*.tar"

            audio_paths = [item["wav_path"] for item in split_data]
            labels = [{"label": item["label"]} for item in split_data]

            write_audio_tar(
                audio_paths=audio_paths,
                labels=labels,
                tar_path=tar_path.as_posix(),
                force=force_generate_audio_tar,
                num_shards=num_shards,
            )

    # Mark as ready
    audio_tar_ready_file_path.touch()
    logger.info("Audio tar creation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory containing 'normal' and 'abnormal' subdirectories.")
    parser.add_argument("--env_root", type=str, default=None, help="Root directory for the environment.")
    parser.add_argument("--force_generate_audio_tar", action="store_true", help="Force generation of the audio tar files.")
    parser.add_argument("--num_shards", type=int, default=4, help="Number of shards.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible shuffling.")
    parser.add_argument("--k_folds", type=int, default=None, help="If set (>1), generate K fold tars named wds-audio-fold-<k>-*.tar (1-based).")
    parser.add_argument("--no_stratify", action="store_true", help="Disable stratified folding by label.")
    args = parser.parse_args()

    make_audio_tar(
        dataset_root=args.dataset_root,
        env_root=args.env_root,
        force_generate_audio_tar=args.force_generate_audio_tar,
        num_shards=args.num_shards,
        seed=args.seed,
        k_folds=args.k_folds,
        stratify=(not args.no_stratify),
    )