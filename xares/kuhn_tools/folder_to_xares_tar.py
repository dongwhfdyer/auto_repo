import argparse
import random
import soundfile as sf
from pathlib import Path
import pandas as pd
from loguru import logger
from xares.audiowebdataset import write_audio_tar
from xares.common import XaresSettings
from xares.utils import mkdir_if_not_exists

def make_audio_tar(dataset_root: str, env_root: None | str = None, force_generate_audio_tar=True, num_shards=4, seed=42):
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

    # Create random splits
    random.seed(seed)  # For reproducibility
    random.shuffle(data_list)  # Shuffle the data list randomly

    total_samples = len(data_list)
    test_size = int(total_samples * 0.1)

    # Split data randomly
    split_data_dic = {
        "test": data_list[:test_size],
        "valid": data_list[test_size:2 * test_size],
        "train": data_list[2 * test_size:],
    }

    # Create tar files for each split
    for split_name, split_data in split_data_dic.items():
        if not split_data:
            logger.warning(f"No data for {split_name} split")
            continue

        tar_path = target_dir / f"wds-audio-split-{split_name}-*.tar"

        # Extract lists for write_audio_tar
        audio_paths = [item["wav_path"] for item in split_data]
        # Convert integer labels to dictionaries as required by write_audio_tar
        labels = [{"label": item["label"]} for item in split_data]

        # Write tar file (only with supported parameters)
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
    args = parser.parse_args()

    make_audio_tar(
        dataset_root=args.dataset_root,
        env_root=args.env_root,
        force_generate_audio_tar=args.force_generate_audio_tar,
        num_shards=args.num_shards,
        seed=args.seed,
    )