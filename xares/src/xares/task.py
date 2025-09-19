from __future__ import annotations

import io
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import webdataset as wds
from loguru import logger
from tqdm import tqdm

from xares.audiowebdataset import create_embedding_webdataset, create_rawaudio_webdataset
from xares.common import XaresSettings
from xares.metrics import METRICS_TYPE
from xares.models.asr import AsrModelForGeneration
from xares.models.retrival import RetrivalMLP
from xares.models.simple import Mlp
from xares.trainer import KNNTrainer, Trainer, prepare_clip_task_batch
from xares.utils import download_hf_model_to_local, download_zenodo_record, mkdir_if_not_exists

# Add binary classification evaluator import
sys.path.append(str(Path(__file__).parent.parent.parent / "binary_classification_task"))
try:
    from binary_classification_evaluator import calculate_metrics
    BINARY_EVALUATOR_AVAILABLE = True
except ImportError:
    logger.warning("Binary classification evaluator not available. DCASE metrics will be skipped.")
    BINARY_EVALUATOR_AVAILABLE = False


@dataclass
class TaskConfig:
    name: str
    formal_name: str = ""
    xares_settings: XaresSettings = field(default_factory=XaresSettings)
    env_root: Path | str | None = None
    disabled: bool = False  # Skip the task, useful for debugging

    # General
    private: bool = False
    torch_num_threads: int = 2  # Do not use too many otherwise slows down
    seed: int = 42  # manual seed for all experiments
    label_processor: Callable | None = None
    merge_processor: Callable | None = None
    task_type: Literal["frame", "clip", "contrastive", "asr"] = "clip"
    eval_weight: int = 0

    # Splits
    train_split: None | str = "train"
    valid_split: None | str = "valid"
    test_split: None | str = "test"
    k_fold_splits: None | List[int | str] = None
    use_mini_dataset: bool = False  # For some large datasets, use subset for faster evaluation

    # Audio tar
    force_download: bool = False
    audio_tar_name_of_split: Dict[Any, Any] = field(default_factory=lambda: dict())
    num_shards_rawaudio = 4
    zenodo_id: str | None = None

    # Encoded tar
    force_encode: bool = False
    encoder: None | Any = None
    pretrained_dependencies: None | List[str] = None
    encoded_tar_name_of_split: Dict[Any, Any] = field(default_factory=lambda: dict())
    trim_length = None
    save_encoded_per_batches: int = 2000
    batch_size_encode: int = 16
    num_encoder_workers: int = 4
    crop_length: None | float = None

    # MLP
    force_retrain_mlp: bool = False
    ckpt_dir_name = "checkpoints"
    embedding_dir_name = "embeddings"
    ckpt_name = "best.ckpt"
    criterion: (
        Literal["CrossEntropyLoss", "BCEWithLogitsLoss", "L1Loss", "MSELoss", "AudioTextContrastiveLoss"]
        | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) = "CrossEntropyLoss"
    batch_size_train: int = 32
    batch_size_valid: int | None = None
    learning_rate: float = 1e-3
    epochs: int = 10
    valid_every: int = 1
    num_training_workers: int = 0
    num_validation_workers: int = 0
    sort_by_length: bool = False
    model: nn.Module | None = None
    output_dim: int | None = None
    metric: METRICS_TYPE = "accuracy"
    metric_args: Dict[str, Any] = field(default_factory=lambda: dict())

    # KNN
    do_knn: bool = True

    def __post_init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.formal_name == "":
            self.formal_name = self.name
        if self.batch_size_valid is None:
            self.batch_size_valid = self.batch_size_train

        self.update_tar_name_of_split()
        if self.env_root is None:
            self.env_root = self.xares_settings.env_root
        torch.set_num_threads(self.torch_num_threads)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def update_tar_name_of_split(self):
        if self.k_fold_splits is not None:
            self.audio_tar_name_of_split = {fold: f"wds-audio-fold-{fold}-*.tar" for fold in self.k_fold_splits}
            self.encoded_tar_name_of_split = {fold: f"wds-encoded-fold-{fold}-*.tar" for fold in self.k_fold_splits}
        else:
            self.audio_tar_name_of_split = {
                self.train_split: f"{self.train_split}*.tar",
                self.valid_split: f"{self.valid_split}*.tar",
                self.test_split: f"{self.test_split}*.tar",
            }
            self.encoded_tar_name_of_split = {
                self.train_split: f"wds-encoded-{self.train_split}*.tar",
                self.valid_split: f"wds-encoded-{self.valid_split}*.tar",
                self.test_split: f"wds-encoded-{self.test_split}*.tar",
            }


class XaresTask:
    def __init__(self, config: TaskConfig):
        self.config = config

        if self.config.encoder is not None:
            self.encoder_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.encoder = self.config.encoder.to(self.encoder_device)
            self.encoder_name = self.encoder.__class__.__name__
        else:
            self.encoder = None
            self.encoder_name = "Unknown"

        torch.set_num_threads(self.config.torch_num_threads)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.env_dir = Path(self.config.env_root) / self.config.name
        self.ckpt_dir = self.env_dir / self.config.ckpt_dir_name / self.encoder_name
        self.encoded_tar_dir = self.env_dir / self.config.embedding_dir_name / self.encoder_name
        self.encoded_ready_path = self.encoded_tar_dir / self.config.xares_settings.encoded_ready_filename
        self.ckpt_path = self.ckpt_dir / self.config.ckpt_name
        mkdir_if_not_exists(self.encoded_tar_dir)

        if self.config.k_fold_splits:
            self.config.train_split = self.config.valid_split = self.config.test_split = None

        self.label_processor = self.config.label_processor
        self.merge_processor = self.config.merge_processor

        if self.config.task_type == "contrastive":
            self.mlp_model = RetrivalMLP
        elif self.config.task_type == "asr":
            self.mlp_model = AsrModelForGeneration
        else:
            self.mlp_model = Mlp

        self.encoded_tar_path_of_split = {
            split: (self.encoded_tar_dir / self.config.encoded_tar_name_of_split[split])
            for split in self.config.encoded_tar_name_of_split
        }

        if self.config.pretrained_dependencies is not None:
            download_hf_model_to_local(self.config.pretrained_dependencies)

    def download_audio_tar(self):
        if self.config.private:
            logger.warning(f"Dataset {self.config.name} is private. Do not download from Zenodo.")
            return

        audio_ready_path = self.env_dir / self.config.xares_settings.audio_ready_filename
        if not self.config.force_download and audio_ready_path.exists():
            logger.warning(f"Skip downloading audio tar: {audio_ready_path} exists.")
            return

        download_zenodo_record(self.config.zenodo_id, self.env_dir, force_download=self.config.force_download)
        audio_ready_path.touch()

    def make_encoded_tar(self):
        if not self.config.force_encode and self.encoded_ready_path.exists():
            logger.warning(f"Skip encoding: {self.encoded_ready_path} exists.")
            return

        audio_ready_path = self.env_dir / self.config.xares_settings.audio_ready_filename
        if not audio_ready_path.exists():
            if self.config.private:
                logger.warning(f"For private dataset {self.config.name}, data must be placed at local path. Skip.")
                return
            download_zenodo_record(self.config.zenodo_id, self.env_dir, force_download=self.config.force_download)
            audio_ready_path.touch()

        audio_tar_path_of_split = {
            split: (self.env_dir / self.config.audio_tar_name_of_split[split]).as_posix()
            for split in self.config.audio_tar_name_of_split
        }

        for split in audio_tar_path_of_split:
            logger.info(f"Encoding audio for split {split} ...")
            logger.debug(f"Using data from {audio_tar_path_of_split[split]} ... ")
            dl = create_rawaudio_webdataset(
                [audio_tar_path_of_split[split]],
                target_sample_rate=self.encoder.sampling_rate,
                audio_key_name="audio",
                num_workers=(
                    0
                    if (mp.current_process().daemon or mp.get_start_method() != "fork")
                    else self.config.num_encoder_workers
                ),
                batch_size=self.config.batch_size_encode,
                crop_length=self.config.crop_length,
                pad_last=True,  # Add crop
            )
            sink = wds.ShardWriter(
                self.encoded_tar_path_of_split[split].as_posix().replace("*", "0%05d"),
                encoder=False,
                compress=False,
                maxcount=self.config.save_encoded_per_batches,
                verbose=False,
            )

            with torch.inference_mode():
                for enum_item, ((audio, _), json_data, filenames) in tqdm(
                    enumerate(dl), desc=f"Encoding {split}", leave=True
                ):
                    audio = audio.to(self.encoder_device)
                    embedding = self.encoder(audio).to("cpu").detach()
                    for embed, json_data_sample, filename in zip(embedding, json_data, filenames):
                        buf = io.BytesIO()
                        np.save(buf, embed.numpy())
                        sink.write(
                            {
                                "npy": buf.getvalue(),
                                "json": json.dumps(json_data_sample).encode("utf-8"),
                                "__key__": f"{filename}{enum_item}",
                            }
                        )

        self.encoded_ready_path.touch()

    def run_mlp(self) -> Tuple[float, int]:
        mlp_score = 0
        eval_size = 0
        score_file = self.ckpt_dir / "mlp_score.txt"

        if score_file.exists():
            with open(score_file, "r") as f:
                lines = f.read().splitlines()
                mlp_score = float(lines[0])
                eval_size = int(lines[1])
            logger.info(f"Loaded MLP score from {score_file}: ({mlp_score}, {eval_size})")
            return mlp_score, eval_size

        if self.config.k_fold_splits:
            # K-fold cross validation
            acc = []
            eval_sizes = []
            splits = self._make_splits()
            wds_encoded_training_fold_k = {
                k: [self.encoded_tar_path_of_split[j].as_posix() for j in splits if j != k] for k in splits
            }

            for k in splits:
                self.config.ckpt_name = f"fold_{k}_best_model.pt"
                self.ckpt_path = self.ckpt_dir / self.config.ckpt_name
                self.train_mlp(
                    wds_encoded_training_fold_k[k],
                    [self.encoded_tar_path_of_split[k].as_posix()],
                )
                score, eval_size = self.evaluate_mlp([self.encoded_tar_path_of_split[k].as_posix()], load_ckpt=True)
                acc.append(score)
                eval_sizes.append(eval_size)

            for k in range(len(splits)):
                logger.info(f"Fold {k + 1} {self.config.metric}: {acc[k]}")

            avg_score = np.mean(acc)
            total_eval_size = int(np.average(eval_sizes))
            logger.info(f"The averaged {self.config.metric} of 5 folds is: {avg_score}")

            # Save individual fold results for MLP
            mlp_fold_scores_file = self.ckpt_dir / "mlp_fold_scores.txt"
            with open(mlp_fold_scores_file, "w") as f:
                f.write(f"# MLP Individual Fold Scores\n")
                f.write(f"# Format: fold_index,score,eval_size\n")
                for i, (fold_idx, score, eval_size) in enumerate(zip(splits, acc, eval_sizes)):
                    f.write(f"{fold_idx},{score},{eval_size}\n")
                f.write(f"# Average: {avg_score},{total_eval_size}\n")
            logger.info(f"Saved individual MLP fold scores to {mlp_fold_scores_file}")

            mlp_score = avg_score
            eval_size = total_eval_size
        else:
            # Single split
            self.train_mlp(
                [self.encoded_tar_path_of_split[self.config.train_split].as_posix()],
                [self.encoded_tar_path_of_split[self.config.valid_split].as_posix()],
            )
            mlp_score, eval_size = self.evaluate_mlp(
                [self.encoded_tar_path_of_split[self.config.test_split].as_posix()],
                load_ckpt=True,
            )
            logger.info(f"The {self.config.metric}: {mlp_score}")

        with open(score_file, "w") as f:
            f.write(f"{mlp_score}\n{eval_size}")
        logger.info(f"Saved MLP score to {score_file}: ({mlp_score}, {eval_size})")

        return mlp_score, eval_size

    def train_mlp(self, train_url: list, validation_url: list) -> None:
        mlp = self.mlp_model(
            in_features=self.encoder.output_dim, out_features=self.config.output_dim, criterion=self.config.criterion
        )

        self.trainer = Trainer(
            mlp,
            ckpt_dir=self.ckpt_dir,
            ckpt_name=self.config.ckpt_name,
            metric=self.config.metric,
            metric_args=self.config.metric_args,
            lr=self.config.learning_rate,
            max_epochs=self.config.epochs,
            task_type=self.config.task_type,
            task_name=self.config.name,
            output_dir=self.config.env_root,
            valid_every=self.config.valid_every,
        )

        if not self.config.force_retrain_mlp and self.ckpt_path.exists():
            logger.info(f"Checkpoint {self.ckpt_path} already exists. Skip training.")
            self.trainer.load_state_dict(torch.load(self.ckpt_path))
            return

        dl_train = create_embedding_webdataset(
            train_url,
            tar_shuffle=2000,
            batch_size=self.config.batch_size_train,
            num_workers=self.config.num_training_workers,
            sort_by_length=self.config.sort_by_length,
            training=True,
            label_processor=self.label_processor,
            merge_processor=self.merge_processor,
        )
        dl_val = create_embedding_webdataset(
            validation_url,
            batch_size=self.config.batch_size_valid,
            num_workers=self.config.num_validation_workers,
            sort_by_length=self.config.sort_by_length,
            training=False,
            label_processor=self.label_processor,
            merge_processor=self.merge_processor,
        )

        try:
            self.trainer.run(dl_train, dl_val)
        except RuntimeError as e:
            if "at least one example" in str(e):
                raise RuntimeError(f"Empty dataloader. Try delete {self.encoded_ready_path} and re-run.")
            else:
                raise e

    def evaluate_mlp(self, eval_url: list, load_ckpt: bool = False) -> Tuple[Dict[METRICS_TYPE, Any], int]:
        if self.trainer is None:
            raise ValueError("Train the model first before evaluation.")

        if load_ckpt:
            if self.ckpt_path.exists():
                self.trainer.load_state_dict(torch.load(self.ckpt_path))
                logger.info(f"Loaded model parameters from {self.ckpt_path}")
            else:
                logger.warning(f"No checkpoint found at {self.ckpt_path}. Skip loading.")

        dl = create_embedding_webdataset(
            eval_url,
            batch_size=self.config.batch_size_valid,
            num_workers=self.config.num_validation_workers,
            label_processor=self.label_processor,
            merge_processor=self.merge_processor,
            training=False,
        )
        
        # Run inference and collect continuous scores for DCASE in one pass
        score = self._run_mlp_inference_with_dcase_collection(dl, eval_url)
        
        return score, self.config.eval_weight

    def _run_mlp_inference_with_dcase_collection(self, dl, eval_url: list):
        """Run MLP inference and collect DCASE scores in one pass"""
        # First run the standard inference to get the metric score
        score = self.trainer.run_inference(dl)
        
        # Then collect DCASE scores if available and applicable
        if BINARY_EVALUATOR_AVAILABLE and self.config.output_dim == 2:
            try:
                logger.info("Collecting MLP scores for DCASE evaluation...")
                
                all_logits = []
                all_labels = []
                all_filenames = []
                
                # Run inference again to collect logits (this is still inefficient, but better than the previous approach)
                # TODO: Modify trainer to collect logits during first pass
                self.trainer.model.eval()
                with torch.inference_mode():
                    for batch in dl:
                        # Extract filenames BEFORE prepare_batch_function discards them
                        filenames = batch[2] if len(batch) > 2 else [f"sample_{i}" for i in range(len(batch[0][0]))]
                        
                        # Get predictions
                        y_pred, y_true = self.trainer.model(*self.trainer.prepare_batch_function(batch, self.trainer.device))
                        
                        # Extract logits (for binary classification, take the positive class logit)
                        if y_pred.shape[1] == 2:  # Binary classification
                            logits = y_pred[:, 1].cpu().numpy()  # Positive class logits
                        else:
                            logits = y_pred.cpu().numpy().flatten()
                        
                        labels = y_true.cpu().numpy()
                        
                        all_logits.extend(logits)
                        all_labels.extend(labels)
                        all_filenames.extend(filenames)
                
                if len(all_logits) > 0:
                    # Convert to numpy arrays
                    y_true = np.array(all_labels)
                    y_pred_logits = np.array(all_logits)
                    
                    # Save continuous scores
                    self.save_continuous_scores(y_true, y_pred_logits, all_filenames, "mlp")
                    
                    # Calculate and save DCASE metrics
                    dcase_metrics = self.evaluate_dcase_metrics(y_true, y_pred_logits, "mlp")
                    
                    # Save DCASE metrics to CSV
                    if dcase_metrics:
                        self._save_dcase_metrics_csv(dcase_metrics, "mlp")
                        
            except Exception as e:
                logger.error(f"Error collecting MLP scores for DCASE: {e}")
        
        return score

    def _save_dcase_metrics_csv(self, metrics: Dict[str, float], method_name: str):
        """Save DCASE metrics to CSV file"""
        try:
            results_dir = self.ckpt_dir / "dcase_results"
            results_dir.mkdir(exist_ok=True)
            
            # Create metrics DataFrame
            metrics_df = pd.DataFrame([metrics])
            metrics_file = results_dir / f"{method_name}_dcase_metrics.csv"
            metrics_df.to_csv(metrics_file, index=False)
            
            logger.info(f"Saved DCASE metrics to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving DCASE metrics: {e}")

    def run_knn(self) -> Tuple[float, int]:
        knn_score = 0
        eval_size = 0
        if not self.config.do_knn:
            logger.warning(f"Skip KNN evaluation for {self.config.name}.")
            return knn_score, eval_size

        if not os.path.exists(self.config.ckpt_dir):
            os.makedirs(self.config.ckpt_dir, exist_ok=True)

        score_file = self.ckpt_dir / "knn_score.txt"

        if score_file.exists():
            with open(score_file, "r") as f:
                lines = f.read().splitlines()
                knn_score = float(lines[0])
                eval_size = int(lines[1])
            logger.info(f"Loaded KNN score from {score_file}: ({knn_score}, {eval_size})")
            return knn_score, eval_size

        if self.config.k_fold_splits:
            # K-fold cross validation
            scores = []
            eval_sizes = []
            splits = self._make_splits()
            wds_encoded_training_fold_k = {
                k: [self.encoded_tar_path_of_split[j].as_posix() for j in splits if j != k] for k in splits
            }

            for k in splits:
                score, size = self.train_and_eval_knn(
                    wds_encoded_training_fold_k[k],
                    [self.encoded_tar_path_of_split[k].as_posix()],
                )
                scores.append(score)
                eval_sizes.append(size)

            for k in range(len(splits)):
                logger.info(f"Fold {k + 1} {self.config.metric}: {scores[k]}")

            avg_score = np.mean(scores)
            total_eval_size = int(np.average(eval_sizes))
            logger.info(f"The averaged KNN {self.config.metric} of 5 folds is: {avg_score}")

            # Save individual fold results for KNN
            knn_fold_scores_file = self.ckpt_dir / "knn_fold_scores.txt"
            with open(knn_fold_scores_file, "w") as f:
                f.write(f"# KNN Individual Fold Scores\n")
                f.write(f"# Format: fold_index,score,eval_size\n")
                for i, (fold_idx, score, eval_size) in enumerate(zip(splits, scores, eval_sizes)):
                    f.write(f"{fold_idx},{score},{eval_size}\n")
                f.write(f"# Average: {avg_score},{total_eval_size}\n")
            logger.info(f"Saved individual KNN fold scores to {knn_fold_scores_file}")

            knn_score = avg_score
            eval_size = total_eval_size
        else:
            # Single split
            score, size = self.train_and_eval_knn(
                [self.encoded_tar_path_of_split[self.config.train_split].as_posix()],
                [self.encoded_tar_path_of_split[self.config.test_split].as_posix()],
            )
            logger.info(f"The KNN score: {score}")
            knn_score = score
            eval_size = size

        with open(score_file, "w") as f:
            f.write(f"{knn_score}\n{eval_size}")
        logger.info(f"Saved KNN score to {score_file}: ({knn_score}, {eval_size})")

        return knn_score, eval_size

    def train_and_eval_knn(self, train_url, eval_url) -> Tuple[float, int]:
        dl_train = create_embedding_webdataset(
            train_url,
            tar_shuffle=2000,
            batch_size=self.config.batch_size_train,
            num_workers=self.config.num_training_workers,
            training=True,
            label_processor=self.label_processor,
        )
        dl_eval = create_embedding_webdataset(
            eval_url,
            batch_size=self.config.batch_size_train,
            num_workers=self.config.num_validation_workers,
            training=False,
            label_processor=self.label_processor,
        )
        knn_trainer = KNNTrainer(num_classes=self.config.output_dim, task_name=self.config.name, output_dir=self.config.env_root)
        
        # Run KNN evaluation and collect continuous scores for DCASE in one pass
        score = self._run_knn_eval_with_dcase_collection(dl_train, dl_eval, eval_url)
        
        return score, self.config.eval_weight

    def _run_knn_eval_with_dcase_collection(self, dl_train, dl_eval, eval_url: list):
        """Run KNN evaluation and collect DCASE scores in one pass"""
        knn_trainer = KNNTrainer(num_classes=self.config.output_dim, task_name=self.config.name, output_dir=self.config.env_root)
        
        # Run standard KNN evaluation
        score = knn_trainer.train_and_eval(dl_train, dl_eval)
        
        # Collect DCASE scores if available and applicable
        if BINARY_EVALUATOR_AVAILABLE and self.config.output_dim == 2:
            try:
                logger.info("Collecting KNN scores for DCASE evaluation...")
                
                # Extract training features and labels (reuse from knn_trainer)
                train_data, train_labels = zip(*knn_trainer.trainer.state.output)
                train_data, train_labels = torch.cat(train_data, 0), torch.cat(train_labels, 0).long()
                
                # Create KNN model
                from xares.trainer import KNN
                knn_model = KNN(train_data, train_labels, nb_knn=knn_trainer.nb_knn, 
                               num_classes=knn_trainer.num_classes, T=knn_trainer.temperature)
                
                all_logits = []
                all_labels = []
                all_filenames = []
                
                # Evaluate on test set
                knn_model.eval()
                with torch.inference_mode():
                    for batch in dl_eval:
                        # Extract filenames BEFORE prepare_clip_task_batch discards them
                        filenames = batch[2] if len(batch) > 2 else [f"sample_{i}" for i in range(len(batch[0][0]))]
                        
                        x, y = prepare_clip_task_batch(batch)
                        y_pred = knn_model(x)
                        
                        # Extract logits (for binary classification, take the positive class logit)
                        if y_pred.shape[1] == 2:  # Binary classification
                            logits = y_pred[:, 1].cpu().numpy()  # Positive class logits
                        else:
                            logits = y_pred.cpu().numpy().flatten()
                        
                        labels = y.cpu().numpy()
                        
                        all_logits.extend(logits)
                        all_labels.extend(labels)
                        all_filenames.extend(filenames)
                
                if len(all_logits) > 0:
                    # Convert to numpy arrays
                    y_true = np.array(all_labels)
                    y_pred_logits = np.array(all_logits)
                    
                    # Save continuous scores
                    self.save_continuous_scores(y_true, y_pred_logits, all_filenames, "knn")
                    
                    # Calculate and save DCASE metrics
                    dcase_metrics = self.evaluate_dcase_metrics(y_true, y_pred_logits, "knn")
                    
                    # Save DCASE metrics to CSV
                    if dcase_metrics:
                        self._save_dcase_metrics_csv(dcase_metrics, "knn")
                        
            except Exception as e:
                logger.error(f"Error collecting KNN scores for DCASE: {e}")
        
        return score

    def run(self):
        self.download_audio_tar()
        self.make_encoded_tar()
        mlp_score = self.run_mlp()
        knn_score = self.run_knn()
        return mlp_score, knn_score

    def _make_splits(self):
        splits = self.config.k_fold_splits or [self.config.train_split, self.config.valid_split, self.config.test_split]
        return list(filter(lambda x: x is not None, splits))

    def evaluate_dcase_metrics(self, y_true: np.ndarray, y_pred_logits: np.ndarray, method_name: str = "mlp") -> Dict[str, float]:
        """
        Evaluate DCASE-compatible metrics for binary classification.
        
        Args:
            y_true: Ground truth labels (0=normal, 1=anomaly)
            y_pred_logits: Raw logits from model (higher = more anomalous)
            method_name: Name of the method (mlp/knn) for file naming
            
        Returns:
            Dictionary of DCASE metrics
        """
        if not BINARY_EVALUATOR_AVAILABLE:
            logger.warning("Binary classification evaluator not available. Skipping DCASE metrics.")
            return {}
        
        try:
            # Convert logits to probabilities using sigmoid
            y_pred_probs = 1 / (1 + np.exp(-y_pred_logits))
            
            # For binary decisions, use 0.5 threshold
            y_decision = (y_pred_probs > 0.5).astype(int)
            
            # Calculate DCASE metrics
            metrics = calculate_metrics(y_true, y_pred_probs, y_decision, max_fpr=0.1)
            
            logger.info(f"DCASE metrics for {method_name}:")
            logger.info(f"  AUC: {metrics.get('auc', 0):.4f}")
            logger.info(f"  pAUC: {metrics.get('pauc', 0):.4f}")
            logger.info(f"  Official Score: {metrics.get('official_score', 0):.4f}")
            logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
            logger.info(f"  Recall: {metrics.get('recall', 0):.4f}")
            logger.info(f"  F1: {metrics.get('f1', 0):.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating DCASE metrics: {e}")
            return {}

    def save_continuous_scores(self, y_true: np.ndarray, y_pred_logits: np.ndarray, filenames: List[str], method_name: str = "mlp"):
        """
        Save continuous scores and ground truth to CSV files.
        
        Args:
            y_true: Ground truth labels
            y_pred_logits: Raw logits from model
            filenames: List of filenames corresponding to each sample
            method_name: Name of the method (mlp/knn) for file naming
        """
        try:
            # Create results directory
            results_dir = self.ckpt_dir / "dcase_results"
            results_dir.mkdir(exist_ok=True)
            
            # Convert logits to probabilities
            y_pred_probs = 1 / (1 + np.exp(-y_pred_logits))
            
            # Save ground truth
            gt_df = pd.DataFrame({
                'filename': filenames,
                'label': y_true
            })
            gt_file = results_dir / f"{method_name}_ground_truth.csv"
            gt_df.to_csv(gt_file, index=False)
            
            # Save continuous scores
            pred_df = pd.DataFrame({
                'filename': filenames,
                'score': y_pred_probs
            })
            pred_file = results_dir / f"{method_name}_predictions.csv"
            pred_df.to_csv(pred_file, index=False)
            
            # Save binary decisions
            y_decision = (y_pred_probs > 0.5).astype(int)
            decision_df = pd.DataFrame({
                'filename': filenames,
                'decision': y_decision
            })
            decision_file = results_dir / f"{method_name}_decisions.csv"
            decision_df.to_csv(decision_file, index=False)
            
            logger.info(f"Saved continuous scores to {results_dir}")
            logger.info(f"  Ground truth: {gt_file}")
            logger.info(f"  Predictions: {pred_file}")
            logger.info(f"  Decisions: {decision_file}")
            
        except Exception as e:
            logger.error(f"Error saving continuous scores: {e}")
