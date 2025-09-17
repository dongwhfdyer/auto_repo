# EAT_pretraining_with_annotation.py
# Heavily annotated version of EAT pretraining model with tensor shapes and deep explanations

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from dataclasses import dataclass, field
from typing import Optional, Callable
from functools import partial
from omegaconf import II
from enum import Enum, auto
from fairseq.modules import EMAModule, EMAModuleConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model

from .base import (
    MaskSeed,
    D2vModalityConfig,
 UtilizationSpecificEncoder, 
    get_annealed_rate,
)

from .modules import (
    D2vDecoderConfig,
    AltBlock,
    Decoder1d,
)

from .images import (
    D2vImageConfig,
    ImageEncoder,
)

logger = logging.getLogger(__name__)

# =============================================================================
# MODALITY ENUM DEFINITION
# =============================================================================
# EAT supports multiple modalities: AUDIO, IMAGE, TEXT
# This follows the work of data2vec 2.0 on image modality and Audio-MAE in EAT
class Modality(Enum):
    AUDIO = auto()    # Audio spectrograms treated as images
    IMAGE = auto()    # Standard image data
    TEXT = auto()     # Text data (not used in current implementation)

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================
@dataclass
class D2vModalitiesConfig(FairseqDataclass):
    """
    Configuration for different modalities in Data2Vec multi-modal model.
    Currently only supports IMAGE modality (which handles audio spectrograms).
    """
    image: D2vImageConfig = D2vImageConfig()

@dataclass
class Data2VecMultiConfig(FairseqDataclass):
    """
    Main configuration class for Data2Vec multi-modal model.
    Contains all hyperparameters for the EAT pretraining model.
    """
    
    # ========================================================================
    # LOSS CONFIGURATION
    # ========================================================================
    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    # Deep Meaning: Controls the smoothness of L1 loss vs L2 loss
    # 0 = Pure L2 loss (MSE), >0 = Smooth L1 loss (Huber loss)
    # Smooth L1 is more robust to outliers than L2, less sensitive than L1
    
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )
    # Deep Meaning: Normalizes loss by feature dimension to prevent scale issues
    # None = auto-scale by 1/sqrt(embed_dim), preventing loss from being too large/small

    # ========================================================================
    # TRANSFORMER ARCHITECTURE CONFIGURATION
    # ========================================================================
    depth: int = 12  # Number of transformer layers
    
    # Vision Transformer standard parameters
    start_drop_path_rate: float = 0      # Starting drop path rate for stochastic depth
    end_drop_path_rate: float = 0        # Ending drop path rate for stochastic depth
    num_heads: int = 12                  # Number of attention heads
    norm_eps: float = 1e-6              # Layer normalization epsilon
    norm_affine: bool = True            # Whether to use affine transformation in LayerNorm
    encoder_dropout: float = 0.1        # Dropout rate in encoder
    post_mlp_drop: float = 0.1          # Dropout after MLP layers
    attention_dropout: float = 0.1       # Dropout in attention layers
    activation_dropout: float = 0.0      # Dropout in activation functions
    dropout_input: float = 0.0          # Dropout on input embeddings
    layerdrop: float = 0.0              # Probability of dropping entire layers
    embed_dim: int = 768                # Hidden dimension size
    mlp_ratio: float = 4                # MLP hidden size = embed_dim * mlp_ratio
    layer_norm_first: bool = False      # Whether to apply LayerNorm before attention/MLP

    # ========================================================================
    # EAT-SPECIFIC CONFIGURATION
    # ========================================================================
    # EAT averages all Transformer block output (12 layers in total) 
    average_top_k_layers: int = field(
        default=12, metadata={"help": "how many layers to average"}
    )
    # Deep Meaning: EAT uses ALL 12 transformer layers' output as teacher targets
    # This is different from typical approaches that use only the last layer
    # Makes the target more stable and informative by averaging multiple representations

    end_of_block_targets: bool = False  # Whether to use end-of-block outputs as targets

    # clone batch for multi-mask strategy
    clone_batch: int = 16
    # Deep Meaning: Creates multiple different masks for the same input
    # This increases training efficiency by learning from multiple masked versions
    # Shape impact: batch_size becomes batch_size * clone_batch

    # ========================================================================
    # TEACHER MODEL NORMALIZATION CONFIGURATION
    # ========================================================================
    # Normalization for teacher Transformer layer output
    layer_norm_target_layer: bool = False    # Apply LayerNorm to each layer output
    batch_norm_target_layer: bool = False    # Apply BatchNorm to each layer output
    instance_norm_target_layer: bool = False # Apply InstanceNorm to each layer output
    instance_norm_targets: bool = False      # Apply InstanceNorm to final targets
    layer_norm_targets: bool = False         # Apply LayerNorm to final targets

    # ========================================================================
    # EMA (EXPONENTIAL MOVING AVERAGE) CONFIGURATION
    # ========================================================================
    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    # Deep Meaning: Controls how fast the teacher model updates
    # 0.999 means teacher updates 99.9% from previous state, 0.1% from student
    # High values make teacher more stable, low values make teacher more responsive
    
    ema_same_dtype: bool = True              # Whether EMA uses same dtype as model
    log_norms: bool = True                   # Whether to log parameter norms
    
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )
    # Deep Meaning: Final EMA decay rate after annealing
    # Even higher than initial decay, making teacher very stable at end of training
    
    ema_anneal_end_step: int = II("optimization.max_update")  # When to stop annealing

    # In EAT, the Transformer encoder and the CNN encoder are both EMA updated
    ema_encoder_only: bool = field(
        default=True,
        metadata={
            "help": "whether to momentum update only the shared transformer encoder"
        },
    )
    # Deep Meaning: If True, only transformer blocks are EMA updated
    # If False, entire model (including CNN encoders) is EMA updated
    # EAT uses False to update both CNN and transformer with EMA

    max_update: int = II("optimization.max_update")  # Maximum training updates

    # ========================================================================
    # MODALITY AND DECODER CONFIGURATION
    # ========================================================================
    modalities: D2vModalitiesConfig = D2vModalitiesConfig()  # Modality configurations
    shared_decoder: Optional[D2vDecoderConfig] = None        # Shared decoder config

    # ========================================================================
    # TRAINING STABILITY CONFIGURATION
    # ========================================================================
    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )
    # Deep Meaning: Prevents collapse when targets/predictions become too similar
    # If variance drops below threshold, training stops to prevent mode collapse

    supported_modality: Optional[Modality] = None  # Which modality to support
    mae_init: bool = False                         # Whether to use MAE initialization
    seed: int = II("common.seed")                 # Random seed
    skip_ema: bool = False                        # Whether to skip EMA (for teacher model)

    # ========================================================================
    # LOSS WEIGHT CONFIGURATION
    # ========================================================================
    # d2v_loss is the frame-level loss while cls_loss is the utterance-level loss
    cls_loss: float = 0      # Weight for utterance-level classification loss
    recon_loss: float = 0    # Weight for reconstruction loss
    d2v_loss: float = 1      # Weight for Data2Vec regression loss (main loss)

    decoder_group: bool = False  # Whether to group decoder parameters separately

    # ========================================================================
    # DINo LOSS EXPERIMENTAL CONFIGURATION
    # ========================================================================
    # The experiment of using dino loss instead of direct utterance loss
    utterance_level: bool = field(
        default=False, 
        metadata={"help": "if true, we will add utterance-level loss to the total loss"}
    )
    init_center_token_zero: bool = field(
        default=False, 
        metadata={"help": "if true, we will initialize the center token with zero vectors"}
    )
    center_exp: float = field(
        default=0.9, 
        metadata={"help": "this value control the exponent decay of center value's coefficient"}
    )
    softmax_temperature_student: float = field(
        default=0.1, 
        metadata={"help": "temperature of softmax function of student output in the dino loss"}
    )
    softmax_temperature_teacher: float = field(
        default=0.05, 
        metadata={"help": "temperature of softmax function in teacher output the dino loss"}
    )

# =============================================================================
# MODEL REGISTRATION AND MAIN CLASS
# =============================================================================
@register_model("data2vec_multi", dataclass=Data2VecMultiConfig)
class Data2VecMultiModel(BaseFairseqModel):
    """
    Main EAT pretraining model class.
    
    This is a self-supervised learning model that learns representations by:
    1. Masking parts of audio spectrograms
    2. Training student transformer to predict teacher transformer outputs
    3. Using EMA to update teacher model
    4. Multiple loss functions: frame-level, utterance-level, reconstruction
    """
    
    # ========================================================================
    # MODALITY ENCODER CREATION
    # ========================================================================
    def make_modality_encoder(
        self,
        cfg: D2vModalityConfig,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases,
        task,
    ) -> ModalitySpecificEncoder:
        """
        Creates the appropriate encoder for the given modality.
        
        Args:
            cfg: Modality-specific configuration
            embed_dim: Embedding dimension (768)
            make_block: Function to create transformer blocks
            norm_layer: Layer normalization function
            layer_norm_first: Whether to apply LayerNorm first
            alibi_biases: ALiBi bias tensors (not used in EAT)
            task: Training task object
            
        Returns:
            ModalitySpecificEncoder: The created encoder
        
        Deep Meaning: Factory method that creates the right encoder based on modality type.
        Currently only supports IMAGE modality which handles audio spectrograms.
        """
        if cfg.type.value == Modality.IMAGE.value:
            enc_cls = ImageEncoder
        else:
            raise Exception(f"unsupported modality {cfg.type}")

        return enc_cls(
            cfg,
            embed_dim,
            make_block,
            norm_layer,
            layer_norm_first,
            alibi_biases,
            task,
        )

    # ========================================================================
    # MODEL INITIALIZATION
    # ========================================================================
    def __init__(self, cfg: Data2VecMultiConfig, modalities, skip_ema=False, task=None):
        """
        Initialize the Data2Vec multi-modal model.
        
        Args:
            cfg: Configuration object
            modalities: List of supported modalities
            skip_ema: Whether to skip EMA setup (for teacher model)
            task: Training task object
            
        Deep Meaning: This is the core initialization that sets up:
        1. Modality encoders (CNN for audio spectrograms)
        2. Transformer blocks (12 layers)
        3. EMA teacher model
        4. Decoders and projection layers
        5. Loss computation components
        """
        super().__init__()
        self.cfg = cfg
        self.modalities = modalities  # List of modalities: [Modality.IMAGE]
        self.task = task

        # ====================================================================
        # LAYER NORMALIZATION FACTORY
        # ====================================================================
        # Create a partial function for LayerNorm with fixed parameters
        # This ensures all LayerNorm layers have consistent configuration
        make_layer_norm = partial(
            nn.LayerNorm, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine
        )
        # Deep Meaning: Partial function creates LayerNorm with consistent settings
        # eps=1e-6 for numerical stability, elementwise_affine=True for learnable parameters

        # ====================================================================
        # TRANSFORMER BLOCK FACTORY
        # ====================================================================
        def make_block(drop_path, dim=None, heads=None):
            """
            Creates a transformer block with specified configuration.
            
            Args:
                drop_path: Drop path rate for stochastic depth
                dim: Hidden dimension (defaults to embed_dim)
                heads: Number of attention heads (defaults to num_heads)
                
            Returns:
                AltBlock: A transformer block
                
            Deep Meaning: Each block contains:
            - Multi-head self-attention
            - Feed-forward network (MLP)
            - Layer normalization
            - Residual connections
            - Dropout for regularization
            """
            return AltBlock(
                cfg.embed_dim if dim is None else dim,        # Hidden dimension: 768
                cfg.num_heads if heads is None else heads,    # Attention heads: 12
                cfg.mlp_ratio,                               # MLP ratio: 4 (3072 hidden units)
                qkv_bias=True,                              # Query, Key, Value bias: True
                drop=cfg.encoder_dropout,                    # Dropout rate: 0.1
                attn_drop=cfg.attention_dropout,             # Attention dropout: 0.1
                mlp_drop=cfg.activation_dropout,             # MLP dropout: 0.0
                post_mlp_drop=cfg.post_mlp_drop,             # Post-MLP dropout: 0.1
                drop_path=drop_path,                         # Stochastic depth rate
                norm_layer=make_layer_norm,                  # Layer normalization function
                layer_norm_first=cfg.layer_norm_first,      # LayerNorm order: False
                ffn_targets=not cfg.end_of_block_targets,   # Whether to use FFN outputs as targets
            )

        # ====================================================================
        # ALIBI BIASES AND MODALITY ENCODERS
        # ====================================================================
        self.alibi_biases = {}  # ALiBi biases (not used in EAT)
        self.modality_encoders = nn.ModuleDict()  # Dictionary of modality encoders
        
        # Extract CNN encoder and CNN decoder from modified data2vec image modality
        for mod in self.modalities:  # For each modality (typically just IMAGE)
            mod_cfg = getattr(cfg.modalities, mod.name.lower())  # Get modality config
            enc = self.make_modality_encoder(  # Create encoder for this modality
                mod_cfg,           # ImageEncoder configuration
                cfg.embed_dim,     # 768
                make_block,        # Transformer block factory
                make_layer_norm,   # LayerNorm factory
                cfg.layer_norm_first,  # False
                self.alibi_biases, # Empty dict
                task,             # Task object
            )
            self.modality_encoders[mod.name] = enc  # Store encoder: {"IMAGE": ImageEncoder}
        # Deep Meaning: Creates one ImageEncoder that processes audio spectrograms as images
        # The ImageEncoder contains CNN layers to convert patches to embeddings

        # ====================================================================
        # EMA TEACHER MODEL INITIALIZATION
        # ====================================================================
        self.ema = None  # Will be created if not skip_ema

        # ====================================================================
        # LOSS CONFIGURATION
        # ====================================================================
        self.average_top_k_layers = cfg.average_top_k_layers  # 12 (use all layers)
        self.loss_beta = cfg.loss_beta                        # 0 (use L2 loss)
        self.loss_scale = cfg.loss_scale                      # None (auto-scale)
        self.utterance_level = cfg.utterance_level            # False (no Dino loss)

        # ====================================================================
        # INPUT DROPOUT
        # ====================================================================
        self.dropout_input = nn.Dropout(cfg.dropout_input)  # 0.0 (no input dropout)

        # ====================================================================
        # STOCHASTIC DEPTH CONFIGURATION
        # ====================================================================
        # Create linearly spaced drop path rates from start to end
        dpr = np.linspace(cfg.start_drop_path_rate, cfg.end_drop_path_rate, cfg.depth)
        # Deep Meaning: Stochastic depth gradually increases drop probability through layers
        # EAT uses [0, 0, 0, ..., 0] since both start and end rates are 0

        # ====================================================================
        # TRANSFORMER BLOCKS CREATION
        # ====================================================================
        # Create 12 transformer blocks with individual drop path rates
        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(cfg.depth)])
        # Shape: ModuleList containing 12 AltBlock instances
        # Deep Meaning: Main transformer encoder that processes patch embeddings
        # Each block refines the representation through attention and MLP

        # ====================================================================
        # FINAL LAYER NORMALIZATION
        # ====================================================================
        self.norm = None
        if cfg.layer_norm_first:  # False in EAT
            self.norm = make_layer_norm(cfg.embed_dim)
        # Deep Meaning: Optional final layer normalization after all transformer blocks
        # EAT doesn't use this since layer_norm_first=False

        # ====================================================================
        # WEIGHT INITIALIZATION
        # ====================================================================
        if self.cfg.mae_init:  # False in EAT
            self.apply(self._init_weights)  # Custom MAE initialization
        else:
            # Use BERT-style initialization from fairseq
            from fairseq.modules.transformer_sentence_encoder import init_bert_params
            self.apply(init_bert_params)
        # Deep Meaning: BERT initialization helps with transformer training stability
        # Xavier uniform for weights, constant 0 for biases, constant 1 for LayerNorm

        # ====================================================================
        # MODALITY ENCODER PARAMETER RESET
        # ====================================================================
        for mod_enc in self.modality_encoders.values():
            mod_enc.reset_parameters()
        # Deep Meaning: Reset parameters in ImageEncoder after main model initialization
        # Ensures CNN layers have proper initial weights

        # ====================================================================
        # TEACHER MODEL AND DECODER SETUP
        # ====================================================================
        if not skip_ema:  # Only create EMA teacher for student model
            # Create EMA teacher model
            self.ema = self.make_ema_teacher(cfg.ema_decay)  # 0.999
            
            # Create shared decoder if specified
            self.shared_decoder = (
                Decoder1d(cfg.shared_decoder, cfg.embed_dim)
                if self.cfg.shared_decoder is not None
                else None
            )
            if self.shared_decoder is not None:
                self.shared_decoder.apply(self._init_weights)
            # Deep Meaning: Decoder converts transformer outputs back to patch predictions
            # Used for reconstruction loss and final target prediction

            # ================================================================
            # RECONSTRUCTION PROJECTION LAYER
            # ================================================================
            self.recon_proj = None
            if cfg.recon_loss > 0:  # 0 in EAT (no reconstruction loss)
                # Project from 768 to 256 dimensions for reconstruction
                self.recon_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim//3)
                # Shape: Linear(768, 256)
                # Deep Meaning: Reduces dimensionality for reconstruction target
                
            # ================================================================
            # CLASSIFICATION PROJECTION LAYER
            # ================================================================
            self.cls_proj = None
            if cfg.utterance_level:  # False in EAT (no Dino loss)
                # Project for utterance-level classification
                self.cls_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
                # Shape: Linear(768, 768)
                # Deep Meaning: Projects CLS token for utterance-level loss

        # ====================================================================
        # PARAMETER GROUPING AND OPTIMIZATION SETTINGS
        # ====================================================================
        for pn, p in self.named_parameters():
            # Disable weight decay for biases and 1D parameters
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}
                # Deep Meaning: Biases don't need weight decay regularization
                
            # Group decoder parameters separately if specified
            if cfg.decoder_group and "decoder" in pn:
                p.param_group = "decoder"
                # Deep Meaning: Allows different learning rates for decoder vs encoder

        # ====================================================================
        # DINo LOSS EXPERIMENTAL SETUP
        # ====================================================================
        self.center = None  # Center vector for Dino loss
        if self.utterance_level:  # False in EAT
            self.center_exp = cfg.center_exp                    # 0.9
            self.soft_tem_s = cfg.softmax_temperature_student   # 0.1
            self.soft_tem_t = cfg.softmax_temperature_teacher   # 0.05
            # Create center parameter for Dino loss
            self.center = nn.Parameter(
                torch.zeros(1, 1, cfg.embed_dim, requires_grad=False)
            )
            # Shape: (1, 1, 768) - global center vector
            # requires_grad=False because it's updated manually
            
            if not cfg.init_center_token_zero:  # False
                nn.init.normal_(self.center)  # Initialize with normal distribution
            elif self.center.size(1) > 1:
                nn.init.normal_(self.center[:, 1:])  # Initialize non-first tokens

        # ====================================================================
        # TRAINING STATE
        # ====================================================================
        self.num_updates = 0  # Track number of training updates

    # ========================================================================
    # WEIGHT INITIALIZATION METHOD
    # ========================================================================
    def _init_weights(self, m):
        """
        Custom weight initialization for MAE-style training.
        
        Args:
            m: Module to initialize
            
        Deep Meaning: MAE uses different initialization than BERT:
        - Xavier uniform for linear layers (better for reconstruction)
        - Standard LayerNorm initialization (bias=0, weight=1)
        """
        try:
            from apex.normalization import FusedLayerNorm
            fn = FusedLayerNorm  # Use fused LayerNorm if available
        except:
            fn = nn.LayerNorm    # Fallback to standard LayerNorm

        if isinstance(m, nn.Linear):
            # Xavier uniform initialization for linear layers
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize bias to 0
        elif isinstance(m, nn.LayerNorm) or isinstance(m, fn):
            # Standard LayerNorm initialization
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)    # Bias = 0
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)  # Weight = 1

    # ========================================================================
    # EMA TEACHER MODEL CREATION
    # ========================================================================
    @torch.no_grad()
    def make_ema_teacher(self, ema_decay):
        """
        Creates EMA teacher model for target generation.
        
        Args:
            ema_decay: Initial EMA decay rate (0.999)
            
        Returns:
            EMAModule: Wrapped teacher model
            
        Deep Meaning: EMA teacher provides stable targets for student learning.
        Teacher parameters are updated slowly using exponential moving average
        of student parameters. This prevents target collapse and improves
        representation quality.
        """
        ema_config = EMAModuleConfig(
            ema_decay=ema_decay,        # 0.999 - high retention of old parameters
            ema_fp32=True,              # Store in float32 for stability
            log_norms=self.cfg.log_norms, # Log parameter norms for monitoring
            add_missing_params=False,   # Don't add missing parameters
        )

        model_copy = self.make_target_model()  # Create teacher model copy

        return EMAModule(
            model_copy,      # The teacher model
            ema_config,      # EMA configuration
            copy_model=False, # Don't copy the model (use reference)
        )

    # ========================================================================
    # TARGET MODEL CREATION
    # ========================================================================
    def make_target_model(self):
        """
        Creates target model (teacher) for EMA updates.
        
        Returns:
            Data2VecMultiModel: Teacher model copy
            
        Deep Meaning: Creates a frozen copy of the student model to serve as teacher.
        The teacher model generates targets for the student to learn from.
        Teacher is updated via EMA to maintain stability while following student.
        """
        logger.info("making target model")

        # Create a new model instance (teacher)
        model_copy = Data2VecMultiModel(
            self.cfg, self.modalities, skip_ema=True, task=self.task
        )
        # Deep Meaning: skip_ema=True means teacher doesn't create its own EMA

        if self.cfg.ema_encoder_only:  # False in EAT
            # Only copy transformer blocks for EMA
            model_copy = model_copy.blocks
            for p_s, p_t in zip(self.blocks.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)  # Copy student parameters to teacher
        else:
            # Copy entire model for EMA (EAT uses this)
            for p_s, p_t in zip(self.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)  # Copy all student parameters

            # Remove decoders from teacher (teacher doesn't need decoders)
            for mod_enc in model_copy.modality_encoders.values():
                mod_enc.decoder = None
                if not mod_enc.modality_cfg.ema_local_encoder:
                    mod_enc.local_encoder = None
                    mod_enc.project_features = None

        # Freeze teacher parameters (no gradients needed)
        model_copy.requires_grad_(False)
        return model_copy

    # ========================================================================
    # EMA UPDATE MANAGEMENT
    # ========================================================================
    def set_num_updates(self, num_updates):
        """
        Updates EMA teacher model based on training progress.
        
        Args:
            num_updates: Current number of training updates
            
        Deep Meaning: Manages EMA updates and annealing schedule.
        EMA decay rate can be annealed from initial to final value
        to make teacher more stable over time.
        """
        super().set_num_updates(num_updates)

        # Skip update if already processed this step
        if self.ema is not None and (
            (self.num_updates == 0 and num_updates > 1)
            or self.num_updates >= num_updates
        ):
            pass
        elif self.training and self.ema is not None:
            ema_weight_decay = None
            
            # Anneal EMA decay rate if configured
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    # Use final decay rate
                    decay = self.cfg.ema_end_decay  # 0.9999
                else:
                    # Linear annealing from initial to final
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,        # 0.999
                        self.cfg.ema_end_decay,    # 0.9999
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay, weight_decay=ema_weight_decay)
            
            # Update teacher parameters if decay < 1
            if self.ema.get_decay() < 1:
                if self.cfg.ema_encoder_only:  # False in EAT
                    self.ema.step(self.blocks)  # Update only transformer blocks
                else:
                    self.ema.step(self)  # Update entire model

        self.num_updates = num_updates

    # ========================================================================
    # MODEL STATE MANAGEMENT
    # ========================================================================
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save model state including EMA parameters."""
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params  # Include EMA parameters

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Load model state including EMA parameters."""
        k = prefix + "_ema"
        if self.ema is not None:
            assert k in state_dict
            self.ema.restore(state_dict[k], True)  # Restore EMA parameters
            del state_dict[k]
        elif k in state_dict:
            del state_dict[k]

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # ========================================================================
    # MODEL BUILDING
    # ========================================================================
    @classmethod
    def build_model(cls, cfg: Data2VecMultiConfig, task=None):
        """
        Build a new model instance.
        
        Args:
            cfg: Model configuration
            task: Training task
            
        Returns:
            Data2VecMultiModel: New model instance
            
        Deep Meaning: Factory method that creates model instances.
        Determines supported modalities and creates appropriate model.
        """
        if task is None or not hasattr(task, "supported_modalities"):
            # Use configuration or default modalities
            modalities = (
                [cfg.supported_modality]  # Single modality from config
                if cfg.supported_modality is not None
                else [
                    Modality.AUDIO,
                    Modality.IMAGE,
                    Modality.TEXT,
                ]  # Default to all modalities
            )
        else:
            modalities = task.supported_modalities  # Use task modalities
            
        return cls(cfg, modalities, task=task, skip_ema=cfg.skip_ema)

    # ========================================================================
    # MAIN FORWARD PASS
    # ========================================================================
    def forward(
        self,
        source,
        target=None,
        id=None,
        mode=None,
        padding_mask=None,
        mask=True,
        features_only=False,
        force_remove_masked=False,
        remove_extra_tokens=True,
        precomputed_mask=None,
    ):
        """
        Main forward pass of the EAT pretraining model.
        
        Args:
            source: Input audio spectrograms
                   Shape: (batch_size, channels, height, width) 
                   Example: (12, 1, 1024, 128) for 12 samples of 1024x128 spectrograms
            target: Target audio (not used in pretraining)
            id: Sample IDs for random masking
            mode: Modality type (defaults to supported_modality)
            padding_mask: Padding mask for variable length sequences
            mask: Whether to apply masking
            features_only: Whether to return only features (for fine-tuning)
            force_remove_masked: Whether to force remove masked tokens
            remove_extra_tokens: Whether to remove CLS and other extra tokens
            precomputed_mask: Pre-computed mask (if available)
            
        Returns:
            dict: Contains losses, sample_size, and logging information
            
        Deep Meaning: This is the core pretraining forward pass that:
        1. Processes input through CNN encoder (patch embedding)
        2. Applies masking strategy
        3. Passes through student transformer
        4. Generates teacher targets via EMA teacher
        5. Computes multiple loss functions
        6. Returns training statistics
        """
        
        # ====================================================================
        # MODE DETERMINATION
        # ====================================================================
        if mode is None:
            assert self.cfg.supported_modality is not None
            mode = self.cfg.supported_modality  # Modality.IMAGE

        if isinstance(mode, Modality):
            mode = mode.name  # Convert to string: "IMAGE"

        # Get the appropriate feature extractor (ImageEncoder)
        feature_extractor = self.modality_encoders[mode]  # ImageEncoder instance

        # ====================================================================
        # MASK SEED GENERATION
        # ====================================================================
        mask_seeds = None
        if id is not None:
            # Create reproducible random masks based on sample IDs
            mask_seeds = MaskSeed(seed=self.cfg.seed, update=self.num_updates, ids=id)
            # Deep Meaning: Ensures same sample gets same mask across epochs
            # Important for consistent training and debugging

        # ====================================================================
        # FEATURE EXTRACTION AND MASKING
        # ====================================================================
        # Extract (unmasked) features using CNN encoder
        extractor_out = feature_extractor(
            source,                           # Input spectrograms
            padding_mask,                     # Padding mask
            mask,                            # Whether to apply masking
            remove_masked=not features_only or force_remove_masked,  # Remove masked tokens
            clone_batch=self.cfg.clone_batch if not features_only else 1,  # Clone batch size
            mask_seeds=mask_seeds,            # Random seed for masking
            precomputed_mask=precomputed_mask, # Pre-computed mask
        )
        # Deep Meaning: ImageEncoder converts spectrograms to patch embeddings
        # and applies masking strategy. Returns:
        # - x: Patch embeddings with CLS token
        # - encoder_mask: Masking information
        # - padding_mask: Attention padding mask
        # - local_features: Raw features before projection

        # ====================================================================
        # EXTRACT OUTPUTS FROM FEATURE EXTRACTOR
        # ====================================================================
        # x in shape (batch_size * clone_batch, num_patches + 1, embed_dim)
        # Example: (192, 513, 768) for batch_size=12, clone_batch=16
        # 192 = 12 * 16, 513 = 64*8*0.2 + 1 = 102.4 + 1 ≈ 103 patches + 1 CLS token
        # EAT does not employ the alibi mechanism in Transformer
        x = extractor_out["x"]                    # Patch embeddings + CLS token
        encoder_mask = extractor_out["encoder_mask"]  # Masking information
        masked_padding_mask = extractor_out["padding_mask"]  # Attention mask
        masked_alibi_bias = extractor_out.get("alibi_bias", None)  # ALiBi bias (not used)
        alibi_scale = extractor_out.get("alibi_scale", None)       # ALiBi scale (not used)

        # ====================================================================
        # INPUT DROPOUT
        # ====================================================================
        if self.dropout_input is not None:
            x = self.dropout_input(x)  # Apply dropout to patch embeddings
        # Deep Meaning: Input dropout helps with regularization during training
        # EAT uses 0.0 dropout rate (no input dropout)

        # ====================================================================
        # STUDENT TRANSFORMER ENCODER
        # ====================================================================
        # Standard Transformer (for student encoder)
        layer_results = []  # Store intermediate layer outputs
        
        for i, blk in enumerate(self.blocks):  # Process through 12 transformer blocks
            # Stochastic depth: randomly skip layers during training
            if (
                not self.training              # Always process during inference
                or self.cfg.layerdrop == 0     # No layer drop if layerdrop=0
                or (np.random.random() > self.cfg.layerdrop)  # Random skip probability
            ):
                # Prepare ALiBi bias (not used in EAT)
                ab = masked_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                # Forward through transformer block
                x, lr = blk(  # AltBlock forward pass
                    x,                           # Input embeddings
                    padding_mask=masked_padding_mask,  # Attention mask
                    alibi_bias=ab,               # ALiBi bias (None in EAT)
                )
                # Deep Meaning: Each transformer block refines the representation
                # x: Updated embeddings after attention and MLP
                # lr: Layer output for target generation (if needed)
                
                if features_only:
                    layer_results.append(lr)  # Store for feature extraction

        # ====================================================================
        # FINAL LAYER NORMALIZATION
        # ====================================================================
        if self.norm is not None:  # False in EAT (no final LayerNorm)
            x = self.norm(x)

        # ====================================================================
        # FEATURE EXTRACTION MODE (FOR FINE-TUNING)
        # ====================================================================
        if features_only:  # Return features for downstream tasks
            if remove_extra_tokens:  # Remove CLS token for patch-based tasks
                x = x[:, feature_extractor.modality_cfg.num_extra_tokens:]
                # Shape: (batch_size * clone_batch, num_patches, embed_dim)
                # Remove CLS token, keep only patch embeddings
                
                if masked_padding_mask is not None:
                    masked_padding_mask = masked_padding_mask[
                        :, feature_extractor.modality_cfg.num_extra_tokens:
                    ]

            return {
                "x": x,                          # Patch embeddings
                "padding_mask": masked_padding_mask,  # Attention mask
                "layer_results": layer_results,  # Intermediate layer outputs
                "mask": encoder_mask,            # Masking information
            }

        # ====================================================================
        # DECODER PROCESSING
        # ====================================================================
        # Decode features merged with masked tokens
        # dx in shape (batch_size * clone_batch, num_patches, embed_dim)
        xs = []  # List of decoder outputs
        
        # Shared decoder (if available)
        if self.shared_decoder is not None:
            dx = self.forward_decoder(
                x,                    # Transformer outputs
                feature_extractor,    # ImageEncoder
                self.shared_decoder,  # Decoder1d
                encoder_mask,         # Masking info
            )
            xs.append(dx)
            # Deep Meaning: Shared decoder processes all tokens (masked and unmasked)
            # Used for reconstruction loss and final predictions

        # Modality-specific decoder (if available)
        if feature_extractor.decoder is not None:
            dx = self.forward_decoder(
                x,                          # Transformer outputs
                feature_extractor,          # ImageEncoder
                feature_extractor.decoder,  # Modality-specific decoder
                encoder_mask,               # Masking info
            )
            xs.append(dx)  # Add to decoder outputs
            orig_x = x     # Store original transformer outputs

        assert len(xs) > 0, "At least one decoder must be available"

        # ====================================================================
        # EMA TEACHER MODEL PROCESSING
        # ====================================================================
        # Process input through teacher model to generate targets
        
        # Get device and dtype information
        p = next(self.ema.model.parameters())
        device = x.device      # Student model device
        dtype = x.dtype        # Student model dtype
        ema_device = p.device  # Teacher model device
        ema_dtype = p.dtype    # Teacher model dtype

        # Handle dtype conversion if needed
        if not self.cfg.ema_same_dtype:
            dtype = ema_dtype

        # Move teacher to same device/dtype as student if needed
        if ema_device != device or ema_dtype != dtype:
            logger.info(f"adjusting ema dtype to {dtype} and device to {device}")
            self.ema.model = self.ema.model.to(dtype=dtype, device=device)
            ema_dtype = dtype

            # Move EMA parameters to correct device
            def to_device(d):
                for k, p in d.items():
                    if isinstance(d[k], dict):
                        to_device(d[k])  # Recursive for nested dicts
                    else:
                        d[k] = p.to(device=device)

            to_device(self.ema.fp32_params)

        tm = self.ema.model  # Teacher model

        # ====================================================================
        # TEACHER MODEL FORWARD PASS
        # ====================================================================
        # Encode audio spectrogram using teacher model
        with torch.no_grad():  # No gradients for teacher
            tm.eval()  # Set teacher to evaluation mode

            if self.cfg.ema_encoder_only:  # False in EAT
                # Only use transformer blocks for EMA
                assert target is None
                ema_input = extractor_out["local_features"]  # Raw CNN features
                ema_input = feature_extractor.contextualized_features(
                    ema_input.to(dtype=ema_dtype),
                    padding_mask,
                    mask=False,          # Teacher sees all tokens (no masking)
                    remove_masked=False,
                )
                ema_blocks = tm  # Use transformer blocks directly
            else:
                # Use entire teacher model (EAT uses this)
                ema_blocks = tm.blocks  # Get transformer blocks
                
                if feature_extractor.modality_cfg.ema_local_encoder:
                    # Use teacher's CNN encoder
                    inp = (
                        target.to(dtype=ema_dtype)
                        if target is not None
                        else source.to(dtype=ema_dtype)
                    )
                    ema_input = tm.modality_encoders[mode](  # Teacher's ImageEncoder
                        inp,
                        padding_mask,
                        mask=False,          # Teacher sees all tokens
                        remove_masked=False,
                    )
                else:
                    # Use student's CNN features (EAT uses this)
                    assert target is None
                    ema_input = extractor_out["local_features"]  # Student's CNN features
                    ema_feature_enc = tm.modality_encoders[mode]  # Teacher's ImageEncoder
                    ema_input = ema_feature_enc.contextualized_features(
                        ema_input.to(dtype=ema_dtype),
                        padding_mask,
                        mask=False,          # Teacher sees all tokens
                        remove_masked=False,
                    )

            # Extract teacher processing results
            ema_padding_mask = ema_input["padding_mask"]      # Teacher attention mask
            ema_alibi_bias = ema_input.get("alibi_bias", None)    # ALiBi bias
            ema_alibi_scale = ema_input.get("alibi_scale", None)  # ALiBi scale
            ema_input = ema_input["x"]                        # Teacher patch embeddings

            # ================================================================
            # TEACHER TRANSFORMER PROCESSING
            # ================================================================
            # Extract target features using teacher CNN encoder
            # ema_input in shape (batch_size, num_patches + 1, embed_dim)
            # Example: (12, 513, 768) for batch_size=12
            y = []  # Store teacher layer outputs
            ema_x = []  # Store teacher embeddings
            extra_tokens = feature_extractor.modality_cfg.num_extra_tokens  # 1 (CLS token)
            
            for i, blk in enumerate(ema_blocks):  # Process through 12 teacher transformer blocks
                # Prepare ALiBi bias (not used in EAT)
                ab = ema_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        ema_alibi_scale[i]
                        if ema_alibi_scale.size(0) > 1
                        else ema_alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                # Forward through teacher transformer block
                ema_input, lr = blk(  # Teacher AltBlock forward pass
                    ema_input,                    # Teacher embeddings
                    padding_mask=ema_padding_mask, # Teacher attention mask
                    alibi_bias=ab,                # ALiBi bias (None in EAT)
                )
                # Deep Meaning: Teacher processes all tokens (no masking)
                # ema_input: Updated teacher embeddings
                # lr: Teacher layer output for target generation
                
                # Store layer outputs (excluding CLS token)
                y.append(lr[:, extra_tokens:])      # Shape: (batch_size, num_patches, embed_dim)
                ema_x.append(ema_input[:, extra_tokens:])  # Shape: (batch_size, num_patches, embed_dim)

        # ====================================================================
        # TARGET GENERATION
        # ====================================================================
        # EAT utilizes total 12 Transformer block layer output average as target
        y = self.make_targets(y, self.average_top_k_layers)  # Average all 12 layers
        # Shape: (batch_size, num_patches, embed_dim)
        # Deep Meaning: EAT uses ALL 12 transformer layers' output as teacher targets
        # This makes targets more stable and informative than using just the last layer
        orig_targets = y  # Store original targets for utterance-level loss

        # ====================================================================
        # CLONE BATCH PROCESSING
        # ====================================================================
        # Multiply the target value according to the number of clone batch
        if self.cfg.clone_batch > 1:  # 16 in EAT
            y = y.repeat_interleave(self.cfg.clone_batch, 0)
            # Shape: (batch_size * clone_batch, num_patches, embed_dim)
            # Example: (192, 512, 768) for batch_size=12, clone_batch=16
            # Deep Meaning: Each sample gets 16 different masks, so targets are repeated

        # ====================================================================
        # MASKED TOKEN EXTRACTION
        # ====================================================================
        # Extract values in masked position to make prediction
        masked = encoder_mask.mask.unsqueeze(-1)  # Shape: (batch_size * clone_batch, num_patches, 1)
        masked_b = encoder_mask.mask.bool()       # Shape: (batch_size * clone_batch, num_patches)
        y = y[masked_b]  # Extract only masked positions
        # Shape: (num_masked_tokens, embed_dim)
        # Example: (num_masked_tokens, 768) where num_masked_tokens = batch_size * clone_batch * num_patches * mask_ratio

        # ====================================================================
        # DECODER OUTPUT PROCESSING
        # ====================================================================
        if xs[0].size(1) == masked_b.size(1):
            # Decoder output matches mask size
            xs = [x[masked_b] for x in xs]  # Extract masked positions from all decoders
        else:
            # Decoder output is different size (flatten)
            xs = [x.reshape(-1, x.size(-1)) for x in xs]
        # Deep Meaning: Decoders produce predictions for all tokens, but we only need masked ones
        # Shape: List of (num_masked_tokens, embed_dim) tensors

        # ====================================================================
        # SAMPLE SIZE COMPUTATION
        # ====================================================================
        sample_size = masked.sum().long()  # Total number of masked tokens
        # Deep Meaning: Used for loss scaling and logging

        # ====================================================================
        # RESULT INITIALIZATION
        # ====================================================================
        result = {
            "losses": {},           # Dictionary of loss components
            "sample_size": sample_size,  # Number of masked tokens
        }

        sample_size = result["sample_size"]

        # ====================================================================
        # UTTERANCE-LEVEL CLASSIFICATION LOSS (STANDARD)
        # ====================================================================
        if self.cfg.cls_loss > 0 and not self.utterance_level:  # False in EAT
            assert extra_tokens > 0  # Need CLS token
            # Compute utterance-level target by averaging patch targets
            cls_target = orig_targets.mean(dim=1)  # Shape: (batch_size, embed_dim)
            # Deep Meaning: Average all patch representations to get utterance representation
            
            if self.cfg.clone_batch > 1:
                cls_target = cls_target.repeat_interleave(self.cfg.clone_batch, 0)
                # Shape: (batch_size * clone_batch, embed_dim)
            
            # Get CLS token prediction from student
            cls_pred = x[:, extra_tokens - 1]  # Shape: (batch_size * clone_batch, embed_dim)
            # Deep Meaning: Use CLS token as utterance representation
            
            # Compute classification loss
            result["losses"]["cls"] = self.d2v_loss(cls_pred, cls_target) * (
                self.cfg.cls_loss * sample_size
            )
            # Deep Meaning: L2 loss between CLS token and averaged patch targets

        # ====================================================================
        # DINo LOSS EXPERIMENT (UTTERANCE-LEVEL)
        # ====================================================================
        if self.cfg.cls_loss > 0 and self.utterance_level:  # False in EAT
            assert extra_tokens > 0  # Need CLS token
            # Compute utterance-level target by averaging patch targets
            cls_target = orig_targets.mean(dim=1)  # Shape: (batch_size, embed_dim)
            
            if self.cfg.clone_batch > 1:
                cls_target = cls_target.repeat_interleave(self.cfg.clone_batch, 0)
                # Shape: (batch_size * clone_batch, embed_dim)
            
            # Get CLS token prediction from student
            cls_pred = x[:, extra_tokens - 1]  # Shape: (batch_size * clone_batch, embed_dim)
            
            # Center the target (Dino loss requires centering)
            cls_target = cls_target - self.center  # Shape: (batch_size * clone_batch, embed_dim)
            # Deep Meaning: Center targets around global center to prevent collapse
            
            # Remove extra dimensions
            cls_pred = cls_pred.squeeze(dim=1)    # Shape: (batch_size * clone_batch, embed_dim)
            cls_target = cls_target.squeeze(dim=1)  # Shape: (batch_size * clone_batch, embed_dim)
            
            # Compute Dino loss
            result["losses"]["cls"] = self.dino_loss(cls_pred, cls_target) * (
                self.cfg.cls_loss * sample_size
            )
            # Deep Meaning: Dino loss prevents representation collapse through centering and temperature
            
            # Update center with exponential moving average
            self.center = self.center_exp * self.center + (1 - self.center_exp) * (cls_target.mean(dim=0))
            # Shape: (1, embed_dim)
            # Deep Meaning: Center follows the mean of current batch targets

        # ====================================================================
        # RECONSTRUCTION LOSS
        # ====================================================================
        if self.cfg.recon_loss > 0:  # 0 in EAT (no reconstruction loss)
            with torch.no_grad():
                # Create reconstruction target from original spectrogram
                target = feature_extractor.patchify(source)  # Shape: (batch_size, 1, 512, 16*16)
                # Deep Meaning: Convert spectrogram back to patches for reconstruction
                
                # Normalize target
                mean = target.mean(dim=-1, keepdim=True)  # Shape: (batch_size, 1, 512, 1)
                var = target.var(dim=-1, keepdim=True)    # Shape: (batch_size, 1, 512, 1)
                target = (target - mean) / (var + 1.0e-6) ** 0.5  # Shape: (batch_size, 1, 512, 1)
                # Deep Meaning: Standardize target for stable training

                if self.cfg.clone_batch > 1:
                    target = target.repeat_interleave(self.cfg.clone_batch, 0)
                    # Shape: (batch_size * clone_batch, 1, 512, 1)

                if masked_b is not None:
                    target = target[masked_b]  # Extract masked positions
                    # Shape: (num_masked_tokens, 1, 512, 1)

            # Get reconstruction prediction
            recon = xs[0]  # Use first decoder output
            if self.recon_proj is not None:
                recon = self.recon_proj(recon)  # Project to target dimension
                # Shape: (num_masked_tokens, embed_dim//3)

            # Compute reconstruction loss
            result["losses"]["recon"] = (
                self.d2v_loss(recon, target.float()) * self.cfg.recon_loss
            )
            # Deep Meaning: L2 loss between predicted and original patches

        # ====================================================================
        # DATA2VEC REGRESSION LOSS (MAIN LOSS)
        # ====================================================================
        if self.cfg.d2v_loss > 0:  # 1.0 in EAT (main loss)
            for i, x in enumerate(xs):  # For each decoder output
                reg_loss = self.d2v_loss(x, y)  # Compute regression loss
                # Deep Meaning: L2 loss between student predictions and teacher targets
                
                # Name the loss based on decoder type
                n = f"{mode}_regression_{i}" if len(xs) > 1 else f"{mode}_regression"
                result["losses"][n] = reg_loss * self.cfg.d2v_loss
                # Deep Meaning: Main pretraining loss that teaches student to predict teacher

        # ====================================================================
        # LOGGING AND MONITORING
        # ====================================================================
        # Compute state for logs
        suffix = "" if len(self.modalities) == 1 else f"_{mode}"
        
        with torch.no_grad():
            # Log masking percentage
            if encoder_mask is not None:
                result["masked_pct"] = 1 - (
                    encoder_mask.ids_keep.size(1) / encoder_mask.ids_restore.size(1)
                )
                # Deep Meaning: Percentage of tokens that are masked (should be ~0.8 for 80% masking)
            
            # Log prediction variance for each decoder
            for i, x in enumerate(xs):
                n = f"pred_var{suffix}_{i}" if len(xs) > 1 else f"pred_var{suffix}"
                result[n] = self.compute_var(x.float())
                # Deep Meaning: Variance of predictions (monitors for collapse)
            
            # Log EMA statistics
            if self.ema is not None:
                for k, v in self.ema.logs.items():
                    result[k] = v
                # Deep Meaning: EMA decay rate and parameter norms

            # Log target variance
            y = y.float()
            result[f"target_var{suffix}"] = self.compute_var(y)
            # Deep Meaning: Variance of teacher targets (monitors for collapse)

            # ================================================================
            # COLLAPSE DETECTION
            # ================================================================
            if self.num_updates > 5000:  # Only check after warmup
                # Check target variance (teacher collapse)
                if result[f"target_var{suffix}"] < self.cfg.min_target_var:
                    logger.error(
                        f"target var is {result[f'target_var{suffix}'].item()} < {self.cfg.min_target_var}, exiting ({mode})"
                    )
                    raise Exception(
                        f"target var is {result[f'target_var{suffix}'].item()} < {self.cfg.min_target_var}, exiting ({mode})"
                    )
                    # Deep Meaning: Stop training if teacher targets become too similar (collapse)

                # Check prediction variance (student collapse)
                for k in result.keys():
                    if k.startswith("pred_var") and result[k] < self.cfg.min_pred_var:
                        logger.error(
                            f"{k} is {result[k].item()} < {self.cfg.min_pred_var}, exiting ({mode})"
                        )
                        raise Exception(
                            f"{k} is {result[k].item()} < {self.cfg.min_pred_var}, exiting ({mode})"
                        )
                        # Deep Meaning: Stop training if student predictions become too similar (collapse)

            # Log EMA decay rate
            result["ema_decay"] = self.ema.get_decay() * 1000
            # Deep Meaning: Current EMA decay rate (scaled for logging)

        return result

    # ========================================================================
    # DECODER FORWARD PASS
    # ========================================================================
    def forward_decoder(
        self,
        x,                    # Transformer outputs
        feature_extractor,    # ImageEncoder
        decoder,              # Decoder1d
        mask_info,            # Masking information
    ):
        """
        Forward pass through decoder.
        
        Args:
            x: Transformer outputs, Shape: (batch_size * clone_batch, num_patches + 1, embed_dim)
            feature_extractor: ImageEncoder instance
            decoder: Decoder1d instance
            mask_info: Masking information
            
        Returns:
            torch.Tensor: Decoder outputs, Shape: (batch_size * clone_batch, num_patches, embed_dim)
            
        Deep Meaning: Decoder converts transformer outputs back to patch predictions.
        It processes all tokens (masked and unmasked) to produce final predictions.
        """
        # Prepare decoder input (add masked tokens back)
        x = feature_extractor.decoder_input(x, mask_info)
        # Deep Meaning: Decoder input restores masked tokens to their original positions
        # Shape: (batch_size * clone_batch, num_patches, embed_dim)
        
        # Forward through decoder
        x = decoder(*x)  # Decoder1d forward pass
        # Deep Meaning: Decoder refines predictions for all patches
        # Shape: (batch_size * clone_batch, num_patches, embed_dim)

        return x

    # ========================================================================
    # DATA2VEC LOSS COMPUTATION
    # ========================================================================
    def d2v_loss(self, x, y):
        """
        Compute Data2Vec regression loss.
        
        Args:
            x: Student predictions, Shape: (num_masked_tokens, embed_dim)
            y: Teacher targets, Shape: (num_masked_tokens, embed_dim)
            
        Returns:
            torch.Tensor: Loss value, Shape: (num_masked_tokens,)
            
        Deep Meaning: Main pretraining loss that teaches student to predict teacher.
        Uses L2 loss (MSE) or Smooth L1 loss depending on configuration.
        """
        # Flatten to 2D for loss computation
        x = x.view(-1, x.size(-1)).float()  # Shape: (num_masked_tokens, embed_dim)
        y = y.view(-1, x.size(-1))          # Shape: (num_masked_tokens, embed_dim)

        if self.loss_beta == 0:  # Use L2 loss (MSE)
            loss = F.mse_loss(x, y, reduction="none")
            # Shape: (num_masked_tokens, embed_dim)
            # Deep Meaning: L2 loss is more sensitive to large errors
        else:  # Use Smooth L1 loss (Huber loss)
            loss = F.smooth_l1_loss(x, y, reduction="none", beta=self.loss_beta)
            # Shape: (num_masked_tokens, embed_dim)
            # Deep Meaning: Smooth L1 is more robust to outliers than L2

        # Scale loss by feature dimension
        if self.loss_scale is not None:
            scale = self.loss_scale  # Use provided scale
        else:
            scale = 1 / math.sqrt(x.size(-1))  # Auto-scale by 1/sqrt(embed_dim)
            # Deep Meaning: Prevents loss from being too large/small due to dimension

        reg_loss = loss * scale  # Shape: (num_masked_tokens, embed_dim)
        # Deep Meaning: Scaled loss for stable training

        return reg_loss

    # ========================================================================
    # DINo LOSS COMPUTATION
    # ========================================================================
    def dino_loss(self, s, t):
        """
        Compute Dino loss for utterance-level learning.
        
        Args:
            s: Student predictions, Shape: (batch_size * clone_batch, embed_dim)
            t: Teacher targets, Shape: (batch_size * clone_batch, embed_dim)
            
        Returns:
            torch.Tensor: Dino loss value, Shape: scalar
            
        Deep Meaning: Dino loss prevents representation collapse through:
        1. Centering: Subtracts global center from targets
        2. Temperature scaling: Controls sharpness of softmax
        3. Cross-entropy: Maximizes agreement between student and teacher
        """
        t = t.detach()  # Stop gradients for teacher
        
        # Apply temperature scaling and softmax to student
        s = F.softmax(s / self.soft_tem_s, dim=1)  # Shape: (batch_size * clone_batch, embed_dim)
        # Deep Meaning: Higher temperature (0.1) makes distribution softer
        
        # Apply temperature scaling and softmax to teacher
        t = F.softmax((t - self.center) / self.soft_tem_t, dim=1)  # Shape: (batch_size * clone_batch, embed_dim)
        # Deep Meaning: Lower temperature (0.05) makes distribution sharper, center prevents collapse
        
        # Compute cross-entropy loss
        return -(t * torch.log(s)).sum(dim=1).mean()
        # Deep Meaning: Cross-entropy between teacher and student distributions
        # Minimizes KL divergence between teacher and student

    # ========================================================================
    # TARGET GENERATION
    # ========================================================================
    def make_targets(self, y, num_layers):
        """
        Generate teacher targets by averaging multiple layer outputs.
        
        Args:
            y: List of teacher layer outputs, Length: 12
                Each element Shape: (batch_size, num_patches, embed_dim)
            num_layers: Number of layers to average (12 in EAT)
            
        Returns:
            torch.Tensor: Averaged targets, Shape: (batch_size, num_patches, embed_dim)
            
        Deep Meaning: EAT uses ALL 12 transformer layers' output as teacher targets.
        This makes targets more stable and informative than using just the last layer.
        Averaging multiple layers provides richer supervision signal.
        """
        with torch.no_grad():
            # Get the last num_layers outputs
            target_layer_results = y[-num_layers:]  # Last 12 layers
            # Deep Meaning: Use all 12 layers for target generation

            # Apply normalization if configured
            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                # Transpose for normalization: (batch, time, channels) -> (batch, channels, time)
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results
                ]
                permuted = True
                # Deep Meaning: Normalization expects channels in second dimension

            if self.cfg.batch_norm_target_layer:
                # Apply batch normalization to each layer
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]
                # Deep Meaning: BatchNorm normalizes across batch dimension

            if self.cfg.instance_norm_target_layer:
                # Apply instance normalization to each layer
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]
                # Deep Meaning: InstanceNorm normalizes across spatial dimension

            if permuted:
                # Transpose back: (batch, channels, time) -> (batch, time, channels)
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results
                ]

            if self.cfg.layer_norm_target_layer:
                # Apply layer normalization to each layer
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]
                # Deep Meaning: LayerNorm normalizes across feature dimension

        # Average all layer outputs
        y = target_layer_results[0].float()  # Start with first layer
        for tl in target_layer_results[1:]:  # Add remaining layers
            y.add_(tl.float())
        y = y.div_(len(target_layer_results))  # Divide by number of layers
        # Deep Meaning: Simple average of all 12 layer outputs
        # Shape: (batch_size, num_patches, embed_dim)

        # Apply final normalization if configured
        if self.cfg.layer_norm_targets:
            y = F.layer_norm(y, y.shape[-1:])
            # Deep Meaning: Final LayerNorm for target stability

        if self.cfg.instance_norm_targets:
            y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)
            # Deep Meaning: Final InstanceNorm for target stability

        return y

    # ========================================================================
    # VARIANCE COMPUTATION
    # ========================================================================
    @staticmethod
    def compute_var(y):
        """
        Compute variance of predictions/targets for collapse detection.
        
        Args:
            y: Predictions or targets, Shape: (num_tokens, embed_dim)
            
        Returns:
            torch.Tensor: Mean standard deviation, Shape: scalar
            
        Deep Meaning: Monitors representation quality and detects collapse.
        Low variance indicates representations are becoming too similar.
        """
        y = y.view(-1, y.size(-1))  # Flatten to 2D
        
        if dist.is_initialized():  # Distributed training
            # Compute variance across all processes
            zc = torch.tensor(y.size(0)).cuda()  # Count
            zs = y.sum(dim=0)                    # Sum
            zss = (y**2).sum(dim=0)             # Sum of squares

            dist.all_reduce(zc)   # Sum across processes
            dist.all_reduce(zs)   # Sum across processes
            dist.all_reduce(zss)  # Sum across processes

            # Compute variance: Var(X) = E[X²] - E[X]²
            var = zss / (zc - 1) - (zs**2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()  # Mean standard deviation
        else:  # Single process
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()
            # Deep Meaning: Compute standard deviation across feature dimension

    # ========================================================================
    # FEATURE EXTRACTION
    # ========================================================================
    def extract_features(
        self, source, mode=None, padding_mask=None, mask=False, remove_extra_tokens=True
    ):
        """
        Extract features for downstream tasks.
        
        Args:
            source: Input audio spectrograms
            mode: Modality type
            padding_mask: Padding mask
            mask: Whether to apply masking
            remove_extra_tokens: Whether to remove CLS token
            
        Returns:
            dict: Extracted features and metadata
            
        Deep Meaning: Used for fine-tuning and feature extraction.
        Returns patch embeddings without computing losses.
        """
        res = self.forward(
            source,
            mode=mode,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,  # Only return features
            remove_extra_tokens=remove_extra_tokens,
        )
        return res

    # ========================================================================
    # PRETRAINING MODULE REMOVAL
    # ========================================================================
    def remove_pretraining_modules(self, modality=None, keep_decoder=False):
        """
        Remove pretraining-specific modules for fine-tuning.
        
        Args:
            modality: Which modality to keep (None for all)
            keep_decoder: Whether to keep decoders
            
        Deep Meaning: Prepares model for fine-tuning by removing:
        1. EMA teacher model
        2. Decoders (unless keep_decoder=True)
        3. Reconstruction projection
        4. Unused modality encoders
        """
        self.ema = None  # Remove EMA teacher
        self.cfg.clone_batch = 1  # Disable clone batch
        self.recon_proj = None  # Remove reconstruction projection

        if not keep_decoder:
            self.shared_decoder = None  # Remove shared decoder

        # Remove unused modality encoders
        modality = modality.lower() if modality is not None else None
        for k in list(self.modality_encoders.keys()):
            if modality is not None and k.lower() != modality:
                del self.modality_encoders[k]  # Remove unused modalities
            else:
                # Remove pretraining modules from kept encoders
                self.modality_encoders[k].remove_pretraining_modules(
                    keep_decoder=keep_decoder
                )
                if not keep_decoder:
                    self.modality_encoders[k].decoder = None  # Remove decoders