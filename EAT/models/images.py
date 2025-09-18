# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import partial
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from timm.models.layers import to_2tuple
from fairseq.tasks import FairseqTask
from enum import Enum, auto

from .mae import PatchEmbed,get_2d_sincos_pos_embed_flexible,PatchEmbed_new
from .mae import get_1d_sincos_pos_embed_from_grid

from .base import (
    D2vModalityConfig,
    ModalitySpecificEncoder,
    get_alibi_bias,
    MaskSeed,
)
from .modules import (
    BlockEncoder,
    Decoder2d,
    FixedPositionalEncoder,
    TransformerDecoder,
    EncDecTransformerDecoder,
)


class Modality(Enum):
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()


@dataclass
class D2vImageConfig(D2vModalityConfig):
    type: Modality = Modality.IMAGE

    input_size: int = 224
    in_chans: int = 3
    patch_size: int = 16
    embed_dim: int = 768

    alibi_dims: int = 2
    alibi_distance: str = "manhattan"

    fixed_positions: bool = True

    transformer_decoder: bool = False
    enc_dec_transformer: bool = False
    target_length: int = 1024
    max_length: int = 768
    
    # Band-splitting parameters for audio
    band_width: int = 32
    shift_size: int = 16
    use_band_splitting: bool = True
    freq_pos_emb_dim: int = 768


def get_sincos_encoding_1d(pos, dim, freq_scale=25):
    """
    Generate 1D sin/cos positional encoding for a normalized position tensor (N, 1)
    Args:
        pos: (N, 1) normalized position in [0, 1]
        dim: output embedding dimension
    Returns:
        (N, dim) tensor
    """
    import math
    device = pos.device
    N = pos.shape[0]
    pe = torch.zeros(N, dim, device=device)
    position = pos  # (N, 1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * freq_scale * div_term)
    pe[:, 1::2] = torch.cos(position * freq_scale * div_term)
    return pe


class ImageEncoder(ModalitySpecificEncoder):

    modality_cfg: D2vImageConfig

    def __init__(
        self,
        modality_cfg: D2vImageConfig,
        embed_dim: int,
        make_block: Callable[[float, Optional[int], Optional[int]], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
        task: Optional[FairseqTask],
    ):
        
        if modality_cfg.in_chans == 1 :  
            img_size = (modality_cfg.target_length,128)
        else:
            img_size =  to_2tuple(modality_cfg.input_size)

        patch_size = to_2tuple(modality_cfg.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # number of patch -> 512
        self.H = img_size[0] // patch_size[0]  # 64
        self.W = img_size[1] // patch_size[1]  # 8
        self.hw = (self.H,self.W)

        # (B,512,768)
        # note: we fix the variable length sequence problem here -> not limited to fixed length data
        local_encoder = PatchEmbed_new(
            img_size,
            modality_cfg.patch_size,
            modality_cfg.in_chans,
            modality_cfg.embed_dim,
        )

        # CNN initialize
        w = local_encoder.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if modality_cfg.embed_dim != embed_dim:
            local_encoder = nn.Sequential(
                local_encoder,
                nn.Linear(modality_cfg.embed_dim, embed_dim),
            )

        project_features = nn.Identity()

        # note: max_length control the maximum time length of audio -> "64" for 10s, here we define it as 2min, you can change it yourself
        max_length = modality_cfg.max_length
        pos_embed = nn.Parameter(
            torch.zeros(1, max_length*self.W, embed_dim), requires_grad=False
        )

        # side_n = int(num_patches ** 0.5)
        # note: we fix the variable length sequence problem here -> support up to 2min audio 
        emb = get_2d_sincos_pos_embed_flexible(
            pos_embed.shape[-1],
            (max_length,self.W),  
            cls_token=False,
        )
        
        pos_embed.data.copy_(torch.from_numpy(emb[:max_length*self.W,:]).float().unsqueeze(0)) 
        fixed_positional_encoder = (
            FixedPositionalEncoder(pos_embed) if modality_cfg.fixed_positions else None
        )
        
        # Band-splitting initialization
        self.use_band_splitting = modality_cfg.use_band_splitting
        self.band_width = modality_cfg.band_width
        self.shift_size = modality_cfg.shift_size
        self.freq_pos_emb_dim = modality_cfg.freq_pos_emb_dim
        
        # Store original dimensions for band-splitting
        # 1D patch embedding for time dimension (like ECHO)
        self.patch_embed_1d = nn.Conv2d(
            1, embed_dim, 
            kernel_size=(self.band_width, self.shift_size), 
            stride=(self.band_width, self.shift_size)
        )
        
        # 1D time positional encoding (like ECHO)
        max_time_patches = max_length // self.shift_size
        if max_length % self.shift_size != 0:
            max_time_patches += 1
        self.time_pos_embed = nn.Parameter(
            torch.zeros(1, max_time_patches, embed_dim), requires_grad=False
        )
        
        # Initialize 1D time positional encoding
        time_pos_indices = np.arange(max_time_patches, dtype=np.float32)
        time_pos_emb = get_1d_sincos_pos_embed_from_grid(
            self.time_pos_embed.shape[-1], time_pos_indices
        )
        self.time_pos_embed.data.copy_(torch.from_numpy(time_pos_emb).float().unsqueeze(0))
        
        # Store original dimensions for band-splitting
        self.original_H = self.H
        self.original_W = self.W
        self.original_hw = self.hw

        dpr = np.linspace(
            modality_cfg.start_drop_path_rate,
            modality_cfg.end_drop_path_rate,
            modality_cfg.prenet_depth,
        )

        context_encoder = BlockEncoder(
            nn.ModuleList(make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)),
            norm_layer(embed_dim) if not layer_norm_first else None,
            layer_norm_first,
            modality_cfg.prenet_layerdrop,
            modality_cfg.prenet_dropout,
        )

        # EAT utilize the CNN decoder
        if modality_cfg.transformer_decoder:
            if modality_cfg.enc_dec_transformer:
                decoder = EncDecTransformerDecoder(modality_cfg.decoder, embed_dim)
            else:
                dec_enc = BlockEncoder(
                    nn.ModuleList(
                        make_block(0, modality_cfg.decoder.decoder_dim, 8)
                        for _ in range(modality_cfg.decoder.decoder_layers)
                    ),
                    None,
                    layer_norm_first,
                    0,
                    0,
                )
                decoder = TransformerDecoder(modality_cfg.decoder, embed_dim, dec_enc)
        else:
            decoder = (
                Decoder2d(modality_cfg.decoder, embed_dim, self.H, self.W)
                if modality_cfg.decoder is not None
                else None
            )

        alibi_bias_fn = partial(
            get_alibi_bias,
            alibi_biases=alibi_biases,
            heads=modality_cfg.num_alibi_heads,
            dims=modality_cfg.alibi_dims,
            distance=modality_cfg.alibi_distance,
        )

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=local_encoder,
            project_features=project_features,
            fixed_positional_encoder=fixed_positional_encoder,
            relative_positional_encoder=None,
            context_encoder=context_encoder,
            decoder=decoder,
            get_alibi_bias=alibi_bias_fn,
        )

    def reset_parameters(self):
        super().reset_parameters()
        if self.decoder is not None:
            self.decoder.reset_parameters()

    def _compute_frequency_position_encoding(self, spectrogram, sample_rate):
        """
        Band splitting and frequency position encoding for single spectrogram.
        
        Args:
            spectrogram: Single spectrogram tensor (freq_bins, time)
            sample_rate: Audio sample rate (scalar)
            
        Returns:
            freq_pos_emb: Frequency position embeddings for all bands
            band_spectrograms: List of band spectrograms (num_bands, band_width, time)
            band_patch_indices: List of band patch indices (num_bands, 2)
        """
        num_freq_bins = spectrogram.shape[0]
        num_bands = max(1, (num_freq_bins + self.modality_cfg.band_width - 1) // self.modality_cfg.band_width)
        
        band_spectrograms = []
        band_patch_indices = []
        all_freq_pos_norm = []
        
        for band_idx in range(num_bands):
            start_freq = band_idx * self.modality_cfg.band_width
            end_freq = min((band_idx + 1) * self.modality_cfg.band_width, num_freq_bins)
            
            if end_freq > start_freq:
                band_spec = spectrogram[start_freq:end_freq, :]  # (band_freq, time)
                
                # Pad the band if it's smaller than band_width
                if band_spec.shape[0] < self.modality_cfg.band_width:
                    padding_height = self.modality_cfg.band_width - band_spec.shape[0]
                    band_spec = torch.nn.functional.pad(band_spec, (0, 0, padding_height, 0), mode='constant', value=0)
                
                band_spectrograms.append(band_spec)
                band_patch_indices.append(torch.tensor([start_freq, end_freq], dtype=torch.long, device=spectrogram.device))
                
                # Compute frequency position encoding
                center_bin = (start_freq + end_freq - 1) / 2.0
                n_fft = int(0.025 * sample_rate)
                center_freq = center_bin * sample_rate / n_fft
                nyquist = sample_rate / 2.0
                freq_pos_norm = center_freq / nyquist if nyquist > 0 else 0.0
                all_freq_pos_norm.append(freq_pos_norm)
        
        # Convert to tensors
        if band_spectrograms:
            band_spectrograms = torch.stack(band_spectrograms, dim=0)  # (num_bands, band_width, T)
            band_patch_indices = torch.stack(band_patch_indices, dim=0)  # (num_bands, 2)
            freq_pos_norm = torch.tensor(all_freq_pos_norm, dtype=torch.float32, device=spectrogram.device).unsqueeze(1)  # (num_bands, 1)
            freq_pos_emb = get_sincos_encoding_1d(freq_pos_norm, self.modality_cfg.freq_pos_emb_dim)  # (num_bands, freq_pos_emb_dim)
            
        else:
            raise ValueError("No bands found in spectrogram")
        
        return freq_pos_emb, band_spectrograms, band_patch_indices

    def _compute_frequency_position_encoding_batch(self, spectrograms, sample_rates):
        """
        Band splitting and frequency position encoding for list of spectrograms.
        
        Args:
            spectrograms: List of full spectrograms, each with shape (freq_bins, time)
            sample_rates: Audio sample rate (B,) or scalar
            
        Returns:
            freq_pos_emb: Frequency position embeddings for all bands from all samples, (total_bands, freq_pos_emb_dim)
            band_spectrograms: Concatenated band spectrograms from all samples, (total_bands, band_width, T)
            band_patch_indices: Concatenated band patch indices from all samples, (total_bands, 2)
        """
        # Handle sample_rate
        if isinstance(sample_rates, (int, float)) or (hasattr(sample_rates, 'dim') and sample_rates.dim() == 0):
            # scalar -> (B,)
            sample_rates = [sample_rates] * len(spectrograms)
        
        all_band_spectrograms = []
        all_band_patch_indices = []
        all_freq_pos_emb = []
        
        for spec, sr in zip(spectrograms, sample_rates):
            # Process each spectrogram using the single spectrogram function
            freq_pos_emb, band_spectrograms, band_patch_indices = self._compute_frequency_position_encoding(spec, sr)
            
            # Collect results
            all_band_spectrograms.append(band_spectrograms)
            all_band_patch_indices.append(band_patch_indices)
            all_freq_pos_emb.append(freq_pos_emb)
        
        # Concatenate all results
        if all_band_spectrograms:
            band_spectrograms = torch.cat(all_band_spectrograms, dim=0)  # (total_bands, band_width, T)
            band_patch_indices = torch.cat(all_band_patch_indices, dim=0)  # (total_bands, 2)
            freq_pos_emb = torch.cat(all_freq_pos_emb, dim=0)  # (total_bands, freq_pos_emb_dim)
        else:
            raise ValueError("No bands found in spectrogram")
        
        return freq_pos_emb, band_spectrograms, band_patch_indices

    def local_features(self, features):
        """
        Override local_features to add band-splitting for audio.
        This is the correct place to add band-splitting functionality.
        """
        # Check if we should use band-splitting for audio
        if (self.modality_cfg.in_chans == 1 and 
            self.modality_cfg.use_band_splitting and 
            hasattr(features, 'shape') and len(features.shape) == 3):
            # Audio spectrogram with band-splitting
            return self._local_features_with_band_splitting(features)
        else:
            # Original EAT processing (for images or audio without band-splitting)
            return super().local_features(features)

    def _local_features_with_band_splitting(self, features):
        """
        Process audio spectrograms with band-splitting.
        """
        # Assume features is a list of spectrograms or a single spectrogram
        if isinstance(features, list):
            spectrograms = features
            sample_rates = [16000] * len(features)  # Default sample rate
        else:
            # Single spectrogram
            spectrograms = [features]
            sample_rates = [16000]  # Default sample rate
        
        # Apply band-splitting
        freq_pos_emb, band_spectrograms, band_patch_indices = self._compute_frequency_position_encoding_batch(
            spectrograms, sample_rates
        )
        # Use 1D sliding window patch embedding (like ECHO)
        # band_spectrograms: (total_bands, band_width, T)
        # Conv2d expects (N, C=1, H=band_width, W=T)
        band_spectrograms = band_spectrograms.unsqueeze(1)
        # Apply 1D patch embedding with sliding window -> (N, D, 1, num_patches)
        patches_4d = self.patch_embed_1d(band_spectrograms)
        # Reshape to (N, num_patches, D)
        patches = patches_4d.flatten(2).transpose(1, 2)
        
        # Store frequency positional encoding for later use (NO positional encoding here)
        self._freq_pos_emb = freq_pos_emb
        self._band_patch_indices = band_patch_indices
        
        return patches        

    def contextualized_features(
        self,
        x,
        padding_mask,
        mask,
        remove_masked,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):
        """
        Override contextualized_features for band-splitting.
        Complete implementation with our 1D positional encoding.
        """
        if padding_mask is not None:
            padding_mask = self.convert_padding_mask(x, padding_mask)

        local_features = x
        if mask and clone_batch == 1:
            local_features = local_features.clone()

        orig_B, orig_T, _ = x.shape
        pre_mask_B = orig_B
        mask_info = None

        # Add our 1D positional encodings (like ECHO, but before masking)
        if hasattr(self, "_freq_pos_emb") and self._freq_pos_emb is not None:
            # Add 1D time positional encoding
            time_patches = x.shape[1]
            x = x + self.time_pos_embed[:, :time_patches, :]
            
            # Add 1D frequency positional encoding
            freq_pos_emb_expanded = self._freq_pos_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = x + freq_pos_emb_expanded

        # Handle masking (copied from parent method)
        if mask:
            if clone_batch > 1:
                x = x.repeat_interleave(clone_batch, 0)
                if mask_seeds is not None:
                    clone_hash = [
                        int(hash((mask_seeds.seed, ind)) % 1e10)
                        for ind in range(clone_batch - 1)
                    ]
                    clone_hash = torch.tensor([0] + clone_hash).long().view(1, -1)

                    id = mask_seeds.ids
                    id = id.repeat_interleave(clone_batch, 0)
                    id = id.view(-1, clone_batch) + clone_hash.to(id)
                    id = id.view(-1)
                    mask_seeds = MaskSeed(
                        seed=mask_seeds.seed, update=mask_seeds.update, ids=id
                    )
                if padding_mask is not None:
                    padding_mask = padding_mask.repeat_interleave(clone_batch, 0)

            x, mask_info = self.compute_mask(
                x,
                padding_mask,
                mask_seed=mask_seeds,
                apply=self.relative_positional_encoder is not None or not remove_masked,
                precomputed_mask=precomputed_mask,
            )

        # Handle relative positional encoding (copied from parent method)
        x_pos = None
        if self.relative_positional_encoder is not None:
            x_pos = self.relative_positional_encoder(x)

        masked_padding_mask = padding_mask
        if mask and remove_masked:
            x = mask_info.x_unmasked
            if x_pos is not None:
                from .base import gather_unmasked
                x = x + gather_unmasked(x_pos, mask_info)

            if padding_mask is not None and padding_mask.any():
                from .base import gather_unmasked_mask
                masked_padding_mask = gather_unmasked_mask(padding_mask, mask_info)
                if not masked_padding_mask.any():
                    masked_padding_mask = None
            else:
                masked_padding_mask = None

        elif x_pos is not None:
            x = x + x_pos

        # Handle alibi bias (copied from parent method)
        alibi_bias = None
        alibi_scale = self.alibi_scale

        if self.get_alibi_bias is not None:
            alibi_bias = self.get_alibi_bias(
                batch_size=pre_mask_B,
                time_steps=orig_T,
                heads=self.modality_cfg.num_alibi_heads,
                dtype=torch.float32,
                device=x.device,
            )

            if alibi_scale is not None:
                alibi_scale = alibi_scale.clamp_min(0)
                if alibi_scale.size(0) == 1:
                    alibi_bias = alibi_bias * alibi_scale.squeeze(0).type_as(alibi_bias)
                    alibi_scale = None

            if clone_batch > 1:
                alibi_bias = alibi_bias.repeat_interleave(clone_batch, 0)

            if mask_info is not None and remove_masked:
                from .base import masked_alibi
                alibi_bias = masked_alibi(alibi_bias, mask_info)

        # Handle extra tokens (copied from parent method)
        if self.extra_tokens is not None:
            num = self.extra_tokens.size(1)
            x = torch.cat([self.extra_tokens.expand(x.size(0), -1, -1), x], dim=1)
            if masked_padding_mask is not None:
                masked_padding_mask = F.pad(masked_padding_mask, (num, 0))
            if alibi_bias is not None:
                alibi_bias = F.pad(alibi_bias, (num, 0, num, 0))

        # Pass through transformer encoder
        x = self.context_encoder(
            x,
            masked_padding_mask,
            alibi_bias,
            alibi_scale[: self.modality_cfg.prenet_depth]
            if alibi_scale is not None
            else None,
        )

        return {
            "x": x,
            "local_features": local_features,
            "padding_mask": masked_padding_mask,
            "alibi_bias": alibi_bias,
            "alibi_scale": alibi_scale[self.modality_cfg.prenet_depth :]
            if alibi_scale is not None and alibi_scale.size(0) > 1
            else alibi_scale,
            "encoder_mask": mask_info,
        }

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)   audio: (N,1,H,W)   1024/16 = 64   128/16 = 8
        x: (N, L, patch_size**2 *3)
        """
        if self.modality_cfg.in_chans == 1:
            p = self.modality_cfg.patch_size
            h = imgs.shape[2] // p
            w = imgs.shape[3] // p
            #h,w = self.patch_embed.patch_hw
            x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
            
        else:
            p = self.modality_cfg.patch_size
            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum("nchpwq->nhwpqc", x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

        return x

    @torch.no_grad()
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.modality_cfg.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def compute_mask(
        self,
        x,
        padding_mask,
        mask_seed: Optional[MaskSeed],
        apply,
        shape=None,
        precomputed_mask=None,
    ):
        mlen = self.modality_cfg.mask_length
        if mlen <= 1:
            return super().compute_mask(
                x, padding_mask, mask_seed, apply, precomputed_mask
            )

        if precomputed_mask is not None:
            mask = precomputed_mask
        else:
            from ..utils.data_utils import compute_block_mask_2d

            if shape is not None:
                B, L, D = shape
            else:
                B, L, D = x.shape

            mask = compute_block_mask_2d(
                shape=(B, L),
                mask_prob=self.modality_cfg.mask_prob,
                mask_length=self.modality_cfg.mask_length,
                mask_prob_adjust=self.modality_cfg.mask_prob_adjust,
                inverse_mask=self.modality_cfg.inverse_mask,
                require_same_masks=True,
                mask_dropout=self.modality_cfg.mask_dropout,
                img_shape=self.hw
            )
            

        mask_info = self.make_maskinfo(x, mask, shape)
        if apply:
            x = self.apply_mask(x, mask_info)

        return x, mask_info

    def decoder_input(self, x, mask_info):
        if (
            not self.modality_cfg.transformer_decoder
            or not self.modality_cfg.enc_dec_transformer
        ):
            return super().decoder_input(x, mask_info)

        inp_drop = self.modality_cfg.decoder.input_dropout
        if inp_drop > 0:
            x = F.dropout(x, inp_drop, training=self.training, inplace=True)

        kv = x[:, self.modality_cfg.num_extra_tokens :]

        assert self.fixed_positional_encoder is not None
        pos = self.fixed_positional_encoder(x, None).expand(x.size(0), -1, -1)

        mask = mask_info.mask.bool()
        if self.modality_cfg.decoder.add_positions_all:
            kv = kv + pos[~mask].view(kv.shape)

        q = pos[mask].view(x.size(0), -1, x.size(-1))

        return q, kv
