"""
Utterance-Frame Objective (UFO) Implementation Code Snippets
From EAT/models/EAT_pretraining.py
"""

import torch
import torch.nn.functional as F
import math

# ============================================================================
# 1. UFO Configuration Parameters
# ============================================================================

class Data2VecMultiConfig:
    # d2v_loss is the frame-level loss while cls_loss is the utterance-level loss
    cls_loss: float = 0          # Utterance-level loss weight
    recon_loss: float = 0        # Reconstruction loss weight  
    d2v_loss: float = 1          # Frame-level loss weight (default enabled)
    
    # UFO configuration
    utterance_level: bool = False  # if true, we will add utterance-level loss to the total loss
    
    # Target computation
    average_top_k_layers: int = 12  # how many layers to average
    
    # Multi-mask strategy
    clone_batch: int = 16  # Create 16 different masks per sample

# ============================================================================
# 2. Target Generation (Teacher Model)
# ============================================================================

def extract_teacher_targets(ema_blocks, ema_input, ema_padding_mask, extra_tokens):
    """
    Extract contextualized targets from teacher model's Transformer layers
    """
    y = []
    ema_x = []
    
    for i, blk in enumerate(ema_blocks):  
        # Process through each Transformer block
        ema_input, lr = blk(
            ema_input,
            padding_mask=ema_padding_mask,
            alibi_bias=None,  # EAT doesn't use alibi
        )
        y.append(lr[:, extra_tokens:])  # Store layer outputs (exclude CLS token)
        ema_x.append(ema_input[:, extra_tokens:])

    # EAT utilize total 12 Transformer block layer output average as target  
    y = make_targets(y, 12)  # average_top_k_layers=12
    return y

def make_targets(y, num_layers):
    """
    Average top-k layers output from teacher model
    """
    with torch.no_grad():
        target_layer_results = y[-num_layers:]  # Take last num_layers
        
        # Apply normalization if configured
        if instance_norm_target_layer or batch_norm_target_layer:
            target_layer_results = [
                tl.transpose(1, 2) for tl in target_layer_results  # BTC -> BCT
            ]
            permuted = True
            
        if batch_norm_target_layer:
            target_layer_results = [
                F.batch_norm(
                    tl.float(), running_mean=None, running_var=None, training=True
                )
                for tl in target_layer_results
            ]
            
        if instance_norm_target_layer:
            target_layer_results = [
                F.instance_norm(tl.float()) for tl in target_layer_results
            ]
            
        if permuted:
            target_layer_results = [
                tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
            ]
            
        if layer_norm_target_layer:
            target_layer_results = [
                F.layer_norm(tl.float(), tl.shape[-1:])
                for tl in target_layer_results
            ]

        # Average all layer outputs
        y = target_layer_results[0].float()
        for tl in target_layer_results[1:]:
            y.add_(tl.float())
        y = y.div_(len(target_layer_results))  # Average across layers

        if layer_norm_targets:
            y = F.layer_norm(y, y.shape[-1:])

        if instance_norm_targets:
            y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        return y

# ============================================================================
# 3. Utterance-Level Learning (CLS Token)
# ============================================================================

def compute_utterance_loss(x, orig_targets, extra_tokens, clone_batch, cls_loss, sample_size):
    """
    EAT employ utterance-level loss by using mean pooling in patch dimension
    """
    if cls_loss > 0 and not utterance_level:
        assert extra_tokens > 0
        cls_target = orig_targets.mean(dim=1)  # Mean pooling across patch dimension
        if clone_batch > 1:
            cls_target = cls_target.repeat_interleave(clone_batch, 0)
        cls_pred = x[:, extra_tokens - 1]  # CLS token from student encoder
        
        utterance_loss = d2v_loss(cls_pred, cls_target) * (cls_loss * sample_size)
        return utterance_loss
    return None

# ============================================================================
# 4. Frame-Level Learning (Masked Prediction)
# ============================================================================

def compute_frame_loss(xs, y, mode, d2v_loss):
    """
    Frame-level learning predicts masked features using the decoder
    """
    frame_losses = {}
    
    if d2v_loss > 0:
        for i, x in enumerate(xs):
            reg_loss = d2v_loss(x, y)  # x: decoder output, y: target features
            n = f"{mode}_regression_{i}" if len(xs) > 1 else f"{mode}_regression"
            frame_losses[n] = reg_loss * d2v_loss
            
    return frame_losses

def forward_decoder(x, feature_extractor, decoder, mask_info):
    """
    Decoder processing: merge with mask tokens and apply lightweight CNN decoder
    """
    x = feature_extractor.decoder_input(x, mask_info)  # Merge with mask tokens
    x = decoder(*x)  # Lightweight CNN decoder
    return x

# ============================================================================
# 5. Loss Function Implementation
# ============================================================================

def d2v_loss(x, y, loss_beta=0, loss_scale=None):
    """
    Both utterance and frame-level losses use the same MSE-based loss function
    """
    x = x.view(-1, x.size(-1)).float()
    y = y.view(-1, x.size(-1))

    if loss_beta == 0:
        loss = F.mse_loss(x, y, reduction="none")  # MSE loss
    else:
        loss = F.smooth_l1_loss(x, y, reduction="none", beta=loss_beta)

    if loss_scale is not None:
        scale = loss_scale
    else:
        scale = 1 / math.sqrt(x.size(-1))  # Scale by 1/sqrt(dim)

    reg_loss = loss * scale
    return reg_loss

# ============================================================================
# 6. Main UFO Forward Pass
# ============================================================================

def ufo_forward_pass(source, model, feature_extractor, mode="AUDIO"):
    """
    Main UFO forward pass combining utterance and frame-level learning
    """
    # 1. Extract features using student encoder (with masking)
    extractor_out = feature_extractor(
        source,
        padding_mask=None,
        mask=True,
        remove_masked=False,
        clone_batch=model.cfg.clone_batch,
        mask_seeds=None,
        precomputed_mask=None,
    )
    
    x = extractor_out["x"]  # Student encoder output
    encoder_mask = extractor_out["encoder_mask"]
    
    # 2. Process through student Transformer blocks
    layer_results = []
    for blk in model.blocks:
        x, lr = blk(x, padding_mask=extractor_out["padding_mask"])
        layer_results.append(lr)
    
    if model.norm is not None:
        x = model.norm(x)
    
    # 3. Decode features for frame-level prediction
    xs = []
    if feature_extractor.decoder is not None:
        dx = forward_decoder(x, feature_extractor, feature_extractor.decoder, encoder_mask)
        xs.append(dx)
    
    # 4. Generate teacher targets
    with torch.no_grad():
        # Teacher model processes full input (no masking)
        ema_input = feature_extractor.contextualized_features(
            extractor_out["local_features"],
            padding_mask=None,
            mask=False,
            remove_masked=False,
        )
        
        y = extract_teacher_targets(
            model.ema.model.blocks,
            ema_input["x"],
            ema_input["padding_mask"],
            feature_extractor.modality_cfg.num_extra_tokens
        )
    
    # 5. Handle multi-mask strategy
    if model.cfg.clone_batch > 1:
        y = y.repeat_interleave(model.cfg.clone_batch, 0)
    
    # 6. Extract masked positions for frame-level loss
    masked_b = encoder_mask.mask.bool()
    y = y[masked_b]
    xs = [x[masked_b] for x in xs]
    
    sample_size = encoder_mask.mask.sum().long()
    
    # 7. Compute UFO losses
    result = {"losses": {}, "sample_size": sample_size}
    
    # Utterance-level loss
    utterance_loss = compute_utterance_loss(
        x, y, 
        feature_extractor.modality_cfg.num_extra_tokens,
        model.cfg.clone_batch,
        model.cfg.cls_loss,
        sample_size
    )
    if utterance_loss is not None:
        result["losses"]["cls"] = utterance_loss
    
    # Frame-level loss
    frame_losses = compute_frame_loss(xs, y, mode, model.cfg.d2v_loss)
    result["losses"].update(frame_losses)
    
    return result

# ============================================================================
# 7. Configuration Example
# ============================================================================

# Example configuration from pretraining_AS2M.yaml
config_example = {
    "model": {
        "cls_loss": 1,              # Enable utterance-level loss
        "d2v_loss": 1,              # Enable frame-level loss (default)
        "average_top_k_layers": 12, # Average all 12 layers for targets
        "clone_batch": 16,          # Multi-mask strategy
        
        "modalities": {
            "image": {  # Audio treated as image modality
                "inverse_mask": True,   # Inverse block masking
                "mask_prob": 0.8,       # 80% masking rate
                "mask_prob_adjust": 0.07,
                "mask_length": 5,       # Block mask length
                "mask_noise_std": 0.01, # Mask noise standard deviation
                "num_extra_tokens": 1,  # CLS token
                "init_extra_token_zero": False,
            }
        }
    }
}

# ============================================================================
# 8. UFO Loss Summary
# ============================================================================

"""
Total UFO Loss: LUFO = Lf + λLu

Where:
- Lf = frame-level loss (d2v_loss) - predicts masked features
- Lu = utterance-level loss (cls_loss) - predicts global audio representation  
- λ = cls_loss weight parameter

Key Features:
1. Asymmetric Architecture: Complex Transformer encoder + lightweight CNN decoder
2. EMA Teacher: Teacher model updated via exponential moving average
3. High Masking Rate: 80% masking for efficiency
4. Layer Averaging: Targets from average of all 12 Transformer layers
5. Direct Regression: No additional projectors for utterance-level learning
6. Multi-Mask Efficiency: 16 different masks per sample
""" 