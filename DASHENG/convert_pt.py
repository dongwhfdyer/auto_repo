import json, torch
import torchaudio.transforms as audio_transforms
from pathlib import Path

def read_config(model_dir):
    with open(Path(model_dir) / "config.json", "r") as f:
        return json.load(f)

def read_state_dict(model_dir):
    from glob import glob
    p = Path(model_dir)
    idx = p / "model.safetensors.index.json"
    if idx.exists():
        # merge safetensors shards
        from safetensors.torch import load_file
        with open(idx, "r") as f:
            index = json.load(f)
        merged = {}
        for shard in {Path(x).name for x in index["weight_map"].values()}:
            merged.update(load_file(str(p / shard), device="cpu"))
        return merged
    st_path = p / "model.safetensors"
    if st_path.exists():
        from safetensors.torch import load_file
        return load_file(str(st_path), device="cpu")
    ptbin = p / "pytorch_model.bin"
    if ptbin.exists():
        return torch.load(str(ptbin), map_location="cpu")
    raise FileNotFoundError("No model.safetensors(.index.json) or pytorch_model.bin found")

def normalize_keys(state_dict):
    def strip_prefix(k):
        for pref in ("model.", "encoder.", "module."):
            if k.startswith(pref):
                return k[len(pref):]
        return k
    return {strip_prefix(k): v for k, v in state_dict.items()}

def map_config_for_dasheng(cfg):
    # Ensure required keys exist; fill/rename as needed for Dasheng constructor
    # Example minimal fields:
    mapped = dict(cfg)
    # If your HF config uses different names, set these explicitly:
    # mapped["embed_dim"] = cfg.get("hidden_size", cfg.get("embed_dim"))
    # mapped["depth"] = cfg.get("num_hidden_layers", cfg.get("depth"))
    # mapped["num_heads"] = cfg.get("num_attention_heads", cfg.get("num_heads"))
    # Also ensure patch settings, pooling, target_length, hop_size match Dasheng
    return mapped

def find_frontend_mapping(hf_state_dict, dasheng_config):
    """Find the correct mapping from HF model to dasheng frontend keys"""
    print("Looking for frontend key mappings...")
    
    # Common patterns for audio frontend in HF models
    hf_frontend_patterns = [
        'feature_extractor', 'feature_encoder', 'conv_layers', 'conv_layers_conv',
        'spectrogram', 'mel_spectrogram', 'stft', 'mel_scale', 'window',
        'audio_encoder', 'preprocessor', 'frontend'
    ]
    
    # Find HF keys that might be frontend related
    hf_frontend_keys = []
    for key in hf_state_dict.keys():
        if any(pattern in key.lower() for pattern in hf_frontend_patterns):
            hf_frontend_keys.append(key)
    
    print(f"Found {len(hf_frontend_keys)} potential frontend keys in HF model:")
    for key in hf_frontend_keys:
        print(f"  {key}")
    
    # Expected dasheng frontend keys
    expected_dasheng_keys = [
        "front_end.0.spectrogram.window",
        "front_end.0.mel_scale.fb"
    ]
    
    missing_keys = []
    for expected_key in expected_dasheng_keys:
        if expected_key not in hf_state_dict:
            missing_keys.append(expected_key)
    
    print(f"Missing dasheng keys: {missing_keys}")
    
    return hf_frontend_keys, missing_keys

def convert_hf_frontend_to_dasheng(hf_state_dict, dasheng_config):
    """Convert HF frontend keys to dasheng format"""
    hf_frontend_keys, missing_keys = find_frontend_mapping(hf_state_dict, dasheng_config)
    
    if not missing_keys:
        print("No missing frontend keys found!")
        return hf_state_dict
    
    # If we have HF frontend keys, try to map them
    if hf_frontend_keys:
        print("Attempting to map HF frontend keys to dasheng format...")
        # This would require specific knowledge of the HF model structure
        # For now, we'll flag this as needing manual investigation
        print("WARNING: Manual key mapping required. Please investigate the HF model structure.")
        print("Consider using the investigate_hf_model.py script to understand the mapping.")
    
    return hf_state_dict

def convert_hf_to_dasheng_pt(model_dir, out_path):
    cfg = map_config_for_dasheng(read_config(model_dir))
    sd = normalize_keys(read_state_dict(model_dir))
    
    # Investigate and convert frontend keys properly
    sd = convert_hf_frontend_to_dasheng(sd, cfg)
    
    torch.save({"model": sd, "config": cfg}, out_path)
    print(f"Saved Dasheng checkpoint to {out_path}")
    print("NOTE: If you see missing key warnings, run investigate_hf_model.py to understand the mapping")

# convert_hf_to_dasheng_pt("/path/to/hf_repo_dir", "/path/to/dasheng_converted.pt")