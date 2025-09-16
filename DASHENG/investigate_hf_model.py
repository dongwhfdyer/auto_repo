#!/usr/bin/env python3
"""
Script to investigate HuggingFace model structure and find proper key mapping
"""
import json
import torch
from pathlib import Path
import sys

def investigate_hf_model(model_dir):
    """Investigate the structure of a HuggingFace model"""
    print(f"Investigating HuggingFace model: {model_dir}")
    
    # Read config
    config_path = Path(model_dir) / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Config keys: {list(config.keys())}")
        print(f"Config: {config}")
    else:
        print("No config.json found")
        return
    
    # Read state dict
    state_dict = None
    p = Path(model_dir)
    
    # Try safetensors first
    idx = p / "model.safetensors.index.json"
    if idx.exists():
        from safetensors.torch import load_file
        with open(idx, "r") as f:
            index = json.load(f)
        merged = {}
        for shard in {Path(x).name for x in index["weight_map"].values()}:
            merged.update(load_file(str(p / shard), device="cpu"))
        state_dict = merged
        print("Loaded from safetensors shards")
    else:
        st_path = p / "model.safetensors"
        if st_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(st_path), device="cpu")
            print("Loaded from single safetensors file")
        else:
            ptbin = p / "pytorch_model.bin"
            if ptbin.exists():
                state_dict = torch.load(str(ptbin), map_location="cpu")
                print("Loaded from pytorch_model.bin")
    
    if state_dict is None:
        print("No model weights found")
        return
    
    print(f"Total keys in state dict: {len(state_dict)}")
    
    # Look for audio/frontend related keys
    audio_keys = [k for k in state_dict.keys() if any(x in k.lower() for x in ['audio', 'spectrogram', 'mel', 'frontend', 'feature', 'conv', 'stft'])]
    print(f"Audio/frontend related keys: {audio_keys}")
    
    # Look for any keys that might be the frontend
    frontend_candidates = [k for k in state_dict.keys() if 'front' in k.lower() or 'spec' in k.lower() or 'mel' in k.lower()]
    print(f"Frontend candidate keys: {frontend_candidates}")
    
    # Show first 30 keys
    print(f"First 30 keys: {list(state_dict.keys())[:30]}")
    
    return state_dict, config

def check_dasheng_expected_keys():
    """Check what keys dasheng model expects"""
    print("\n" + "="*50)
    print("Checking dasheng model expected keys...")
    
    try:
        from dasheng import dasheng_base
        model = dasheng_base()
        state_dict = model.state_dict()
        
        print(f"Dasheng model expects {len(state_dict)} keys")
        
        # Look for frontend keys
        frontend_keys = [k for k in state_dict.keys() if 'front_end' in k]
        print(f"Frontend keys: {frontend_keys}")
        
        # Look for spectrogram keys
        spec_keys = [k for k in state_dict.keys() if 'spectrogram' in k]
        print(f"Spectrogram keys: {spec_keys}")
        
        return state_dict
        
    except Exception as e:
        print(f"Error creating dasheng model: {e}")
        return None

def find_key_mapping(hf_keys, dasheng_keys):
    """Find potential mapping between HF and dasheng keys"""
    print("\n" + "="*50)
    print("Looking for key mappings...")
    
    # Look for similar patterns
    mappings = []
    
    for hf_key in hf_keys:
        for dasheng_key in dasheng_keys:
            # Check if there's a pattern match
            if any(part in dasheng_key.lower() for part in hf_key.lower().split('.')):
                mappings.append((hf_key, dasheng_key))
    
    print(f"Potential mappings found: {len(mappings)}")
    for hf_key, dasheng_key in mappings[:10]:  # Show first 10
        print(f"  {hf_key} -> {dasheng_key}")
    
    return mappings

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python investigate_hf_model.py <path_to_hf_model_dir>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    hf_state_dict, hf_config = investigate_hf_model(model_dir)
    dasheng_state_dict = check_dasheng_expected_keys()
    
    if hf_state_dict and dasheng_state_dict:
        find_key_mapping(list(hf_state_dict.keys()), list(dasheng_state_dict.keys()))
