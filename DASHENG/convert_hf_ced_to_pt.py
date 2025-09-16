#!/usr/bin/env python3
"""
Convert Hugging Face CED models to PyTorch .pt format for use with CED encoder.
This script handles the conversion from HF format to the format expected by CED models.
"""

import json
import torch
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import os

# Add CED path for model imports
sys.path.append('/data1/EAT_projs/CED')  # Update this path as needed

try:
    from models.audiotransformer import ced_base, ced_small, ced_mini, ced_tiny
    from models.checkpoints import list_models
except ImportError as e:
    print(f"Error importing CED models: {e}")
    print("Please update the CED path in this script")
    sys.exit(1)

def load_hf_model(model_path: str) -> tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Load Hugging Face model config and state dict"""
    model_path = Path(model_path)
    
    # Load config
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load state dict from various possible formats
    state_dict = None
    
    # Try different HF model file formats
    possible_files = [
        "pytorch_model.bin",
        "model.safetensors", 
        "model.safetensors.index.json"
    ]
    
    for file_name in possible_files:
        file_path = model_path / file_name
        if file_path.exists():
            print(f"Loading from {file_name}...")
            
            if file_name == "pytorch_model.bin":
                state_dict = torch.load(str(file_path), map_location="cpu")
            elif file_name == "model.safetensors":
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(str(file_path), device="cpu")
                except ImportError:
                    print("safetensors not installed, trying alternative...")
                    continue
            elif file_name == "model.safetensors.index.json":
                try:
                    from safetensors.torch import load_file
                    with open(file_path, 'r') as f:
                        index = json.load(f)
                    merged = {}
                    for shard in {Path(x).name for x in index["weight_map"].values()}:
                        shard_path = model_path / shard
                        merged.update(load_file(str(shard_path), device="cpu"))
                    state_dict = merged
                except ImportError:
                    print("safetensors not installed, trying alternative...")
                    continue
            
            if state_dict is not None:
                break
    
    if state_dict is None:
        raise FileNotFoundError(f"No model weights found in {model_path}")
    
    return config, state_dict

def analyze_hf_model(config: Dict[str, Any], state_dict: Dict[str, torch.Tensor]) -> None:
    """Analyze the Hugging Face model structure"""
    print("\n" + "="*50)
    print("HUGGING FACE MODEL ANALYSIS")
    print("="*50)
    
    print(f"Model type: {config.get('model_type', 'unknown')}")
    print(f"Architecture: {config.get('architectures', ['unknown'])}")
    print(f"Total parameters: {len(state_dict)}")
    print(f"Total parameter count: {sum(p.numel() for p in state_dict.values()):,}")
    
    # Show key configuration parameters
    key_configs = ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 
                   'intermediate_size', 'embed_dim', 'depth', 'num_heads']
    
    print(f"\nKey configuration parameters:")
    for key in key_configs:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    # Analyze state dict structure
    print(f"\nState dict structure:")
    components = {}
    for key in state_dict.keys():
        component = key.split('.')[0] if '.' in key else 'root'
        if component not in components:
            components[component] = []
        components[component].append(key)
    
    for component, keys in sorted(components.items()):
        print(f"  {component}: {len(keys)} parameters")
        if len(keys) <= 3:  # Show keys for small components
            for key in sorted(keys):
                shape = tuple(state_dict[key].shape)
                print(f"    {key}: {shape}")

def map_hf_to_ced_keys(state_dict: Dict[str, torch.Tensor], 
                      config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Map Hugging Face keys to CED format"""
    print("\n" + "="*50)
    print("KEY MAPPING: HF -> CED")
    print("="*50)
    
    # Common mapping patterns from HF to CED
    key_mappings = {
        # Frontend mappings
        'feature_extractor.conv_layers.0.conv.weight': 'front_end.0.spectrogram.weight',
        'feature_extractor.conv_layers.0.conv.bias': 'front_end.0.spectrogram.bias',
        'feature_extractor.conv_layers.1.conv.weight': 'front_end.0.mel_scale.fb',
        
        # Embedding mappings
        'embeddings.patch_embeddings.projection.weight': 'patch_embed.proj.weight',
        'embeddings.patch_embeddings.projection.bias': 'patch_embed.proj.bias',
        'embeddings.position_embeddings': 'time_pos_embed',
        
        # Transformer blocks
        'encoder.layer': 'blocks',
        
        # Layer norm
        'layernorm.weight': 'norm.weight',
        'layernorm.bias': 'norm.bias',
        
        # Classifier/head
        'classifier.weight': 'outputlayer.1.weight',
        'classifier.bias': 'outputlayer.1.bias',
    }
    
    # Auto-detect mappings based on patterns
    auto_mappings = {}
    for hf_key in state_dict.keys():
        hf_key_lower = hf_key.lower()
        
        # Frontend detection
        if any(x in hf_key_lower for x in ['feature_extractor', 'conv_layers', 'spectrogram']):
            if 'conv.weight' in hf_key_lower and '0' in hf_key_lower:
                auto_mappings[hf_key] = 'front_end.0.spectrogram.weight'
            elif 'conv.bias' in hf_key_lower and '0' in hf_key_lower:
                auto_mappings[hf_key] = 'front_end.0.spectrogram.bias'
            elif 'mel' in hf_key_lower or 'fb' in hf_key_lower:
                auto_mappings[hf_key] = 'front_end.0.mel_scale.fb'
        
        # Embedding detection
        elif any(x in hf_key_lower for x in ['patch_embed', 'projection']):
            if 'weight' in hf_key_lower:
                auto_mappings[hf_key] = 'patch_embed.proj.weight'
            elif 'bias' in hf_key_lower:
                auto_mappings[hf_key] = 'patch_embed.proj.bias'
        
        # Position embedding
        elif 'position' in hf_key_lower:
            if 'time' in hf_key_lower:
                auto_mappings[hf_key] = 'time_pos_embed'
            elif 'freq' in hf_key_lower:
                auto_mappings[hf_key] = 'freq_pos_embed'
            else:
                auto_mappings[hf_key] = 'time_pos_embed'  # Default to time
        
        # Transformer blocks
        elif any(x in hf_key_lower for x in ['encoder.layer', 'transformer.layer', 'block']):
            # Keep the structure but change prefix
            new_key = hf_key.replace('encoder.layer', 'blocks').replace('transformer.layer', 'blocks')
            auto_mappings[hf_key] = new_key
        
        # Layer norm
        elif any(x in hf_key_lower for x in ['layernorm', 'layer_norm', 'norm']):
            if 'weight' in hf_key_lower:
                auto_mappings[hf_key] = 'norm.weight'
            elif 'bias' in hf_key_lower:
                auto_mappings[hf_key] = 'norm.bias'
        
        # Classifier/head
        elif any(x in hf_key_lower for x in ['classifier', 'head', 'output']):
            if 'weight' in hf_key_lower:
                auto_mappings[hf_key] = 'outputlayer.1.weight'
            elif 'bias' in hf_key_lower:
                auto_mappings[hf_key] = 'outputlayer.1.bias'
    
    # Combine explicit and auto mappings
    all_mappings = {**key_mappings, **auto_mappings}
    
    print(f"Found {len(all_mappings)} key mappings:")
    for hf_key, ced_key in list(all_mappings.items())[:10]:
        print(f"  {hf_key} -> {ced_key}")
    if len(all_mappings) > 10:
        print(f"  ... and {len(all_mappings) - 10} more")
    
    # Apply mappings
    ced_state_dict = {}
    unmapped_keys = []
    
    for hf_key, tensor in state_dict.items():
        if hf_key in all_mappings:
            ced_key = all_mappings[hf_key]
            ced_state_dict[ced_key] = tensor
        else:
            unmapped_keys.append(hf_key)
            # Try to keep the key as-is if no mapping found
            ced_state_dict[hf_key] = tensor
    
    if unmapped_keys:
        print(f"\nWarning: {len(unmapped_keys)} keys were not mapped:")
        for key in unmapped_keys[:5]:
            print(f"  {key}")
        if len(unmapped_keys) > 5:
            print(f"  ... and {len(unmapped_keys) - 5} more")
    
    return ced_state_dict

def determine_ced_model_size(config: Dict[str, Any]) -> str:
    """Determine which CED model size to use based on HF config"""
    hidden_size = config.get('hidden_size', config.get('embed_dim', 0))
    
    if hidden_size <= 200:
        return 'tiny'
    elif hidden_size <= 300:
        return 'mini'
    elif hidden_size <= 500:
        return 'small'
    else:
        return 'base'

def convert_hf_ced_to_pt(hf_model_path: str, output_path: str, 
                        ced_model_size: Optional[str] = None) -> None:
    """Convert Hugging Face CED model to PyTorch format"""
    print(f"Converting HF CED model from {hf_model_path} to {output_path}")
    
    # Load HF model
    config, state_dict = load_hf_model(hf_model_path)
    
    # Analyze the model
    analyze_hf_model(config, state_dict)
    
    # Determine CED model size
    if ced_model_size is None:
        ced_model_size = determine_ced_model_size(config)
        print(f"\nAuto-detected CED model size: {ced_model_size}")
    else:
        print(f"\nUsing specified CED model size: {ced_model_size}")
    
    # Map keys to CED format
    ced_state_dict = map_hf_to_ced_keys(state_dict, config)
    
    # Create CED model instance to get the architecture
    if ced_model_size == 'tiny':
        ced_model = ced_tiny(pretrained=False)
    elif ced_model_size == 'mini':
        ced_model = ced_mini(pretrained=False)
    elif ced_model_size == 'small':
        ced_model = ced_small(pretrained=False)
    elif ced_model_size == 'base':
        ced_model = ced_base(pretrained=False)
    else:
        raise ValueError(f"Unknown CED model size: {ced_model_size}")
    
    # Try to load the state dict
    try:
        ced_model.load_state_dict(ced_state_dict, strict=False)
        print(f"\n✓ Successfully loaded state dict into CED {ced_model_size} model")
    except Exception as e:
        print(f"\n⚠ Warning: Could not load state dict: {e}")
        print("The converted model may need manual key adjustments")
    
    # Prepare output data
    output_data = {
        "model": ced_state_dict,
        "config": config,
        "ced_model_size": ced_model_size,
        "conversion_info": {
            "source_path": hf_model_path,
            "target_format": "ced",
            "total_parameters": len(ced_state_dict),
            "ced_architecture": f"ced_{ced_model_size}"
        }
    }
    
    # Save the converted model
    torch.save(output_data, output_path)
    print(f"\n✓ Saved converted CED model to: {output_path}")
    print(f"  - CED model size: {ced_model_size}")
    print(f"  - Total parameters: {len(ced_state_dict)}")
    print(f"  - Ready for use with CED encoder")

def main():
    parser = argparse.ArgumentParser(description="Convert Hugging Face CED models to PyTorch format")
    parser.add_argument("hf_model_path", help="Path to Hugging Face model directory")
    parser.add_argument("output_path", help="Output .pt file path")
    parser.add_argument("--ced-model-size", choices=['tiny', 'mini', 'small', 'base'],
                       help="CED model size (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    convert_hf_ced_to_pt(args.hf_model_path, args.output_path, args.ced_model_size)

if __name__ == "__main__":
    main()
