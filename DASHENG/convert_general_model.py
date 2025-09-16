import json, torch
import torchaudio.transforms as audio_transforms
from pathlib import Path
import argparse
from typing import Dict, Any, Optional, Tuple

def read_config(model_dir: str) -> Dict[str, Any]:
    """Read model configuration from various formats"""
    model_path = Path(model_dir)
    
    # Try different config file formats
    config_files = [
        "config.json",
        "model_config.json", 
        "config.yaml",
        "model_config.yaml"
    ]
    
    for config_file in config_files:
        config_path = model_path / config_file
        if config_path.exists():
            print(f"Reading config from: {config_path}")
            if config_file.endswith('.json'):
                with open(config_path, "r") as f:
                    return json.load(f)
            elif config_file.endswith('.yaml'):
                import yaml
                with open(config_path, "r") as f:
                    return yaml.safe_load(f)
    
    print("Warning: No config file found, using empty config")
    return {}

def read_state_dict(model_dir: str) -> Dict[str, torch.Tensor]:
    """Read model state dict from various formats"""
    model_path = Path(model_dir)
    
    # Try safetensors format first
    idx_path = model_path / "model.safetensors.index.json"
    if idx_path.exists():
        print("Loading from safetensors shards...")
        from safetensors.torch import load_file
        with open(idx_path, "r") as f:
            index = json.load(f)
        merged = {}
        for shard in {Path(x).name for x in index["weight_map"].values()}:
            shard_path = model_path / shard
            print(f"Loading shard: {shard_path}")
            merged.update(load_file(str(shard_path), device="cpu"))
        return merged
    
    # Try single safetensors file
    st_path = model_path / "model.safetensors"
    if st_path.exists():
        print("Loading from single safetensors file...")
        from safetensors.torch import load_file
        return load_file(str(st_path), device="cpu")
    
    # Try PyTorch binary format
    ptbin = model_path / "pytorch_model.bin"
    if ptbin.exists():
        print("Loading from pytorch_model.bin...")
        return torch.load(str(ptbin), map_location="cpu")
    
    # Try other common PyTorch formats
    for pt_file in ["model.pt", "checkpoint.pt", "weights.pt", "state_dict.pt"]:
        pt_path = model_path / pt_file
        if pt_path.exists():
            print(f"Loading from {pt_file}...")
            checkpoint = torch.load(str(pt_path), map_location="cpu")
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    return checkpoint['model']
                elif 'state_dict' in checkpoint:
                    return checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    return checkpoint['model_state_dict']
                else:
                    return checkpoint
            else:
                return checkpoint
    
    raise FileNotFoundError(f"No model weights found in {model_dir}")

def normalize_keys(state_dict: Dict[str, torch.Tensor], 
                  key_mappings: Optional[Dict[str, str]] = None) -> Dict[str, torch.Tensor]:
    """Normalize state dict keys by removing common prefixes and applying custom mappings"""
    def strip_prefix(k: str) -> str:
        # Common prefixes to remove
        prefixes = ["model.", "encoder.", "module.", "backbone.", "feature_extractor."]
        for pref in prefixes:
            if k.startswith(pref):
                return k[len(pref):]
        return k
    
    normalized = {}
    for k, v in state_dict.items():
        # Apply prefix stripping
        new_key = strip_prefix(k)
        
        # Apply custom key mappings if provided
        if key_mappings and new_key in key_mappings:
            new_key = key_mappings[new_key]
        
        normalized[new_key] = v
    
    return normalized

def analyze_model_structure(state_dict: Dict[str, torch.Tensor], 
                          config: Dict[str, Any]) -> None:
    """Analyze and print model structure information"""
    print("\n" + "="*50)
    print("MODEL STRUCTURE ANALYSIS")
    print("="*50)
    
    print(f"Total parameters: {len(state_dict)}")
    print(f"Total parameter count: {sum(p.numel() for p in state_dict.values()):,}")
    
    # Group keys by component
    components = {}
    for key in state_dict.keys():
        component = key.split('.')[0] if '.' in key else 'root'
        if component not in components:
            components[component] = []
        components[component].append(key)
    
    print(f"\nModel components ({len(components)}):")
    for component, keys in sorted(components.items()):
        print(f"  {component}: {len(keys)} parameters")
        if len(keys) <= 5:  # Show keys for small components
            for key in sorted(keys):
                shape = tuple(state_dict[key].shape)
                print(f"    {key}: {shape}")
    
    # Look for common audio model patterns
    audio_patterns = {
        'spectrogram': [k for k in state_dict.keys() if 'spectrogram' in k.lower()],
        'mel': [k for k in state_dict.keys() if 'mel' in k.lower()],
        'conv': [k for k in state_dict.keys() if 'conv' in k.lower()],
        'transformer': [k for k in state_dict.keys() if any(x in k.lower() for x in ['transformer', 'attention', 'self_attn'])],
        'embedding': [k for k in state_dict.keys() if 'embed' in k.lower()],
        'position': [k for k in state_dict.keys() if 'pos' in k.lower()],
    }
    
    print(f"\nAudio model patterns found:")
    for pattern, keys in audio_patterns.items():
        if keys:
            print(f"  {pattern}: {len(keys)} parameters")
            if len(keys) <= 3:
                for key in keys:
                    print(f"    {key}")

def create_key_mapping_guide(state_dict: Dict[str, torch.Tensor], 
                           target_format: str = "ced") -> Dict[str, str]:
    """Create a key mapping guide for converting to target format"""
    print(f"\n" + "="*50)
    print(f"KEY MAPPING GUIDE FOR {target_format.upper()}")
    print("="*50)
    
    # Common mapping patterns for different target formats
    if target_format.lower() == "ced":
        target_patterns = {
            "front_end": ["spectrogram", "mel", "conv", "feature"],
            "patch_embed": ["patch", "embed", "projection"],
            "blocks": ["transformer", "layer", "block", "attention"],
            "norm": ["norm", "ln", "layer_norm"],
            "outputlayer": ["head", "classifier", "output", "fc"]
        }
    elif target_format.lower() == "dasheng":
        target_patterns = {
            "front_end": ["spectrogram", "mel", "conv", "feature"],
            "patch_embed": ["patch", "embed", "projection"],
            "blocks": ["transformer", "layer", "block", "attention"],
            "norm": ["norm", "ln", "layer_norm"],
            "outputlayer": ["head", "classifier", "output", "fc"]
        }
    else:
        print(f"Unknown target format: {target_format}")
        return {}
    
    mapping = {}
    print("Suggested key mappings:")
    for target_key, patterns in target_patterns.items():
        matching_keys = []
        for key in state_dict.keys():
            if any(pattern in key.lower() for pattern in patterns):
                matching_keys.append(key)
        
        if matching_keys:
            print(f"\n{target_key}:")
            for key in matching_keys[:5]:  # Show first 5 matches
                print(f"  {key} -> {target_key}.{key.split('.')[-1] if '.' in key else 'weight'}")
                # Create a simple mapping
                new_key = f"{target_key}.{key.split('.')[-1] if '.' in key else 'weight'}"
                mapping[key] = new_key
    
    return mapping

def convert_model(model_dir: str, 
                 output_path: str,
                 target_format: str = "general",
                 key_mappings: Optional[Dict[str, str]] = None,
                 analyze: bool = True) -> None:
    """Convert a model to the specified format"""
    print(f"Converting model from {model_dir} to {output_path}")
    print(f"Target format: {target_format}")
    
    # Read model components
    config = read_config(model_dir)
    state_dict = read_state_dict(model_dir)
    
    # Analyze model structure
    if analyze:
        analyze_model_structure(state_dict, config)
    
    # Create key mapping guide
    if not key_mappings:
        key_mappings = create_key_mapping_guide(state_dict, target_format)
    
    # Normalize keys
    normalized_state_dict = normalize_keys(state_dict, key_mappings)
    
    # Prepare output
    output_data = {
        "model": normalized_state_dict,
        "config": config,
        "original_config": config,
        "conversion_info": {
            "source_dir": model_dir,
            "target_format": target_format,
            "key_mappings_applied": key_mappings,
            "total_parameters": len(normalized_state_dict)
        }
    }
    
    # Save converted model
    torch.save(output_data, output_path)
    print(f"\n✓ Saved converted model to: {output_path}")
    print(f"  - {len(normalized_state_dict)} parameters")
    print(f"  - Target format: {target_format}")
    
    if key_mappings:
        print(f"  - Applied {len(key_mappings)} key mappings")

def main():
    parser = argparse.ArgumentParser(description="Convert models to different formats")
    parser.add_argument("model_dir", help="Path to source model directory")
    parser.add_argument("output_path", help="Path for output converted model")
    parser.add_argument("--target-format", default="general", 
                       choices=["general", "ced", "dasheng"],
                       help="Target format for conversion")
    parser.add_argument("--key-mappings", type=str,
                       help="JSON file containing custom key mappings")
    parser.add_argument("--no-analyze", action="store_true",
                       help="Skip model structure analysis")
    
    args = parser.parse_args()
    
    # Load custom key mappings if provided
    key_mappings = None
    if args.key_mappings:
        with open(args.key_mappings, 'r') as f:
            key_mappings = json.load(f)
    
    convert_model(
        model_dir=args.model_dir,
        output_path=args.output_path,
        target_format=args.target_format,
        key_mappings=key_mappings,
        analyze=not args.no_analyze
    )

if __name__ == "__main__":
    main()
