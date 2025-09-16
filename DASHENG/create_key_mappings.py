#!/usr/bin/env python3
"""
Interactive script to create key mappings for model conversion.
This helps you map keys from your source model to the target format.
"""

import json
import argparse
from pathlib import Path
from convert_general_model import read_state_dict, analyze_model_structure

def interactive_key_mapping(state_dict, target_format="ced"):
    """Interactive key mapping creation"""
    print(f"\nCreating key mappings for {target_format.upper()} format")
    print("="*50)
    
    # Common target patterns for different formats
    if target_format.lower() == "ced":
        target_components = [
            "front_end.0.spectrogram",
            "front_end.0.mel_scale", 
            "patch_embed",
            "time_pos_embed",
            "freq_pos_embed",
            "blocks",
            "norm",
            "outputlayer"
        ]
    elif target_format.lower() == "dasheng":
        target_components = [
            "front_end.0.spectrogram",
            "front_end.0.mel_scale",
            "patch_embed", 
            "time_pos_embed",
            "freq_pos_embed",
            "blocks",
            "norm",
            "outputlayer"
        ]
    else:
        target_components = []
    
    mappings = {}
    
    # Group source keys by similarity
    source_keys = list(state_dict.keys())
    print(f"Found {len(source_keys)} parameters in source model")
    
    # Show first few keys as examples
    print("\nFirst 10 source keys:")
    for i, key in enumerate(source_keys[:10]):
        shape = tuple(state_dict[key].shape)
        print(f"  {i+1:2d}. {key:<40} {shape}")
    
    if len(source_keys) > 10:
        print(f"  ... and {len(source_keys) - 10} more")
    
    print(f"\nTarget components for {target_format.upper()}:")
    for i, component in enumerate(target_components):
        print(f"  {i+1:2d}. {component}")
    
    print("\n" + "="*50)
    print("KEY MAPPING INSTRUCTIONS:")
    print("1. For each source key, specify which target component it should map to")
    print("2. You can use component numbers (1, 2, 3...) or component names")
    print("3. Type 'skip' to skip a key")
    print("4. Type 'auto' to use automatic mapping")
    print("5. Type 'quit' to finish")
    print("="*50)
    
    # Auto-mapping based on keywords
    auto_mappings = {}
    for key in source_keys:
        key_lower = key.lower()
        if any(x in key_lower for x in ['spectrogram', 'mel', 'conv']):
            auto_mappings[key] = "front_end"
        elif any(x in key_lower for x in ['patch', 'embed']):
            auto_mappings[key] = "patch_embed"
        elif any(x in key_lower for x in ['pos', 'position']):
            if 'time' in key_lower:
                auto_mappings[key] = "time_pos_embed"
            elif 'freq' in key_lower:
                auto_mappings[key] = "freq_pos_embed"
        elif any(x in key_lower for x in ['transformer', 'attention', 'block']):
            auto_mappings[key] = "blocks"
        elif any(x in key_lower for x in ['norm', 'ln']):
            auto_mappings[key] = "norm"
        elif any(x in key_lower for x in ['head', 'classifier', 'output', 'fc']):
            auto_mappings[key] = "outputlayer"
    
    print(f"\nAuto-detected {len(auto_mappings)} mappings:")
    for key, target in list(auto_mappings.items())[:10]:
        print(f"  {key} -> {target}")
    if len(auto_mappings) > 10:
        print(f"  ... and {len(auto_mappings) - 10} more")
    
    # Interactive mapping
    print(f"\nStarting interactive mapping...")
    print("Type 'auto' to use all auto-mappings, or map manually")
    
    choice = input("\nUse auto-mappings? (y/n/auto): ").lower().strip()
    
    if choice in ['y', 'yes', 'auto']:
        mappings = auto_mappings
        print(f"Using {len(mappings)} auto-mappings")
    else:
        # Manual mapping for unmatched keys
        unmatched_keys = [k for k in source_keys if k not in auto_mappings]
        print(f"\nManually mapping {len(unmatched_keys)} unmatched keys...")
        
        for key in unmatched_keys:
            shape = tuple(state_dict[key].shape)
            print(f"\nKey: {key} {shape}")
            print("Target components:")
            for i, component in enumerate(target_components):
                print(f"  {i+1}. {component}")
            
            while True:
                response = input("Map to (number/name/skip/quit): ").strip()
                
                if response.lower() in ['quit', 'q']:
                    return mappings
                elif response.lower() in ['skip', 's']:
                    break
                elif response.isdigit():
                    idx = int(response) - 1
                    if 0 <= idx < len(target_components):
                        mappings[key] = target_components[idx]
                        break
                    else:
                        print("Invalid number")
                elif response in target_components:
                    mappings[key] = response
                    break
                else:
                    print("Invalid input")
    
    return mappings

def create_mapping_file(model_dir, output_file, target_format="ced"):
    """Create a key mapping file for model conversion"""
    print(f"Analyzing model in: {model_dir}")
    
    # Read model state dict
    try:
        state_dict = read_state_dict(model_dir)
    except Exception as e:
        print(f"Error reading model: {e}")
        return
    
    # Analyze model structure
    analyze_model_structure(state_dict, {})
    
    # Create mappings
    mappings = interactive_key_mapping(state_dict, target_format)
    
    # Save mappings
    mapping_data = {
        "target_format": target_format,
        "source_model": str(model_dir),
        "mappings": mappings,
        "total_mappings": len(mappings)
    }
    
    with open(output_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"\n✓ Saved key mappings to: {output_file}")
    print(f"  - {len(mappings)} key mappings created")
    print(f"  - Target format: {target_format}")
    
    # Show summary
    if mappings:
        print(f"\nMapping summary:")
        target_counts = {}
        for source_key, target_key in mappings.items():
            component = target_key.split('.')[0]
            target_counts[component] = target_counts.get(component, 0) + 1
        
        for component, count in sorted(target_counts.items()):
            print(f"  {component}: {count} parameters")

def main():
    parser = argparse.ArgumentParser(description="Create key mappings for model conversion")
    parser.add_argument("model_dir", help="Path to source model directory")
    parser.add_argument("output_file", help="Output JSON file for key mappings")
    parser.add_argument("--target-format", default="ced",
                       choices=["ced", "dasheng", "general"],
                       help="Target format for conversion")
    
    args = parser.parse_args()
    
    create_mapping_file(args.model_dir, args.output_file, args.target_format)

if __name__ == "__main__":
    main()
