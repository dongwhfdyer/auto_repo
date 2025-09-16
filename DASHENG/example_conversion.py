#!/usr/bin/env python3
"""
Example usage of the general model conversion scripts.
"""

from convert_general_model import convert_model
import json

def example_conversion():
    """Example of how to use the conversion scripts"""
    
    # Example 1: Basic conversion with analysis
    print("Example 1: Basic conversion")
    print("-" * 30)
    
    model_dir = "/path/to/your/source/model"
    output_path = "/path/to/converted_model.pt"
    
    try:
        convert_model(
            model_dir=model_dir,
            output_path=output_path,
            target_format="ced",
            analyze=True
        )
    except Exception as e:
        print(f"Conversion failed: {e}")
        print("Make sure the model directory path is correct")
    
    # Example 2: Conversion with custom key mappings
    print("\nExample 2: Conversion with custom mappings")
    print("-" * 40)
    
    # Custom key mappings (example)
    custom_mappings = {
        "encoder.spectrogram.weight": "front_end.0.spectrogram.weight",
        "encoder.mel_scale.fb": "front_end.0.mel_scale.fb",
        "transformer.layers": "blocks",
        "classifier.weight": "outputlayer.1.weight",
        "classifier.bias": "outputlayer.1.bias"
    }
    
    try:
        convert_model(
            model_dir=model_dir,
            output_path=output_path.replace(".pt", "_custom.pt"),
            target_format="ced",
            key_mappings=custom_mappings,
            analyze=True
        )
    except Exception as e:
        print(f"Custom conversion failed: {e}")
    
    # Example 3: Save mappings to file for reuse
    print("\nExample 3: Save mappings to file")
    print("-" * 35)
    
    mappings_file = "my_model_mappings.json"
    mapping_data = {
        "target_format": "ced",
        "source_model": model_dir,
        "mappings": custom_mappings,
        "total_mappings": len(custom_mappings)
    }
    
    with open(mappings_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"Saved mappings to: {mappings_file}")
    print("You can now use this file with: --key-mappings my_model_mappings.json")

def show_usage_examples():
    """Show command line usage examples"""
    print("\n" + "="*60)
    print("COMMAND LINE USAGE EXAMPLES")
    print("="*60)
    
    examples = [
        {
            "description": "Basic conversion with analysis",
            "command": "python convert_general_model.py /path/to/model /path/to/output.pt --target-format ced"
        },
        {
            "description": "Conversion with custom key mappings",
            "command": "python convert_general_model.py /path/to/model /path/to/output.pt --target-format ced --key-mappings my_mappings.json"
        },
        {
            "description": "Quick conversion without analysis",
            "command": "python convert_general_model.py /path/to/model /path/to/output.pt --target-format ced --no-analyze"
        },
        {
            "description": "Create key mappings interactively",
            "command": "python create_key_mappings.py /path/to/model my_mappings.json --target-format ced"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}:")
        print(f"   {example['command']}")

def show_supported_formats():
    """Show supported model formats and target formats"""
    print("\n" + "="*60)
    print("SUPPORTED FORMATS")
    print("="*60)
    
    print("\nInput formats (source models):")
    print("  - Hugging Face models (config.json + model files)")
    print("  - PyTorch models (.pt, .pth files)")
    print("  - SafeTensors models (.safetensors)")
    print("  - Model directories with multiple files")
    
    print("\nTarget formats:")
    print("  - ced: CED AudioTransformer format")
    print("  - dasheng: Dasheng format")
    print("  - general: Generic format (minimal processing)")
    
    print("\nKey mapping features:")
    print("  - Automatic prefix removal (model., encoder., module.)")
    print("  - Custom key mappings via JSON file")
    print("  - Interactive mapping creation")
    print("  - Pattern-based auto-mapping")

if __name__ == "__main__":
    print("General Model Conversion Tool - Examples")
    print("="*50)
    
    show_supported_formats()
    show_usage_examples()
    
    print("\n" + "="*60)
    print("PROGRAMMATIC USAGE")
    print("="*60)
    
    # Run the example (will fail with placeholder paths, but shows the structure)
    example_conversion()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Update the model_dir path in the examples above")
    print("2. Run: python create_key_mappings.py /path/to/model mappings.json")
    print("3. Review and edit the generated mappings.json if needed")
    print("4. Run: python convert_general_model.py /path/to/model output.pt --key-mappings mappings.json")
    print("5. Use the converted model in your CED encoder script")
