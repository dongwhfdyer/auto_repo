#!/usr/bin/env python3
"""
Simple test script for MultiAudioDataset
Usage: python EAT/test_multi_dataset.py --old-data /path/to/old/data --new-data /path/to/new/data
"""

import argparse
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.multi_audio_dataset import MultiAudioDataset


def test_multi_dataset(old_data_path, new_data_path):
    """Test the MultiAudioDataset implementation."""
    print("=== Testing MultiAudioDataset ===")
    print(f"Old data path: {old_data_path}")
    print(f"New data path: {new_data_path}")
    
    try:
        # Create dataset
        dataset = MultiAudioDataset(
            old_data_path=old_data_path,
            new_data_path=new_data_path,
            split="train",
            audio_mae=True,
            h5_format=False,
            downsr_16hz=True,
            target_length=1024,
            flexible_mask=False,
            input_size=(1024, 128),
            key="source",
            compute_mask=False
        )
        
        print(f"✓ Dataset created successfully")
        print(f"✓ Total samples: {len(dataset)}")
        
        # Test epoch progression
        print("\n=== Testing Epoch Progression ===")
        for epoch in [1, 5, 6, 10, 11, 15, 16]:
            dataset.set_epoch(epoch)
            stats = dataset.get_sampling_stats()
            print(f"Epoch {epoch}: Distribution = {stats['distribution']} (old:new)")
        
        # Test sample access
        print("\n=== Testing Sample Access ===")
        sample = dataset[0]
        print(f"✓ Sample keys: {list(sample.keys())}")
        if 'source' in sample:
            print(f"✓ Sample source shape: {sample['source'].shape}")
        
        print("\n=== All Tests Passed! ===")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test MultiAudioDataset")
    parser.add_argument("--old-data", required=True, help="Path to old data directory")
    parser.add_argument("--new-data", required=True, help="Path to new data directory")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isdir(args.old_data):
        print(f"Error: Old data directory does not exist: {args.old_data}")
        return 1
    
    if not os.path.isdir(args.new_data):
        print(f"Error: New data directory does not exist: {args.new_data}")
        return 1
    
    # Run test
    success = test_multi_dataset(args.old_data, args.new_data)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 