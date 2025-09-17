#!/usr/bin/env python3
"""
Test script to understand why torchaudio MelSpectrogram keys are missing
"""
import torch
import torchaudio.transforms as audio_transforms

def test_melspectrogram_keys():
    """Test what keys MelSpectrogram actually has in its state_dict"""
    print("Testing torchaudio.transforms.MelSpectrogram state_dict...")
    
    # Create a MelSpectrogram transform
    mel_spec = audio_transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=160,
        n_mels=128,
        f_min=0,
        f_max=8000,
        center=True
    )
    
    # Get its state dict
    state_dict = mel_spec.state_dict()
    print(f"MelSpectrogram state_dict keys: {list(state_dict.keys())}")
    
    # Check if it has the expected keys
    expected_keys = ["window", "mel_scale.fb"]
    for key in expected_keys:
        if key in state_dict:
            print(f"✓ Found key: {key}")
        else:
            print(f"✗ Missing key: {key}")
    
    # Let's also check what attributes the MelSpectrogram has
    print(f"\nMelSpectrogram attributes:")
    for attr_name in dir(mel_spec):
        if not attr_name.startswith('_'):
            attr_value = getattr(mel_spec, attr_name)
            if not callable(attr_value):
                print(f"  {attr_name}: {type(attr_value)} = {attr_value}")

def test_dasheng_model_keys():
    """Test what keys dasheng model actually expects"""
    print("\n" + "="*50)
    print("Testing dasheng model state_dict...")
    
    try:
        from dasheng import dasheng_base
        model = dasheng_base()
        
        # Get state dict
        state_dict = model.state_dict()
        
        # Look for frontend keys
        frontend_keys = [k for k in state_dict.keys() if 'front_end' in k]
        print(f"Frontend keys in dasheng model: {frontend_keys}")
        
        # Check specifically for the missing keys
        missing_keys = ["front_end.0.spectrogram.window", "front_end.0.mel_scale.fb"]
        for key in missing_keys:
            if key in state_dict:
                print(f"✓ Found key: {key}")
                print(f"  Shape: {state_dict[key].shape}")
            else:
                print(f"✗ Missing key: {key}")
        
        # Let's see what the front_end actually contains
        print(f"\nFront-end module structure:")
        print(f"front_end type: {type(model.front_end)}")
        print(f"front_end[0] type: {type(model.front_end[0])}")
        
        # Check if front_end[0] has state_dict
        if hasattr(model.front_end[0], 'state_dict'):
            frontend_state = model.front_end[0].state_dict()
            print(f"front_end[0] state_dict keys: {list(frontend_state.keys())}")
        
    except Exception as e:
        print(f"Error testing dasheng model: {e}")

def test_loading_with_strict_false():
    """Test loading with strict=False to see what happens"""
    print("\n" + "="*50)
    print("Testing loading with strict=False...")
    
    try:
        from dasheng import dasheng_base
        
        # Create a model
        model = dasheng_base()
        
        # Create a dummy state dict missing some keys
        dummy_state = {}
        for key, value in model.state_dict().items():
            if 'front_end.0.spectrogram.window' not in key and 'front_end.0.mel_scale.fb' not in key:
                dummy_state[key] = value
        
        print(f"Loading with {len(dummy_state)} keys (missing frontend keys)...")
        
        # Try loading with strict=False
        missing_keys, unexpected_keys = model.load_state_dict(dummy_state, strict=False)
        
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        
    except Exception as e:
        print(f"Error testing strict=False: {e}")

if __name__ == "__main__":
    test_melspectrogram_keys()
    test_dasheng_model_keys()
    test_loading_with_strict_false()


