import torch
import sys
import os

# Add the CED directory to the path so we can import the models
# TODO: Modify this path to point to your CED directory
sys.path.append('/data1/EAT_projs/CED')

from models.audiotransformer import ced_base, ced_small, ced_mini, ced_tiny


class CedEncoder(torch.nn.Module):
    def __init__(self, model_size='mini', checkpoint_path=None):
        super().__init__()
        self.sampling_rate = 16000
        self.hop_size_in_ms = 40  # Based on the model's hop_size=160 and sampling_rate=16000
        
        # Load the CED model based on size
        if model_size == 'tiny':
            self.model = ced_tiny(pretrained=checkpoint_path is None)
            self.output_dim = 192
        elif model_size == 'mini':
            self.model = ced_mini(pretrained=checkpoint_path is None)
            self.output_dim = 256
        elif model_size == 'small':
            self.model = ced_small(pretrained=checkpoint_path is None)
            self.output_dim = 384
        elif model_size == 'base':
            self.model = ced_base(pretrained=checkpoint_path is None)
            self.output_dim = 768
        else:
            raise ValueError(f"Unknown model size: {model_size}. Choose from: tiny, mini, small, base")
        
        # Load custom checkpoint if provided
        if checkpoint_path is not None:
            print(f"Loading CED {model_size} model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                # Standard format with 'model' key
                self.model.load_state_dict(checkpoint['model'], strict=False)
                print(f"Loaded model from checkpoint['model']")
            elif 'state_dict' in checkpoint:
                # Format with 'state_dict' key
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                print(f"Loaded model from checkpoint['state_dict']")
            elif 'model_state_dict' in checkpoint:
                # Format with 'model_state_dict' key
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print(f"Loaded model from checkpoint['model_state_dict']")
            else:
                # Direct state dict format
                self.model.load_state_dict(checkpoint, strict=False)
                print(f"Loaded model from direct state dict")
            
            # Print conversion info if available
            if 'conversion_info' in checkpoint:
                info = checkpoint['conversion_info']
                print(f"Conversion info: {info.get('ced_architecture', 'unknown')} from {info.get('source_path', 'unknown')}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Calculate max length for chunking (10 seconds)
        self.max_length = int(10 * self.sampling_rate)

    def forward(self, audio: torch.Tensor):
        assert isinstance(audio, torch.Tensor)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        
        batch_size = audio.shape[0]
        
        with torch.inference_mode():
            if audio.shape[-1] > self.max_length:
                # Handle long audio by chunking
                outputs = []
                for chunk in audio.split(self.max_length, dim=-1):
                    if chunk.shape[-1] < self.sampling_rate:
                        # Pad short chunks to at least 1 second
                        chunk = torch.nn.functional.pad(chunk, (0, self.sampling_rate - chunk.shape[-1]))
                    
                    # Get features from the model (before the final classification layer)
                    chunk_output = self.model.forward_features(self.model.front_end(chunk))
                    outputs.append(chunk_output)
                
                # Concatenate all chunks
                output = torch.cat(outputs, dim=1)
            else:
                # Handle short audio
                if audio.shape[-1] < self.sampling_rate:
                    # Pad short audio to at least 1 second
                    audio = torch.nn.functional.pad(audio, (0, self.sampling_rate - audio.shape[-1]))
                
                # Get features from the model (before the final classification layer)
                output = self.model.forward_features(self.model.front_end(audio))
        
        return output


if __name__ == "__main__":
    from xares.audio_encoder_checker import check_audio_encoder

    # Test with different model sizes
    for size in ['tiny', 'mini', 'small', 'base']:
        print(f"Testing CED {size} encoder...")
        encoder = CedEncoder(model_size=size)
        if check_audio_encoder(encoder):
            print(f"✓ CED {size} encoder passed validation")
        else:
            print(f"✗ CED {size} encoder failed validation")
        print()
    
    # Example of using custom checkpoint path
    # encoder = CedEncoder(model_size='mini', checkpoint_path='/path/to/your/ced_model.pt')
