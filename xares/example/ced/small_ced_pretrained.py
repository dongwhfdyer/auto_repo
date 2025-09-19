import torch
import sys
import os

sys.path.append('/data1/repos/EAT_projs/CED')

from models.audiotransformer import ced_small


class CedEncoder(torch.nn.Module):
    def __init__(self, checkpoint_path: str | None = None):
        super().__init__()
        self.sampling_rate = 16000
        self.hop_size_in_ms = 10

        # Default checkpoint for small (updated path)
        default_ckpt = '/data1/repos/EAT_projs/checkpoints/weights/audiotransformer_small_mae_as_10s.pt'
        use_ckpt = checkpoint_path or (default_ckpt if os.path.isfile(default_ckpt) else None)

        self.model = ced_small(pretrained=use_ckpt is None)
        self.output_dim = 384

        if use_ckpt is not None:
            print(f"Loading CED small model from: {use_ckpt}")
            checkpoint = torch.load(use_ckpt, map_location='cpu')

            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'], strict=False)
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)

            if 'conversion_info' in checkpoint:
                info = checkpoint['conversion_info']
                print(f"Conversion info: {info.get('ced_architecture', 'unknown')} from {info.get('source_path', 'unknown')}")

        self.model.eval()
        self.max_length = int(10 * self.sampling_rate)

    def to(self, device):
        super().to(device)
        self.model = self.model.to(device)
        return self

    def forward(self, audio: torch.Tensor):
        assert isinstance(audio, torch.Tensor)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        device = next(self.model.parameters()).device
        audio = audio.to(device)

        with torch.inference_mode():
            if audio.shape[-1] > self.max_length:
                outputs = []
                for chunk in audio.split(self.max_length, dim=-1):
                    if chunk.shape[-1] < self.sampling_rate:
                        chunk = torch.nn.functional.pad(chunk, (0, self.sampling_rate - chunk.shape[-1]))

                    spectrogram = self.model.front_end(chunk)
                    spectrogram = spectrogram.unsqueeze(1)
                    spectrogram = self.model.init_bn(spectrogram)
                    chunk_output = self.model.forward_features(spectrogram)
                    outputs.append(chunk_output)

                output = torch.cat(outputs, dim=1)
            else:
                if audio.shape[-1] < self.sampling_rate:
                    audio = torch.nn.functional.pad(audio, (0, self.sampling_rate - audio.shape[-1]))

                spectrogram = self.model.front_end(audio)
                spectrogram = spectrogram.unsqueeze(1)
                spectrogram = self.model.init_bn(spectrogram)
                output = self.model.forward_features(spectrogram)

        return output


if __name__ == "__main__":
    from xares.audio_encoder_checker import check_audio_encoder

    print("Testing CED small encoder...")
    encoder = CedEncoder()
    if check_audio_encoder(encoder):
        print("✓ CED small encoder passed validation")
    else:
        print("✗ CED small encoder failed validation")