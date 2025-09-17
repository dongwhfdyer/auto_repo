import torch
from dasheng.pretrained.pretrained import Dasheng


class LocalDashengEncoder(torch.nn.Module):
    def __init__(self, checkpoint_path: str, max_seconds: int = 10):
        super().__init__()
        self.sampling_rate = 16000
        self.hop_size_in_ms = 40
        self.max_length = int(max_seconds * self.sampling_rate)
        self.model = Dasheng.from_pretrained(checkpoint_path)
        # Infer embed_dim from the loaded model to set output_dim
        self.output_dim = getattr(self.model, "embed_dim", 768)

    def forward(self, audio: torch.Tensor):
        assert isinstance(audio, torch.Tensor)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        self.model.eval()
        with torch.inference_mode():
            if audio.shape[-1] > self.max_length:
                output = []
                for chunk in audio.split(self.max_length, dim=-1):
                    if chunk.shape[-1] < self.sampling_rate:
                        chunk = torch.nn.functional.pad(
                            chunk, (0, self.sampling_rate - chunk.shape[-1])
                        )
                    tmp_output = self.model(chunk)
                    output.append(tmp_output)
                output = torch.cat(output, dim=1)
            else:
                output = self.model(audio)
        return output


if __name__ == "__main__":
    from xares.audio_encoder_checker import check_audio_encoder
    import sys

    assert len(sys.argv) == 2, "Usage: python dasheng_local_encoder.py /absolute/path/to/checkpoint.pt"
    encoder = LocalDashengEncoder(sys.argv[1])
    assert check_audio_encoder(encoder)
    print("Encoder check passed.")
