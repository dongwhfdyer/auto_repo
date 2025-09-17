The input to the model is the audio, and the output is a set of extracted features. The tensor shape changes significantly as the data moves through the feature extraction pipeline.

### Input Tensor

The initial input is a mono `.wav` file. The audio data is read into a PyTorch tensor, and its shape is `[T]`, where `T` is the number of audio samples. For example, a 10-second clip at a 16 kHz sampling rate would have $10 \text{ sec} \times 16000 \text{ samples/sec} = 160000$ samples, so the tensor shape would be `[160000]`.

After the audio is converted to a mel-spectrogram, the shape changes to `[1, num_frames, num_mel_bins]`. The code uses `num_mel_bins=128`, and for a 10-second clip, `num_frames` is approximately 1000. So, the shape becomes `[1, 1000, 128]`.

This mel-spectrogram is then padded or truncated to the `target_length` and an extra dimension is added, making the final input tensor to the model `[1, 1, target_length, 128]`. The first `1` is the batch size, and the second `1` is the number of channels.

---

### Output Tensor

The output of the model, after processing the input tensor, has the shape `[1, S, D]`, where:
* `1` is the batch size.
* `S` is the sequence length, which is the number of feature frames. This is a downsampled version of `target_length`. The code notes that the CNN encoder uses a 16x16 kernel with a stride of 16. This downsamples the sequence length by a factor of 16. For `target_length = 1024`, the sequence length becomes $1024 / 16 = 64$.
* `D` is the feature dimension, which is the embedding size of the model.

The `S` dimension includes a special **CLS token** at the beginning, which represents an utterance-level embedding. The other `S-1` tokens are frame-level features.

Depending on the `granularity` argument, the final output tensor shape will be:
* `all`: `[S, D]` (e.g., `[64, D]`)
* `frame`: `[S-1, D]` (e.g., `[63, D]`)
* `utterance`: `[D]` (e.g., `[D]`)