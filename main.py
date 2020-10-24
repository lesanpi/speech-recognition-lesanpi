import torch
import torchaudio
from textTransform import TextTransform

device = "cuda" if torch.cuda.is_available() else "cpu"

textTransform = TextTransform()

train_dataset = torchaudio.datasets.LIBRISPEECH("./data", url="dev-clean", download=True)
test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url="test-clean", download=True)

train_audio_transforms = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

if __name__ == '__main__':
    pass
