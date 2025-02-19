import torch

from wavtokenizer.decoder.pretrained import WavTokenizer
from wavtokenizer.encoder.utils import convert_audio


class CustomWavTokenizer:
    WAV_SR = 24000

    def __init__(self, config_path: str, model_path: str, device: str):
        self.wav_model = WavTokenizer.from_pretrained0802(
            config_path, model_path).to(device)
        self.device = device

    def tokenize(self, audio: torch.Tensor, sr: int):
        resampled_audio = convert_audio(
            audio.reshape(1, -1), sr, self.WAV_SR, 1)
        bandwidth_id = torch.tensor([0])
        _, wav_tokens = self.wav_model.encode_infer(
            resampled_audio.to(self.device),
            bandwidth_id=bandwidth_id
        )
        return wav_tokens.flatten()

    def decode(self, wav_tokens: torch.Tensor):
        feats = self.wav_model.codes_to_features(
            wav_tokens.reshape((1, 1, -1)).to(self.device))
        bandwidth_id = torch.tensor([0])
        audio = self.wav_model.decode(
            feats, bandwidth_id=bandwidth_id.to(self.device))
        return audio[0]
