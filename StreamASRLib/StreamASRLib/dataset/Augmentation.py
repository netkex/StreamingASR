import torch
import numpy as np
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT
from torch.utils.data import Dataset
from dataclasses import dataclass


def coin(p: float) -> bool:
    return torch.rand(1).item() < p


class FFMPEGAugmentations:
    class DummyEffector:
        def __init__(self):
            pass

        def apply(self, wav, sr):
            return wav

    # echo effect augmentation very bad with wav tokens
    def __init__(self,
                 apply_atempo_p: float = 0.0, atempo_min: float = 0.8, atempo_max: float = 1.2,
                 apply_aecho_p: float = 0.0, aecho_in_g: float = 0.8, aecho_out_g: float = 0.1,
                 aecho_delays: float = 50, aecho_decayes: float = 0.15,
                 apply_noise_p: float = 0.0, noise_std: float = 0.01, noise_snr: float = 30):
        self.apply_atempo_p, self.atempo_min, self.atempo_max = apply_atempo_p, atempo_min, atempo_max

        self.apply_aecho_p, self.aecho_in_g, self.aecho_out_g = apply_aecho_p, aecho_in_g, aecho_out_g
        self.aecho_delays, self.aecho_decayes = aecho_delays, aecho_decayes

        self.apply_noise_p, self.noise_std, self.noise_snr = apply_noise_p, noise_std, noise_snr

    def __call__(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        effector = self._build_ffmpeg_effects()
        augmented_wav = effector.apply(wav.reshape((-1, 1)), sr).reshape(-1)
        noise_augmented_wav = self._add_noise(augmented_wav)
        return noise_augmented_wav

    def _build_ffmpeg_effects(self):
        effects_list = []

        attempo_effect = self._atempo_effect()
        if attempo_effect is not None:
            effects_list.append(attempo_effect)

        aecho_effect = self._aecho_effect()
        if aecho_effect is not None:
            effects_list.append(aecho_effect)

        effect = ','.join(effects_list)
        if effect == '':
            return FFMPEGAugmentations.DummyEffector()
        print(f'Effect: {effect}')
        return torchaudio.io.AudioEffector(effect=effect)

    def _atempo_effect(self):
        if torch.rand(1).item() > self.apply_atempo_p:
            return None
        tempo = self.atempo_min + \
            torch.rand(1).item() * (self.atempo_max - self.atempo_min)
        return f'atempo={tempo:.2f}'

    def _aecho_effect(self):
        if torch.rand(1).item() > self.apply_aecho_p:
            return None
        return f'aecho={self.aecho_in_g:0.2f}:{self.aecho_out_g:0.2f}:{self.aecho_delays:0.2f}:{self.aecho_decayes:0.2f}'

    def _add_noise(self, wav):
        if torch.rand(1).item() > self.apply_noise_p:
            return wav
        noise = torch.normal(0, self.noise_std, (1, wav.shape[0]))
        return AF.add_noise(wav.reshape((1, -1)), noise, torch.tensor([self.noise_snr])).reshape(-1)


class WhiteNoise:
    def __init__(self, noise_std: float = 0.01, noise_snr: float = 35):
        self.noise_std = noise_std
        self.noise_snr = noise_snr

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        noise = torch.normal(0, self.noise_std, (1, wav.nelement()))
        return AF.add_noise(wav.reshape((1, -1)), noise, torch.tensor([self.noise_snr])).reshape(-1)


class SpeedUpDown:
    def __call__(self, spec: torch.Tensor, speed_up_factor: float = 1.0) -> torch.Tensor:
        freq_num = int(speed_up_factor * spec.shape[-1])
        ind = np.round(np.linspace(
            0, spec.shape[-1] - 1, freq_num)).astype(int)
        return spec[..., :, ind]


class PitchUpDown:
    # TODO: fix. At the moment, works quit badly
    def _pitch_up(self, spec: torch.Tensor, pitch_factor: int) -> torch.Tensor:
        spec_pitched = torch.zeros_like(spec, dtype=spec.dtype)
        final_freq = spec.shape[-2] - pitch_factor
        spec_pitched[..., pitch_factor:, :] = spec[..., :final_freq, :]
        return spec_pitched

    def _pitch_down(self, spec: torch.Tensor, pitch_factor: int) -> torch.Tensor:
        spec_pitched = torch.zeros_like(spec, dtype=spec.dtype)
        final_freq = spec.shape[-2] - pitch_factor
        spec_pitched[..., :final_freq, :] = spec[..., pitch_factor:, :]
        return spec_pitched

    def __call__(self, spec: torch.Tensor, pith_factor: int = 0) -> torch.Tensor:
        if pith_factor == 0:
            return spec
        return self._pitch_up(spec, pith_factor) if pith_factor > 0 else self._pitch_down(spec, -pith_factor)


class FrequencyMask:
    def __init__(self, freq_mask_param: int):
        self.freq_mask_param = freq_mask_param

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        start = torch.randint(
            spec.shape[-2] - self.freq_mask_param, size=(1,)).item()
        spec[..., start:start + self.freq_mask_param, :] = 0
        return spec


@dataclass
class SpecAugmentationConfig:
    apply_aug_p: float = 0.0

    apply_speed: bool = False
    speed_p: float = 0.0
    min_speed_up: float = 0.75
    max_speed_up: float = 1.25

    apply_masking: bool = False
    masking_p: float = 0.0
    freq_mask_param: int = 100

    apply_noise: bool = False
    noise_p: float = 0.0
    noise_std: float = 0.01
    noise_snr: float = 35

    # apply_pitch: bool = False
    # pitch_p: float = 0.0
    # min_pitch: int = -1
    # max_pitch: int = 1


class SpecAugmenation:
    speed_transform = SpeedUpDown()

    def __init__(self,
                 n_fft=1024,
                 config: SpecAugmentationConfig = SpecAugmentationConfig()):
        self.n_fft = n_fft
        self.config = config

        self.wav_to_spec = AT.Spectrogram(n_fft=n_fft, power=None)
        self.spec_to_wav = AT.InverseSpectrogram(n_fft=n_fft)
        self.mask_transform = FrequencyMask(config.freq_mask_param)
        self.noise = WhiteNoise(config.noise_std, config.noise_snr)

    def _noise(self, wav: torch.Tensor) -> torch.Tensor:
        if not self.config.apply_noise or not coin(self.config.noise_p):
            return wav
        print('Applied noise')
        return self.noise(wav)

    def _speedup(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.config.apply_speed or not coin(self.config.speed_p):
            return spec
        speed_coef = self.config.min_speed_up + \
            torch.rand(1).item() * (self.config.max_speed_up -
                                    self.config.min_speed_up)
        print(f'Applied speedup {speed_coef}')
        return self.speed_transform(spec, speed_coef)

    def _mask(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.config.apply_masking or not coin(self.config.masking_p):
            return spec
        print('Applied masking')
        return self.mask_transform(spec)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if not coin(self.config.apply_aug_p):
            return wav
        noise_wav = self._noise(wav)
        spec = self.wav_to_spec(noise_wav)
        aug_spec = self._speedup(spec)
        aug_spec = self._mask(aug_spec)
        return self.spec_to_wav(aug_spec).reshape(-1)
