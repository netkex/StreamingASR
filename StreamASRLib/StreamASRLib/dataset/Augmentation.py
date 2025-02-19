import torch
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset


class DummyEffector:
    def __init__(self):
        pass

    def apply(self, wav, sr):
        return wav


class AudioAugmentation:
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
            return DummyEffector()
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


class WavTextDataset:
    def __init__(self,
                 wav_model,
                 tokenizer,
                 librispeech_dataset,
                 augmentator,
                 eos_token: int,
                 boa_token: int,
                 eoa_token: int,
                 audio_token_threshold: int,
                 void_token_prob: float = 0.0,
                 void_token: int = 2548,
                 add_text: bool = False
                 ):
        self.wav_model = wav_model
        self.tokenizer = tokenizer
        self.librispeech_dataset = librispeech_dataset
        self.augmentator = augmentator
        self.eos_token = eos_token
        self.boa_token = boa_token
        self.eoa_token = eoa_token
        self.audio_token_threshold = audio_token_threshold
        self.void_token_prob = void_token_prob
        self.void_token = void_token
        self.add_text = add_text

    def __len__(self):
        return len(self.librispeech_dataset)

    def __item__(self, id):
        audio_item = self.librispeech_dataset[id]
        wav, sr = audio_item['audio'].flatten(), audio_item['sr']
        aug_wav = self.augmentator(wav, sr)
        wav_tokens = self.wav_model.tokenize(aug_wav, sr)
        aug_wav_tokens = self._drop_tokens(wav_tokens)
        text_tokens = self.tokenizer(
            audio_item['text'], add_special_tokens=False).input_ids.flatten()

        input_ids = torch.concatenate((
            torch.tensor([self.boa_token]),
            aug_wav_tokens,
            torch.tensor([self.eoa_token]),
            text_tokens,
            torch.tensor([self.eos_token])
        ), 0)
        attention_mask = torch.ones_like(input_ids)
        text_start = 2 + aug_wav_tokens.shape[0]
        text_end = input_ids.shape[0]
        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_text_start': text_start,
            'label_text_end': text_end
        }
        if self.add_text:
            res['text'] = audio_item['text']
        return res
