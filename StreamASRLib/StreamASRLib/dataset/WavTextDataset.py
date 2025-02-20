import torch


class WavTextDataset:
    def __init__(self,
                 wav_model,
                 tokenizer,
                 librispeech_dataset,
                 spec_augmentation,
                 eos_token: int,
                 boa_token: int,
                 eoa_token: int,
                 audio_token_threshold: int,
                 drop_tokens: bool = False,
                 drop_token_prob: float = 0.0,
                 void_token: int = 2548,
                 add_text: bool = False
                 ):
        self.wav_model = wav_model
        self.tokenizer = tokenizer
        self.librispeech_dataset = librispeech_dataset
        self.spec_augmentation = spec_augmentation

        self.eos_token = eos_token
        self.boa_token = boa_token
        self.eoa_token = eoa_token
        self.audio_token_threshold = audio_token_threshold

        self.drop_tokens = drop_tokens
        self.drop_token_prob = drop_token_prob
        self.void_token = void_token
        self.add_text = add_text

    def __len__(self):
        return len(self.librispeech_dataset)

    def _drop_tokens(self, wav_tokens: torch.Tensor) -> torch.Tensor:
        if not self._drop_tokens:
            return wav_tokens
        drop_msk = torch.rand(wav_tokens.shape) < self.drop_token_prob
        wav_tokens[drop_msk] = self.void_token
        return wav_tokens

    def __getitem__(self, id):
        audio_item = self.librispeech_dataset[id]
        wav, sr = audio_item['audio'].flatten(), audio_item['sr']
        aug_wav = self.spec_augmentation(wav)
        wav_tokens = self.wav_model.tokenize(aug_wav, sr).cpu()
        aug_wav_tokens = self._drop_tokens(wav_tokens)
        text_tokens = self.tokenizer(
            audio_item['text'], add_special_tokens=False, return_tensors='pt').input_ids.flatten()

        input_ids = torch.concatenate((
            torch.tensor([self.boa_token]),
            self.audio_token_threshold + aug_wav_tokens,
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
