import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from StreamASRLib.dataset import RawLibriSpeechPT
from StreamASRLib.model.config import DEVICE
from StreamASRLib.eval.eval import calculate_wer


class WhisperDataset(torch.utils.data.Dataset):
    def __init__(self, raw_dataset: RawLibriSpeechPT, device=DEVICE):
        self.raw_dataset = raw_dataset
        self.processor = WhisperProcessor.from_pretrained(
            "openai/whisper-tiny.en")
        self.device = device

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, item):
        raw_item = self.raw_dataset[item]
        audio, text, sample_rate = raw_item['audio'], raw_item['text'], raw_item['sample_rate']
        audio_pt = self.processor(
            audio.numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            device=self.device
        ).input_features[0].to(self.device)
        return {
            'audio': audio_pt,
            'text': text,
            'sample_rate': sample_rate
        }


class WhisperModel():
    def __init__(self, model='openai/whisper-tiny.en', device=DEVICE):
        self.model = WhisperForConditionalGeneration\
            .from_pretrained("openai/whisper-tiny.en")\
            .to(device)
        self.processor = WhisperProcessor.from_pretrained(
            "openai/whisper-tiny.en")

    def generate(self, batch):
        predicted_ids = self.model.generate(batch)
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)


if __name__ == "__main__":
    raw_dataset = RawLibriSpeechPT(device='cpu')
    whisper_dataset = WhisperDataset(raw_dataset)
    dataloader = torch.utils.data.DataLoader(whisper_dataset, batch_size=64)
    wmodel = WhisperModel()
    wer = calculate_wer(wmodel, dataloader)
    print(f'WER: {wer}')
