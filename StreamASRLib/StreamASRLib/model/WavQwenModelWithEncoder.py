import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from StreamASRLib.model import WavQwenBaseModel, WavTokensEncoder


class EmbeddingWithEncoder(nn.Module):
    '''
    Custom embedding class for wav-tokens with encoder
    '''

    def __init__(self, base_emb: nn.Embedding, wav_encoder: nn.Module, ext_token_threshold: int):
        super().__init__()
        self.base_emb = base_emb
        self.hidden_size = base_emb.embedding_dim
        self.wav_encoder = wav_encoder
        self.ext_token_threshold = ext_token_threshold

    def freeze_base_emb(self):
        for param in self.base_emb.parameters():
            param.requires_grad = False

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        ext_token_mask = (input_tokens >= self.ext_token_threshold)
        inp_emb = torch.zeros(list(
            input_tokens.shape) + [self.hidden_size], requires_grad=True).to(self.base_emb.weight.device)
        inp_emb[~ext_token_mask] = self.base_emb(input_tokens[~ext_token_mask])

        wav_emb = []
        for i in range(input_tokens.shape[0]):
            wav_input = input_tokens[i][ext_token_mask[i]].reshape(
                1, -1) - self.ext_token_threshold
            wav_emb.append(self.wav_encoder(
                wav_input).reshape(-1, self.hidden_size))
        inp_emb[ext_token_mask] = torch.concatenate(wav_emb, dim=0)
        return inp_emb


class WavQwenModelWithEncoder(WavQwenBaseModel):
    def __init__(self, model, fast_tokenizer=True):
        super().__init__()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            WavQwenModelWithEncoder.QWEN_REPO, use_fast=fast_tokenizer)

    def tokenizer(self):
        return self.tokenizer

    def model(self):
        return self.model

    def freeze_qwen(self):
        for parameter in self.model.model.parameters():
            parameter.requires_grad = False
        for parameter in self.model.model.embed_tokens.parameters():
            parameter.requires_grad = True
        self.model.model.embed_tokens.freeze_base_emb()

    def freeze_emb(self):
        for parameter in self.model.model.embed_tokens.parameters():
            parameter.requires_grad = False

    @classmethod
    def init_with_encoder(cls, wav_encoder: nn.Module):
        qwen_model = AutoModelForCausalLM.from_pretrained(
            WavQwenModelWithEncoder.QWEN_REPO)
        base_emb = qwen_model.model.embed_tokens
        enc_emb = EmbeddingWithEncoder(
            base_emb, wav_encoder, WavQwenModelWithEncoder.AUDIO_TOKEN_THRESHOLD)
        qwen_model.model.embed_tokens = enc_emb
        return cls(qwen_model)
