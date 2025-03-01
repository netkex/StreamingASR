import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM

import os


class ExtendedEmbedding(nn.Module):
    '''
    Custom embedding class for wav-tokens
    '''

    def __init__(self, base_emb: nn.Embedding, ext_tokens_size: int, ext_token_threshold: int):
        super().__init__()
        self.base_emb = base_emb
        self.hidden_size = base_emb.embedding_dim
        self.ext_emb = nn.Embedding(ext_tokens_size, self.hidden_size)
        self.ext_token_threshold = ext_token_threshold

    def freeze_base_emb(self):
        for param in self.base_emb.parameters():
            param.requires_grad = False

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        ext_token_mask = (input_tokens >= self.ext_token_threshold)
        inp_emb = torch.zeros(list(
            input_tokens.shape) + [self.hidden_size], requires_grad=True).to(self.base_emb.weight.device)
        inp_emb[~ext_token_mask] = self.base_emb(input_tokens[~ext_token_mask])
        inp_emb[ext_token_mask] = self.ext_emb(
            input_tokens[ext_token_mask] - self.ext_token_threshold)
        return inp_emb


class WavQwenModel:
    # TODO: make WavQwenModel to inherite from AutoModelForCausalLM
    # TODO: fix placing on devices (explecitly check how embedding layer is placed in case of multiple GPU)

    QWEN_REPO = 'Qwen/Qwen2.5-0.5B'
    EOS_TOKEN = 151643                               # taken from qwen tokenizer
    AUDIO_TOKENS = 4096                              # taken from wav tokenizer doc
    AUDIO_TOKEN_THRESHOLD = 151936                   # taken from qwen tokenizer
    PADDING_TOKEN = EOS_TOKEN
    BOA_TOKEN = AUDIO_TOKEN_THRESHOLD + AUDIO_TOKENS
    EOA_TOKEN = BOA_TOKEN + 1
    EXT_TOKENS = AUDIO_TOKENS + 2

    def __init__(self, model, fast_tokenizer=True):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            WavQwenModel.QWEN_REPO, use_fast=fast_tokenizer)

    @classmethod
    def init(cls):
        qwen_model = AutoModelForCausalLM.from_pretrained(
            WavQwenModel.QWEN_REPO)
        base_emb = qwen_model.model.embed_tokens
        ext_emb = ExtendedEmbedding(
            base_emb, WavQwenModel.EXT_TOKENS, WavQwenModel.AUDIO_TOKEN_THRESHOLD)
        qwen_model.model.embed_tokens = ext_emb
        return cls(qwen_model)

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, sft_path: str = 'model.safetensors'):
        qwen_model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        sft_path = os.path.join(checkpoint_path, sft_path)
        with safe_open(sft_path, framework="pt") as sft:
            base_emb = qwen_model.model.embed_tokens
            ext_emb = ExtendedEmbedding(
                base_emb, WavQwenModel.EXT_TOKENS, WavQwenModel.AUDIO_TOKEN_THRESHOLD)
            ext_emb_state_dict = ext_emb.state_dict()
            ext_emb_state_dict['base_emb.weight'] = sft.get_tensor(
                'model.embed_tokens.base_emb.weight')
            ext_emb_state_dict['ext_emb.weight'] = sft.get_tensor(
                'model.embed_tokens.ext_emb.weight')
            ext_emb.load_state_dict(ext_emb_state_dict)
            qwen_model.model.embed_tokens = ext_emb
        return cls(qwen_model)
