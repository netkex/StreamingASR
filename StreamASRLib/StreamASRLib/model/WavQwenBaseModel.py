from transformers import AutoTokenizer


class WavQwenBaseModel:
    QWEN_REPO = 'Qwen/Qwen2.5-0.5B'
    EOS_TOKEN = 151643                               # taken from qwen tokenizer
    AUDIO_TOKENS = 4096                              # taken from wav tokenizer doc
    AUDIO_TOKEN_THRESHOLD = 151936                   # taken from qwen tokenizer
    PADDING_TOKEN = EOS_TOKEN
    BOA_TOKEN = AUDIO_TOKEN_THRESHOLD + AUDIO_TOKENS
    EOA_TOKEN = BOA_TOKEN + 1
    EXT_TOKENS = AUDIO_TOKENS + 2

    def __init__(self):
        pass

    def model(self):
        raise NotImplemented('Not implemented')

    def tokenizer(self) -> AutoTokenizer:
        raise NotImplemented('Not implemented')
