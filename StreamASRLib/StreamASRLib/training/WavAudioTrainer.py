import torch.nn as nn
from transformers import Trainer


class WavAudioTrainer(Trainer):
    '''
    Custom trainer class with cross-entropy loss only for text tokens
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        label_text_start = inputs.label_text_start
        label_text_end = inputs.label_text_end
        output = model(input_ids=inputs.input_ids,
                       attention_mask=inputs.attention_mask)

        logits = output.logits.float()
        batch_size = logits.shape[0]
        loss = 0

        for id, (start, end) in enumerate(zip(label_text_start, label_text_end)):
            seq_logits = logits[id, (start - 1):(end - 1), :]
            seq_target = inputs.input_ids[id, start:end]
            loss += self.loss(seq_logits, seq_target)
        loss /= batch_size

        return (loss, output) if (return_outputs) else loss
