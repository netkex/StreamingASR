import numpy as np
from transformers import EvalPrediction
from .utils import build_ref_seq

from StreamASRLib.eval import calculate_wer


class WerMetric:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.wer_hist = []

    def __call__(self, eval_pred: EvalPrediction, compute_result: bool) -> float:
        logits = eval_pred.predictions
        input_ids = getattr(eval_pred, 'inputs')['input_ids']
        label_text_start = eval_pred.label_ids[0]
        label_text_end = eval_pred.label_ids[1]

        for id, (start, end) in enumerate(zip(label_text_start, label_text_end)):
            ref_seq = build_ref_seq(
                self.tokenizer, logits[id, (start - 1):(end - 2), :])
            orig_seq = self.tokenizer.decode(
                input_ids[id, start:(end - 1)].detach().cpu())
            if len(ref_seq) > 0 and len(orig_seq) > 0:
                self.wer_hist.append(calculate_wer([ref_seq], [orig_seq]))
        if compute_result:
            wer_ = np.mean(self.wer_hist)
            self.wer_hist = []
            return {'WER*': wer_}
