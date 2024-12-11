import wandb
from dataclasses import dataclass
from typing import List


@dataclass
class MetricLog:
    train_loss: float | None = None

    eval_loss: float | None = None
    eval_wer: float | None = None

    sample_wer: float | None = None
    sample_refs: List[str] | None = None
    sample_targets: List[str] | None = None

    def set_train_loss(self, train_loss: float | None):
        self.train_loss = train_loss

    def set_eval_metrics(self, eval_loss: float | None, eval_wer: float | None):
        self.eval_loss = eval_loss
        self.eval_wer = eval_wer

    def set_sample(self, sample_wer, sample_refs: List[str], sample_targets: List[str]):
        self.sample_wer = sample_wer
        self.sample_refs = sample_refs
        self.sample_targets = sample_targets

    def has_eval(self) -> bool:
        return self.eval_loss is not None

    def has_sample(self) -> bool:
        return self.sample_wer is not None

    def create_sample_table(self):
        columns = ['Reference', 'Target']
        data = [[ref, target]
                for ref, target in zip(self.sample_refs, self.sample_targets)]
        return wandb.Table(data=data, columns=columns)


def save_metric(step, metric_log: MetricLog):
    wandb_log = {}
    wandb_log['Train loss'] = metric_log.train_loss

    if metric_log.has_eval():
        wandb_log['Eval loss'] = metric_log.eval_loss
        wandb_log['Eval WER*'] = metric_log.eval_wer

    if metric_log.has_sample():
        wandb_log['Train WER*'] = metric_log.sample_wer
        wandb_log['Reference examples'] = metric_log.create_sample_table()

    wandb.log(wandb_log, step=step)
