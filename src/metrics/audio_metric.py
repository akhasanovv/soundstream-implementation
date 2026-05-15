import torch

from src.metrics.base_metric import BaseMetric


class AudioMetric(BaseMetric):
    """
    metrics for soundstream
    """
    def __init__(self, metric, sample_rate=16000, pred_key="reconstructed_audio", target_key="audio", *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.metric = metric
        self.sample_rate = sample_rate
        self.pred_key = pred_key
        self.target_key = target_key

    def __call__(self, **batch):
        """
        Metric calculation logic.
        """
        pred = batch[self.pred_key] 
        target = batch[self.target_key]

        if pred.dim() == 3 and pred.shape[1] == 1: # -> (batch_size, len)
            pred = pred.squeeze(1)
        
        if target.dim() == 3 and target.shape[1] == 1:
            target = target.squeeze(1)

        min_len = min(pred.shape[-1], target.shape[-1]) # in case one is shorter
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        value = self.metric(pred, target)
        if torch.is_tensor(value):
            return value.detach().item()
        return float(value)
