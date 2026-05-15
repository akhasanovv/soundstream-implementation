import torch

from src.metrics.base_metric import BaseMetric


class AudioMetric(BaseMetric):
    """
    metrics for soundstream
    """

    def __init__(
        self,
        metric,
        sample_rate=16000,
        pred_key="reconstructed_audio",
        target_key="audio",
        output_index=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.metric = metric
        self.sample_rate = sample_rate
        self.pred_key = pred_key
        self.target_key = target_key
        self.output_index = output_index

    def __call__(self, **batch):
        """
        Metric calculation logic.
        """
        pred = batch[self.pred_key]

        target = None
        if self.target_key is not None:
            target = batch[self.target_key]

        audio_length = batch.get("audio_length")

        if pred.dim() == 3 and pred.shape[1] == 1:  # -> (batch_size, len)
            pred = pred.squeeze(1)

        if target is not None and (target.dim() == 3 and target.shape[1] == 1):
            target = target.squeeze(1)

        if audio_length is not None:
            values = []
            for i, length in enumerate(audio_length):
                if target is None:
                    value = self.metric(pred[i : i + 1, :length])
                else:
                    value = self.metric(
                        pred[i : i + 1, :length], target[i : i + 1, :length]
                    )

                if self.output_index is not None:  # nisqa -> R^5
                    value = value[..., self.output_index]
                if torch.is_tensor(value):
                    value = value.detach().mean()

                values.append(value)

            return torch.stack(values).mean().item()

        if target is not None:
            min_len = min(pred.shape[-1], target.shape[-1])
            value = self.metric(pred[..., :min_len], target[..., :min_len])
        else:
            value = self.metric(pred)

        if self.output_index is not None:
            value = value[..., self.output_index]
        if torch.is_tensor(value):
            return value.detach().mean().item()

        return value
