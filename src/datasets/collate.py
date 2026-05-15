import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = {}

    if len(dataset_items) == 0:
        return result_batch

    first_item = dataset_items[0]

    if "audio" in first_item:
        lengths = torch.tensor([elem["audio"].shape[-1] for elem in dataset_items])
        max_len = int(lengths.max().item())

        padded_audio = []
        for elem in dataset_items:
            audio = elem["audio"]  # (1, T)
            pad_amount = max_len - audio.shape[-1]
            padded_audio.append(F.pad(audio, (0, pad_amount)))

        batch = {
            "audio": torch.stack(padded_audio, dim=0),  # (B, 1, max_len)
            "audio_length": lengths,
            "audio_path": [elem["audio_path"] for elem in dataset_items],
        }
        return batch

    result_batch["data_object"] = torch.vstack(
        [elem["data_object"] for elem in dataset_items]
    )
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])

    return result_batch
