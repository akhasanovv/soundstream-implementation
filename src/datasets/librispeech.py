import numpy as np
import torch
import torchaudio
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import random

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class LibriSpeechDataset(BaseDataset):
    """
    LibriSpeech dataset
    """

    def __init__(
        self,
        split,
        data_dir,
        sample_rate=16000,
        segment_seconds=None,
        random_crop=False,
        storage_path=None,
        limit=None,
        shuffle_rows=False,
        instance_transforms=None,
        allowed_suffixes=(".flac", ".wav"),
    ):
        self.split = split
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.random_crop = random_crop
        self.limit = limit
        self.shuffle_rows = shuffle_rows
        self.instance_transforms = instance_transforms
        self.allowed_suffixes = tuple(s.lower() for s in allowed_suffixes)

        if storage_path is None:
            storage_path = (
                ROOT_PATH / "data" / "librispeech_index" / f"{self.split}_index.json"
            )

        self.storage_path = Path(storage_path)
        self.index = self._create_index()

        if self.shuffle_rows:
            random.seed(42)
            random.shuffle(self.index)

        if self.limit is not None:
            self.index = self.index[: self.limit]

        self.segment_length = None
        if self.segment_seconds is not None:
            self.segment_length = int(round(self.segment_seconds * self.sample_rate))
            if self.segment_length <= 0:
                raise ValueError("segment_length <= 0")

    def _create_index(self):
        """
        Create index for the dataset
        """
        if self.storage_path.exists():
            return read_json(self.storage_path)

        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"train/test directory not found: {split_dir}")

        files = []
        for path in split_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.allowed_suffixes:
                files.append(path)

        if len(files) == 0:
            raise RuntimeError(f"only {self.allowed_suffixes} files are allowed")

        paths = [{"path": str(path)} for path in sorted(files)]

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(paths, self.storage_path)

        return paths

    def __len__(self):
        return len(self.index)

    def crop_or_pad(self, waveform, segment_length, random_crop):
        """
        align len of waveform to segment_length. make random crop if needed
        """
        total_len = waveform.shape[-1]
        if total_len < segment_length:
            n_repeat = (
                segment_length + total_len - 1
            ) // total_len  # ceil(seg_len / tot_len)
            waveform = waveform.repeat(1, n_repeat)
            total_len = segment_length

        if random_crop:
            start = random.randint(0, total_len - segment_length)
            return waveform[:, start : start + segment_length]

        return waveform[:, :segment_length]

    def __getitem__(self, ind):
        item = self.index[ind]
        audio_path = item["path"]

        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )

        if self.segment_length is not None:
            waveform = self.crop_or_pad(
                waveform,
                segment_length=self.segment_length,
                random_crop=self.random_crop,
            )

        sample = {
            "audio": waveform,
            "audio_length": waveform.shape[-1],
            "audio_path": audio_path,
        }

        if self.instance_transforms is not None:
            for transform_name, transform in self.instance_transforms.items():
                sample[transform_name] = transform(sample[transform_name])

        return sample
