import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


class ASVSpoofDataset(Dataset):
    """ASVspoof2019 Logical Access countermeasure dataset.

    Trials are brought to a fixed length of `max_length` samples, which for the
    default value corresponds to 600 STFT frames -- the input size used by the
    FFT-LCNN system of Lavrentyeva et al. (arXiv:1904.05576). Longer trials are
    trimmed (at a random offset during training, as in Wang & Yamagishi,
    arXiv:2103.11326, Sec. 3.1), shorter trials are padded by repeating the
    waveform, which avoids the constant-silence artefact that zero padding
    introduces in the log power spectrum.
    """

    def __init__(
        self,
        root_dir,
        part="train",
        sample_rate=16000,
        max_length=77870,
        random_crop=True,
    ):
        self.root_dir = Path(root_dir)
        self.part = part
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.random_crop = random_crop

        self.protocol_dir = self.root_dir / "ASVspoof2019_LA_cm_protocols"

        self.protocol_paths = {
            "train": self.protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt",
            "dev": self.protocol_dir / "ASVspoof2019.LA.cm.dev.trl.txt",
            "eval": self.protocol_dir / "ASVspoof2019.LA.cm.eval.trl.txt",
        }

        self.audio_dirs = {
            "train": self.root_dir / "ASVspoof2019_LA_train" / "flac",
            "dev": self.root_dir / "ASVspoof2019_LA_dev" / "flac",
            "eval": self.root_dir / "ASVspoof2019_LA_eval" / "flac",
        }

        if self.part not in self.protocol_paths:
            raise ValueError(f"Unknown part: {self.part}")

        self.protocol_path = self.protocol_paths[self.part]
        self.audio_dir = self.audio_dirs[self.part]

        if not self.protocol_path.exists():
            raise FileNotFoundError(self.protocol_path)

        if not self.audio_dir.exists():
            raise FileNotFoundError(self.audio_dir)

        self.index = self._load_protocol()

    def _load_protocol(self):
        index = []

        with open(self.protocol_path, "r") as f:
            for line in f:
                fields = line.strip().split()

                if not fields:
                    continue

                speaker_id = fields[0]
                file_id = fields[1]
                attack_type = fields[3]
                label_text = fields[4]

                label = 1.0 if label_text == "bonafide" else 0.0
                audio_path = self.audio_dir / f"{file_id}.flac"

                index.append(
                    {
                        "speaker_id": speaker_id,
                        "file_id": file_id,
                        "attack_type": attack_type,
                        "label": label,
                        "label_text": label_text,
                        "audio_path": audio_path,
                    }
                )

        return index

    def __len__(self):
        return len(self.index)

    def _fix_audio_length(self, waveform):
        num_samples = waveform.shape[-1]

        if num_samples > self.max_length:
            max_start = num_samples - self.max_length
            # random offset during training, deterministic prefix otherwise
            start = random.randint(0, max_start) if self.random_crop else 0
            waveform = waveform[:, start : start + self.max_length]

        elif num_samples < self.max_length:
            repeats = self.max_length // num_samples + 1
            waveform = waveform.repeat(1, repeats)[:, : self.max_length]

        return waveform

    def __getitem__(self, idx):
        item = self.index[idx]

        if not item["audio_path"].exists():
            raise FileNotFoundError(item["audio_path"])

        waveform, sr = torchaudio.load(item["audio_path"])

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=self.sample_rate,
            )

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = self._fix_audio_length(waveform)
        label = torch.tensor(item["label"], dtype=torch.float32)

        return {
            "data_object": waveform,
            "labels": label,
            "file_id": item["file_id"],
            "audio_path": str(item["audio_path"]),
            "attack_type": item["attack_type"],
            "label_text": item["label_text"],
        }
