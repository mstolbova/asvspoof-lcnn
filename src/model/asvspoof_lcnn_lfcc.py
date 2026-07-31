import math

import torch
from torch import nn

from src.model.asvspoof_lcnn import ConvMFM, LinearMFM


def linear_triangular_filterbank(n_filters, n_freqs, sample_rate):
    """Triangular filters spaced linearly in frequency.

    Returns a matrix of shape (n_freqs, n_filters) that maps a power spectrum
    onto filter-bank energies. Linear spacing is what distinguishes LFCC from
    MFCC, whose filters follow the mel scale.
    """
    f_max = sample_rate / 2
    # n_filters + 2 edges: each filter spans three consecutive points
    edges = torch.linspace(0, f_max, n_filters + 2)
    freqs = torch.linspace(0, f_max, n_freqs)

    fb = torch.zeros(n_freqs, n_filters)
    for i in range(n_filters):
        left, centre, right = edges[i], edges[i + 1], edges[i + 2]
        rising = (freqs - left) / (centre - left)
        falling = (right - freqs) / (right - centre)
        fb[:, i] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return fb


def dct_matrix(n_out, n_in):
    """Orthonormal DCT-II matrix of shape (n_in, n_out)."""
    n = torch.arange(n_in).unsqueeze(1).float()
    k = torch.arange(n_out).unsqueeze(0).float()
    basis = torch.cos(math.pi / n_in * (n + 0.5) * k)
    basis *= math.sqrt(2.0 / n_in)
    basis[:, 0] *= math.sqrt(0.5)
    return basis


def add_deltas(x):
    """Append first and second order regression coefficients along dim 1.

    x has shape (B, C, T); the result has shape (B, 3C, T). The regression
    window is two frames on each side, the usual choice for cepstral deltas.
    """

    def delta(feat):
        padded = nn.functional.pad(feat, (2, 2), mode="replicate")
        num = torch.zeros_like(feat)
        for n in (1, 2):
            left = padded[:, :, 2 - n : 2 - n + feat.size(2)]
            right = padded[:, :, 2 + n : 2 + n + feat.size(2)]
            num = num + n * (right - left)
        return num / 10.0  # 2 * (1^2 + 2^2)

    d1 = delta(x)
    d2 = delta(d1)
    return torch.cat([x, d1, d2], dim=1)


class ASVSpoofLCNNLFCC(nn.Module):
    """LCNN countermeasure with an LFCC front end.

    The network is the architecture of Lavrentyeva et al. (arXiv:1904.05576,
    Table 1), unchanged. Only the input representation differs: instead of the
    raw log power spectrum, the model consumes linear frequency cepstral
    coefficients following the recipe of Wang & Yamagishi (arXiv:2103.11326,
    Sec. 3.1) — a 20 ms window, a 10 ms shift, a 512-point FFT, a linearly
    spaced triangular filter bank of 20 channels, and delta plus delta-delta
    coefficients, with the first coefficient replaced by the log spectral
    energy of the frame. That study reports LFCC to outperform both the linear
    filter bank and the spectrogram on this database by a statistically
    significant margin.

    As in both papers, no voice activity detection and no feature
    normalisation are applied.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        win_length=320,
        hop_length=160,
        n_filters=20,
        n_ceps=20,
        input_length=77870,
        dropout=0.75,
        n_classes=2,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_ceps = n_ceps

        self.register_buffer(
            "window", torch.hamming_window(win_length), persistent=False
        )
        self.register_buffer(
            "filterbank",
            linear_triangular_filterbank(n_filters, n_fft // 2 + 1, sample_rate),
            persistent=False,
        )
        self.register_buffer(
            "dct", dct_matrix(n_ceps, n_filters), persistent=False
        )

        self.features = nn.Sequential(
            ConvMFM(1, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvMFM(32, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            ConvMFM(32, 48, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            ConvMFM(48, 48, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(48),
            ConvMFM(48, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvMFM(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            ConvMFM(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            ConvMFM(32, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            ConvMFM(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten_dim = self._infer_flatten_dim(input_length)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearMFM(self.flatten_dim, 80),
            nn.Dropout(dropout),
            nn.BatchNorm1d(80),
            nn.Linear(80, n_classes),
        )

        self._init_weights()

    def _infer_flatten_dim(self, input_length):
        was_training = self.features.training
        self.features.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            feats = self._extract_features(dummy)
            out = self.features(feats)
            flatten_dim = out.flatten(1).shape[1]
        if was_training:
            self.features.train()
        return flatten_dim

    def _extract_features(self, data_object):
        waveform = data_object
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        power = spec.abs().pow(2)  # (B, F, T)

        # filter-bank energies, then log and DCT to decorrelate
        energies = torch.matmul(power.transpose(1, 2), self.filterbank)
        log_energies = torch.log(energies + 1e-6)
        ceps = torch.matmul(log_energies, self.dct)  # (B, T, n_ceps)
        ceps = ceps.transpose(1, 2)  # (B, n_ceps, T)

        # the zeroth coefficient carries little information for this task and
        # is replaced by the log energy of the whole frame (Sec. 3.1)
        log_energy = torch.log(power.sum(dim=1) + 1e-6).unsqueeze(1)
        ceps = torch.cat([log_energy, ceps[:, 1:, :]], dim=1)

        feats = add_deltas(ceps)  # (B, 3 * n_ceps, T)
        return feats.unsqueeze(1)

    def forward(self, data_object: torch.Tensor, **batch):
        x = self._extract_features(data_object)
        x = self.features(x)
        logits = self.classifier(x)
        return {"logits": logits}

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return (
            super().__str__()
            + f"\nFlatten dim: {self.flatten_dim}"
            + f"\nAll parameters: {all_parameters}"
            + f"\nTrainable parameters: {trainable_parameters}"
        )
