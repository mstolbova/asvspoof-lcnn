import torch
from torch import nn


class MFM(nn.Module):
    """Max-Feature-Map 2/1 (Wu et al., 2018): splits the input in half along
    `dim` and takes the element-wise maximum of the two halves."""

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        assert x.size(self.dim) % 2 == 0, "MFM input size must be divisible by 2"
        chunks = torch.chunk(x, 2, dim=self.dim)
        return torch.maximum(chunks[0], chunks[1])


class ConvMFM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            MFM(dim=1),
        )

    def forward(self, x):
        return self.block(x)


class LinearMFM(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2)
        self.mfm = MFM(dim=1)

    def forward(self, x):
        return self.mfm(self.linear(x))


class ASVSpoofLCNN(nn.Module):
    """LCNN countermeasure following Lavrentyeva et al., "STC Antispoofing
    Systems for the ASVspoof2019 Challenge" (arXiv:1904.05576), Table 1.

    Front-end is the FFT-LCNN configuration from the same paper: raw log power
    magnitude spectrum, 1724-point window, 0.0081 s step, Blackman window.
    Following the paper and Wang & Yamagishi (arXiv:2103.11326, Sec. 3.2),
    neither VAD nor feature normalisation is applied.
    """

    def __init__(
        self,
        n_fft=1724,
        hop_length=130,
        win_length=1724,
        input_length=77870,
        dropout=0.5,
        n_classes=2,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer(
            "window", torch.blackman_window(win_length), persistent=False
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
        """Run a dummy waveform through the front-end and the conv stack to
        determine the flattened feature size. Done in eval mode so that the
        BatchNorm running statistics are not polluted."""
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

        power = spec.abs().pow(2)
        log_power = torch.log(power + 1e-6)
        return log_power.unsqueeze(1)

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
