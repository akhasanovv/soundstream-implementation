import torch
import torch.nn.functional as F
from torch import nn


class WaveDiscriminator(nn.Module):
    """
    wave discriminator from [paper](https://arxiv.org/abs/2107.03312)
    """

    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
                nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
                nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=4),
                nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=4),
                nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=4),
                nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2),
                nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1),
            ]
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        hs = []
        h = x

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = self.act(h)
            hs.append(h)

        return hs[-1], hs


class STFTResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        ker = (stride[0] + 2, stride[1] + 2)  # (3, 4) or (4, 4)
        pad = (1, 1)

        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=ker, stride=stride, padding=pad
        )

        self.conv3 = nn.Conv2d(  # skip connection
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = self.conv2(self.act(self.conv1(x)))
        z = self.conv3(x)
        z = z[..., : y.shape[-2], : y.shape[-1]]  # just in case..
        return self.act(y + z)


class STFTDiscriminator(nn.Module):
    """
    stft discriminator from [paper](https://arxiv.org/abs/2107.03312)
    """

    def __init__(self, in_channels=32, n_fft=1024, hop_length=256):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft

        self.input_conv = nn.Conv2d(  # complex -> in_channels
            2, in_channels, kernel_size=(7, 7), stride=1, padding=(3, 3)
        )
        self.act = nn.LeakyReLU(0.2)

        self.blocks = nn.ModuleList(
            [
                STFTResidualUnit(
                    in_channels=in_channels, out_channels=2 * in_channels, stride=(1, 2)
                ),
                STFTResidualUnit(
                    in_channels=2 * in_channels,
                    out_channels=4 * in_channels,
                    stride=(2, 2),
                ),
                STFTResidualUnit(
                    in_channels=4 * in_channels,
                    out_channels=4 * in_channels,
                    stride=(1, 2),
                ),
                STFTResidualUnit(
                    in_channels=4 * in_channels,
                    out_channels=8 * in_channels,
                    stride=(2, 2),
                ),
                STFTResidualUnit(
                    in_channels=8 * in_channels,
                    out_channels=8 * in_channels,
                    stride=(1, 2),
                ),
                STFTResidualUnit(
                    in_channels=8 * in_channels,
                    out_channels=16 * in_channels,
                    stride=(2, 2),
                ),
            ]
        )
        self.final_conv = nn.Conv2d(
            16 * in_channels, 1, kernel_size=(1, 8), stride=1, padding=0
        )

        self.register_buffer("window", torch.hann_window(self.win_length))

    def get_stft(self, audio):
        x = audio.squeeze(1)

        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=x.device, dtype=x.dtype),
            center=True,
            return_complex=True,
        )

        spec = spec[:, :-1, :]
        out = torch.stack([spec.real, spec.imag], dim=1)
        out = out.permute(0, 1, 3, 2).contiguous()

        return out

    def forward(self, audio):
        h = self.get_stft(audio)
        h = self.act(self.input_conv(h))
        hs = [h]

        for block in self.blocks:
            h = block(h)
            hs.append(h)

        last = self.final_conv(h).squeeze(-1)
        hs.append(last)
        return last, hs


class SoundStreamDiscriminator(nn.Module):
    """
    final soundstream discriminator stack (1 stft + 3 wave with 1x, 2x and 4x downsampling)
    """

    def __init__(self):
        super().__init__()

        self.stft_disc = STFTDiscriminator(in_channels=32)
        self.wave_discs = nn.ModuleList(
            [WaveDiscriminator(), WaveDiscriminator(), WaveDiscriminator()]
        )

        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

    def forward(self, audio):
        wave = []
        x = audio
        stft = self.stft_disc(audio)

        for i, wave_disc in enumerate(self.wave_discs):
            wave.append(wave_disc(x))
            if i + 1 < len(self.wave_discs):
                x = self.downsample(x)

        return {"stft": stft, "wave": wave}
