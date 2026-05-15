import torch.nn.functional as F
from torch import nn

from src.model.rvq import ResidualVectorQuantizer


class CausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super().__init__()

        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
        )

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class CausalConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.trim_right = kernel_size - stride
        self.conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

    def forward(self, x):
        x = self.conv(x)
        if self.trim_right > 0:
            x = x[..., : -self.trim_right]
        return x


class ResidualUnit(nn.Module):
    def __init__(self, channels, kernel_size=7, dilation=1):
        super().__init__()

        self.layer = nn.Sequential(
            CausalConv(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            nn.ELU(),
            nn.Conv1d(  # same as CausalConv, because pad=0
                in_channels=channels, out_channels=channels, kernel_size=1
            ),
        )

    def forward(self, x):
        return x + self.layer(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(channels=in_channels, kernel_size=7, dilation=1),
            ResidualUnit(channels=in_channels, kernel_size=7, dilation=3),
            ResidualUnit(channels=in_channels, kernel_size=7, dilation=9),
            nn.ELU(),
            CausalConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ELU(),
            CausalConvTranspose(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
            ),
            ResidualUnit(channels=out_channels, kernel_size=7, dilation=1),
            ResidualUnit(channels=out_channels, kernel_size=7, dilation=3),
            ResidualUnit(channels=out_channels, kernel_size=7, dilation=9),
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConv(in_channels=1, out_channels=in_channels, kernel_size=7),
            EncoderBlock(
                in_channels=in_channels, out_channels=2 * in_channels, stride=2
            ),
            EncoderBlock(
                in_channels=2 * in_channels, out_channels=4 * in_channels, stride=4
            ),
            EncoderBlock(
                in_channels=4 * in_channels, out_channels=8 * in_channels, stride=5
            ),
            EncoderBlock(
                in_channels=8 * in_channels, out_channels=16 * in_channels, stride=5
            ),
            nn.ELU(),
            CausalConv(
                in_channels=16 * in_channels, out_channels=out_channels, kernel_size=3
            ),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConv(
                in_channels=out_channels, out_channels=16 * in_channels, kernel_size=7
            ),
            DecoderBlock(
                in_channels=16 * in_channels, out_channels=8 * in_channels, stride=5
            ),
            DecoderBlock(
                in_channels=8 * in_channels, out_channels=4 * in_channels, stride=5
            ),
            DecoderBlock(
                in_channels=4 * in_channels, out_channels=2 * in_channels, stride=4
            ),
            DecoderBlock(
                in_channels=2 * in_channels, out_channels=in_channels, stride=2
            ),
            nn.ELU(),
            CausalConv(in_channels=in_channels, out_channels=1, kernel_size=7),
        )

    def forward(self, x):
        return self.layers(x)


class SoundStreamModel(nn.Module):
    """
    SoundStream model from [paper](https://arxiv.org/abs/2107.03312)
    """

    def __init__(
        self, in_channels=32, out_channels=128, num_quantizers=8, num_embeddings=1024
    ):
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels, out_channels=out_channels)
        self.rvq = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            num_embeddings=num_embeddings,
            embedding_dim=out_channels,
        )

        self.decoder = Decoder(in_channels=in_channels, out_channels=out_channels)

    def forward(self, audio, **batch):
        lat_x = self.encoder(audio)
        quantized_lat, rvq_loss, rvq_indices = self.rvq(lat_x)
        x_hat = self.decoder(quantized_lat)

        assert x_hat.shape[-1] == audio.shape[-1]

        return {
            "reconstructed_audio": x_hat,
            "latent": lat_x,
            "quantized_latent": quantized_lat,
            "commit_loss": rvq_loss,
            "codebook_indices": rvq_indices,
        }
