"""Segmentation models. All take uint8 base codes (B, L) and return per-base logits (B, L)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(codes):
    """(B, L) codes A=0 C=1 G=2 T=3 N=4 -> (B, 4, L) floats; N becomes all zeros."""
    x = F.one_hot(codes.long(), 5)[..., :4]
    return x.permute(0, 2, 1).float()


def conv_block(c_in, c_out, kernel, n_convs, dropout, norm):
    layers = []
    for i in range(n_convs):
        layers += [nn.Conv1d(c_in if i == 0 else c_out, c_out, kernel, padding="same")]
        if norm:
            layers += [nn.BatchNorm1d(c_out)]
        layers += [nn.ReLU(inplace=True)]
    if dropout:
        layers += [nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class UNet1D(nn.Module):
    """Configurable 1D U-Net with the layout of the 2023 Keras model.

    Encoder level i: convolutions, then max-pooling by `pool`. Decoder level i (deepest first):
    convolutions, upsampling by `pool`, concatenation with the matching encoder output.
    A 1x1 convolution gives the per-base logit. channels=(16, 32, 64), kernel=41, pool=5,
    convs_per_level=1, dropout=0.2 reproduces the draft's final model (kernel 40 there; odd
    kernels keep 'same' padding symmetric).
    """

    def __init__(self, channels=(16, 32, 64), kernel=41, pool=5, convs_per_level=1,
                 dropout=0.2, norm=False):
        super().__init__()
        self.pool = pool
        self.enc = nn.ModuleList()
        c_prev = 4
        for c in channels:
            self.enc.append(conv_block(c_prev, c, kernel, convs_per_level, dropout, norm))
            c_prev = c
        self.dec = nn.ModuleList()
        for c in reversed(channels):
            self.dec.append(conv_block(c_prev, c, kernel, convs_per_level, dropout, norm))
            c_prev = 2 * c  # after concatenation with the skip connection of the same width
        self.head = nn.Conv1d(c_prev, 1, 1)

    def forward(self, codes):
        x = one_hot(codes)
        L = x.shape[-1]
        m = self.pool ** len(self.enc)
        pad = (-L) % m
        if pad:
            x = F.pad(x, (0, pad))
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = F.max_pool1d(x, self.pool)
        for block, skip in zip(self.dec, reversed(skips)):
            x = F.interpolate(block(x), scale_factor=self.pool, mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.head(x)[:, 0, :L]


class PlainCNN(nn.Module):
    """Stack of same-length convolutions without down-sampling (the Feb 2025 notebook model)."""

    def __init__(self, channels=(16, 32, 64), kernels=(10, 5, 5), dropout=0.0, norm=False):
        super().__init__()
        layers, c_prev = [], 4
        for c, k in zip(channels, kernels):
            layers.append(conv_block(c_prev, c, k, 1, dropout, norm))
            c_prev = c
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(c_prev, 1, 1)

    def forward(self, codes):
        return self.head(self.body(one_hot(codes)))[:, 0]


class DilatedResNet(nn.Module):
    """SpliceAI-style residual blocks with growing dilation: wide receptive field, full resolution."""

    def __init__(self, channels=32, kernel=11, dilations=(1, 1, 2, 2, 4, 4, 8, 8), norm=True):
        super().__init__()
        self.stem = nn.Conv1d(4, channels, 1)
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.Sequential(
                nn.BatchNorm1d(channels) if norm else nn.Identity(), nn.ReLU(),
                nn.Conv1d(channels, channels, kernel, dilation=d, padding="same"),
                nn.BatchNorm1d(channels) if norm else nn.Identity(), nn.ReLU(),
                nn.Conv1d(channels, channels, kernel, dilation=d, padding="same")))
        self.head = nn.Conv1d(channels, 1, 1)

    def forward(self, codes):
        x = self.stem(one_hot(codes))
        for b in self.blocks:
            x = x + b(x)
        return self.head(x)[:, 0]


class CNNBiLSTM(nn.Module):
    """Helixer-style: convolutions for local motifs, a bidirectional LSTM for long-range context."""

    def __init__(self, channels=64, kernel=9, n_convs=2, hidden=64, lstm_layers=2):
        super().__init__()
        self.conv = conv_block(4, channels, kernel, n_convs, 0, True)
        self.lstm = nn.LSTM(channels, hidden, lstm_layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, codes):
        x = self.conv(one_hot(codes)).transpose(1, 2)
        x, _ = self.lstm(x)
        return self.head(x)[..., 0]


MODELS = {"unet": UNet1D, "cnn": PlainCNN, "dilated": DilatedResNet, "bilstm": CNNBiLSTM}


def build(name, **kwargs):
    return MODELS[name](**kwargs)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
