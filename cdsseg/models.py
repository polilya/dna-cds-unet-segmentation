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
                 dropout=0.2, norm=False, full_res_conv=False):
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
        # the 2023 decoder has no convolution at 1-bp resolution; full_res_conv adds one before the head
        self.final = conv_block(c_prev, channels[0], kernel, 1, 0.0, norm) if full_res_conv else nn.Identity()
        if full_res_conv:
            c_prev = channels[0]
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
        return self.head(self.final(x))[:, 0, :L]


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


class ResBlock(nn.Module):
    """Two convolutions with BatchNorm and an identity (or 1x1) shortcut."""

    def __init__(self, c_in, c_out, kernel, dropout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding="same"), nn.BatchNorm1d(c_out), nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Conv1d(c_out, c_out, kernel, padding="same"), nn.BatchNorm1d(c_out))
        self.skip = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1)

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class UNet2(nn.Module):
    """1D U-Net with the weak points of the 2023 layout fixed.

    - Standard decoder: upsample first (linear), concatenate the skip, then convolve, so the last
      decoder block runs at 1-bp resolution (the 2023 decoder convolves before upsampling and has
      no convolution at full resolution).
    - residual=True: two-convolution residual blocks with BatchNorm (stable at this depth);
      residual=False: one convolution + BatchNorm per block, as in the best 2023-layout model.
    - strands=2: one output per strand (labels bit 0 = +strand, bit 1 = -strand); the CDS
      probability is P(+ or -). strands=1 predicts the merged label.
    The encoder pools `len(channels)` times; a bottleneck block of width channels[-1] runs at the
    lowest resolution, so the receptive field matches the 2023 model for the same kernel and pool.
    """

    def __init__(self, channels=(32, 64, 128), kernel=41, pool=5, dropout=0.2, residual=True, strands=2, gate=False,
                 gate_detach=False):
        super().__init__()
        self.pool, self.strands = pool, strands
        self.gate_detach = gate_detach  # train the gate on frozen features: its loss cannot change the segmenter
        # gate=True adds a window-level head on the bottleneck ("does this window contain any CDS?"). In eval mode
        # with gate_threshold set, the decoder runs only on windows that pass; the others get "no CDS".
        self.gate_head = nn.Linear(2 * channels[-1], 1) if gate else None
        self.gate_threshold, self.last_gate, self.gate_counts = None, None, [0, 0]
        block = (lambda a, b: ResBlock(a, b, kernel, dropout)) if residual else \
            (lambda a, b: conv_block(a, b, kernel, 1, dropout, True))
        self.enc = nn.ModuleList()
        c_prev = 4
        for c in channels:
            self.enc.append(block(c_prev, c))
            c_prev = c
        self.bottleneck = block(c_prev, channels[-1])
        self.dec = nn.ModuleList()
        c_prev = channels[-1]
        for c in reversed(channels):
            self.dec.append(block(c_prev + c, c))
            c_prev = c
        self.head = nn.Conv1d(c_prev, strands, 1)

    def forward(self, codes):
        x = one_hot(codes)
        L = x.shape[-1]
        pad = (-L) % (self.pool ** len(self.enc))
        if pad:
            x = F.pad(x, (0, pad))
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = F.max_pool1d(x, self.pool)
        x = self.bottleneck(x)
        keep = None
        if self.gate_head is not None:
            g = x.detach() if self.gate_detach else x
            self.last_gate = self.gate_head(torch.cat([g.amax(-1), g.mean(-1)], dim=1))[:, 0]
            if not self.training and self.gate_threshold is not None:
                keep = torch.sigmoid(self.last_gate.float()) >= self.gate_threshold
                self.gate_counts[0] += keep.numel()
                self.gate_counts[1] += int(keep.sum())
                if not keep.any():
                    out = x.new_full((keep.numel(), self.strands, L), -20.0)
                    return out[:, 0] if self.strands == 1 else out
                x, skips = x[keep], [s[keep] for s in skips]
        for block, skip in zip(self.dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        out = self.head(x)[..., :L]
        if keep is not None:  # windows rejected by the gate get a confident "no CDS"
            full = out.new_full((keep.numel(), *out.shape[1:]), -20.0)
            full[keep] = out
            out = full
        return out[:, 0] if self.strands == 1 else out


MODELS = {"unet": UNet1D, "unet2": UNet2, "cnn": PlainCNN, "dilated": DilatedResNet, "bilstm": CNNBiLSTM}


def build(name, **kwargs):
    return MODELS[name](**kwargs)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
