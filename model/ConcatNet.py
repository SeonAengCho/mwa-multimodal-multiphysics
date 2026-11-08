import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np


def PositionalEncoding(freqs_per_dim, coords, max_cycles=16):
    """Sinusoidal encoding of each input value.

    coords: (B, D), each value normalized to [0, 1]
    returns (B, D * 2 * freqs_per_dim): sin and cos at freqs_per_dim frequencies,
    log-spaced from 1 to max_cycles cycles over [0, 1]
    """
    B, D = coords.shape
    device = coords.device

    cycles = torch.logspace(
        start=0.0,
        end=math.log2(max_cycles),
        steps=freqs_per_dim,
        base=2.0,
        device=device
    )
    freqs = 2.0 * math.pi * cycles  # (freqs_per_dim,), in radians

    pe_list = []
    for i in range(D):
        c = coords[:, i].unsqueeze(1)           # (B, 1)
        scaled = c * freqs.unsqueeze(0)         # (B, freqs_per_dim)
        pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        pe_list.append(pe)

    return torch.cat(pe_list, dim=-1)


def make_gn(num_channels, preferred_groups=(32, 16, 8, 4, 2, 1)):
    """GroupNorm with the largest group count in preferred_groups that divides num_channels."""
    for g in preferred_groups:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


# ---- Network blocks ----

def VectoMap(coord, cond, size=(80, 80, 80)):
    """Broadcasts coord (B, 6) and cond (B, 2) to constant maps: (B, 8, 80, 80, 80)."""
    B = coord.shape[0]
    D, H, W = size
    map = torch.cat([coord, cond], dim=1)  # (B, 8)
    map = map.view(B, 8, 1, 1, 1).expand(B, 8, D, H, W)  # (B, 8, 80, 80, 80)
    return map


class ConvDown(nn.Module):
    """3x3x3 conv with stride 2 (halves the resolution), then optional InstanceNorm, GELU and dropout."""
    def __init__(self, in_size, out_size, dropout, normalization, activation):
        super(ConvDown, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, 3, 2, 1, bias=False)]
        if normalization: layers.append(nn.InstanceNorm3d(out_size))
        if activation : layers.append(nn.GELU())
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ConvSame(nn.Module):
    """3x3x3 conv at the same resolution, then optional InstanceNorm, GELU and dropout."""
    def __init__(self, in_size, out_size, dropout, normalization, activation):
        super(ConvSame, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, 3, 1, 1, bias=False)]
        if normalization : layers.append(nn.InstanceNorm3d(out_size))
        if activation : layers.append(nn.GELU())
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.Layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.Layers(x)
        return x


class ConvUp(nn.Module):
    """2x2x2 transposed conv with stride 2 (doubles the resolution), then optional InstanceNorm, GELU and dropout."""
    def __init__(self, in_size, out_size, dropout, normalization, activation):
        super(ConvUp, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2, bias=False),
            ]
        if normalization : layers.append(nn.InstanceNorm3d(out_size))
        if activation : layers.append(nn.GELU())
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.Layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.Layers(x)
        return x


class ImageEncoder(nn.Module):
    """CNN image encoder (not used by ConcatNet)."""
    def __init__(self, in_size, out_size, dropout):
        super(ImageEncoder,self).__init__()
        self.Layers = nn.Sequential(
            ConvDown(2, 64, dropout, normalization=True, activation=True),
            ConvDown(64, 128, dropout, normalization=True, activation=True),
            ConvDown(128, 256, dropout, normalization=True, activation=True),
            ConvDown(256, 512, dropout, normalization=True, activation=True))

    def forward(self,x):
        x = self.Layers(x)
        return x


class ParamEncoder(nn.Module):
    """MLP (not used by ConcatNet)."""
    def __init__(self, in_size, mid_size, out_size, layer_num, dropout):
        super(ParamEncoder,self).__init__()

        layers = []
        for l in range(layer_num):
            if l == 0 : layers.append(nn.Linear(in_size, mid_size))
            else : layers.append(nn.Linear(mid_size, mid_size))
            layers.append(nn.GELU())
            if dropout:
                layers.append(nn.Dropout(dropout))

        self.Layers = nn.Sequential(*layers)

        self.Extraction = nn.Linear(mid_size, out_size)

    def forward(self, x):

        x = self.Layers(x)
        x = self.Extraction(x)

        return x


class ConcatNet(nn.Module):
    """Baseline 3D U-Net.

    The 8 parameters are broadcast to constant 80^3 maps and concatenated to the image as input channels.
    """
    def __init__(self, mode = "dmg"):
        super(ConcatNet, self).__init__()

        self.mode = mode
        dropout = 0.1

        self.down1 = ConvDown(10, 64, dropout, normalization=True, activation=True)
        self.down2 = ConvDown(64, 128, dropout, normalization=True, activation=True)
        self.down3 = ConvDown(128, 256, dropout, normalization=True, activation=True)
        self.down4 = ConvDown(256, 512, dropout, normalization=True, activation=True)

        self.midlayer = nn.Sequential(ConvSame(512, 512, dropout=dropout, normalization=True, activation=True))

        self.up1 = ConvUp(1024, 256, dropout=dropout, normalization=True, activation=True)
        self.up2 = ConvUp(512, 128, dropout=dropout, normalization=True, activation=True)
        self.up3 = ConvUp(256, 64, dropout=dropout, normalization=True, activation=True)
        self.up4 = ConvUp(128, 32, dropout=dropout, normalization=True, activation=True)

        if mode == "dmg":
            self.last_layer = nn.Sequential(
                nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1),
                nn.Sigmoid(),
                )
        else :
            self.last_layer = nn.Sequential(
                nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1))

    def forward(self, img, coord, cond):
        """
        img:   (B, 2, 81, 81, 81)  tumor and vessel masks
        coord: (B, 6)              needle position (3) and direction (3), normalized
        cond:  (B, 2)              ablation time and power, normalized
        returns (B, 81, 81, 81)    damage probability (mode="dmg") or scaled temperature (mode="temp")
        """
        img = F.interpolate(img, [80, 80, 80])  # (B, 2, 80, 80, 80)

        vecmap = VectoMap(coord, cond)  # (B, 8, 80, 80, 80)

        x = torch.cat((img, vecmap),dim=1)  # (B, 10, 80, 80, 80): 2 image + 8 parameter channels
        x1 = self.down1(x)  # (B, 64, 40, 40, 40)
        x2 = self.down2(x1)  # (B, 128, 20, 20, 20)
        x3 = self.down3(x2)  # (B, 256, 10, 10, 10)
        x4 = self.down4(x3)  # (B, 512, 5, 5, 5)

        x = self.midlayer(x4)  # (B, 512, 5, 5, 5)

        x = self.up1(torch.cat((x, x4),dim=1))  # (B, 256, 10, 10, 10)
        x = self.up2(torch.cat((x, x3),dim=1))  # (B, 128, 20, 20, 20)
        x = self.up3(torch.cat((x, x2),dim=1))  # (B, 64, 40, 40, 40)
        x = self.up4(torch.cat((x, x1),dim=1))  # (B, 32, 80, 80, 80)
        x = self.last_layer(x)  # (B, 1, 80, 80, 80)

        return F.interpolate(x, [81, 81, 81]).squeeze(1)  # (B, 81, 81, 81)
