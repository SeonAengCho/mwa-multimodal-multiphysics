import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

from .ViTEncoder import *


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


# ---- Network blocks ----

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
    """Trilinear x2 upsampling and a 3x3x3 conv (doubles the resolution), then optional InstanceNorm, GELU and dropout."""
    def __init__(self, in_size, out_size, dropout, normalization, activation):
        super(ConvUp, self).__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_size, out_size, kernel_size=3, padding=1, bias=False)
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
    """Alternative CNN image encoder (not used; AffineUNet3 uses ViTEncoder): (B, in_size, 80, 80, 80) -> (B, out_size, 5, 5, 5)."""
    def __init__(self, in_size, out_size, dropout):
        super(ImageEncoder,self).__init__()
        self.Layers = nn.Sequential(
            ConvDown(in_size, 64, dropout, normalization=True, activation=True),
            ConvDown(64, 128, dropout, normalization=True, activation=True),
            ConvDown(128, 256, dropout, normalization=True, activation=True),
            ConvDown(256, out_size, dropout, normalization=True, activation=True))

    def forward(self,x):
        x = self.Layers(x)
        return x


class VectorEncoder(nn.Module):
    """MLP with layer_num hidden layers (not used by AffineUNet3)."""
    def __init__(self, in_size, mid_size, out_size, layer_num, dropout):
        super(VectorEncoder, self).__init__()
        layers = [nn.Linear(in_size, mid_size), nn.GELU()]
        for _ in range(layer_num):
            layers.append(nn.Linear(mid_size, mid_size))
            layers.append(nn.GELU())
            if dropout:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(mid_size, out_size))
        self.Layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.Layers(x)


class ParamEncoder(nn.Module):
    """MLP with a residual branch: encoded parameters -> FiLM parameters of the 4 decoder stages.

    returns (B, 1024), (B, 512), (B, 256), (B, 128): [gamma, beta] for 512, 256, 128 and 64 channels
    """
    def __init__(self, in_size, mid_size, layer_num, dropout):
        super(ParamEncoder,self).__init__()

        layers = []
        for l in range(layer_num):
            if l == 0 : layers.append(nn.Linear(in_size, mid_size))
            else : layers.append(nn.Linear(mid_size, mid_size))
            layers.append(nn.GELU())
            if dropout:
                layers.append(nn.Dropout(dropout))

        self.Layers = nn.Sequential(*layers)

        self.Residual = nn.Sequential(
            nn.Linear(in_size,mid_size),
            nn.GELU())
        self.ResFusion = nn.Sequential(
            nn.Linear(mid_size,512),
            nn.GELU())
        self.ExtractParam1 = nn.Sequential(
            nn.Linear(512,1024))
        self.ExtractParam2 = nn.Sequential(
            nn.Linear(1024,512))
        self.ExtractParam3 = nn.Sequential(
            nn.Linear(512,256))
        self.ExtractParam4 = nn.Sequential(
            nn.Linear(256,128))
        self.GELU = nn.GELU()

    def forward(self, x):

        residual = self.Residual(x)
        latent_with_res = self.Layers(x) + residual
        param0 = self.ResFusion(latent_with_res)
        param1 = self.ExtractParam1(param0)
        hidden1 = self.GELU(param1)
        param2 = self.ExtractParam2(hidden1)
        hidden2 = self.GELU(param2)
        param3 = self.ExtractParam3(hidden2)
        hidden3 = self.GELU(param3)
        param4 = self.ExtractParam4(hidden3)

        return param1, param2, param3, param4


class DecoderBlock(nn.Module):
    """One decoder stage: three (conv + FiLM + GELU) steps, one per parameter group, then ConvUp (x2 resolution)."""
    def __init__(self,in_size, out_size, dropout):
        super(DecoderBlock,self).__init__()

        self.sameconv1 = ConvSame(in_size, in_size, dropout, normalization=True, activation=False)
        self.sameconv2 = ConvSame(in_size, in_size, dropout, normalization=True, activation=False)
        self.sameconv3 = ConvSame(in_size, in_size, dropout, normalization=True, activation=False)
        self.upconv = ConvUp(in_size, out_size, dropout, normalization=True, activation=True)
        self.GELU = nn.GELU()

    def forward(self, x, param, param_, param__):

        B, C, D, H, W = x.shape
        mid = param.shape[1] // 2

        gamma1 = param[:, :mid].view(B, C, 1, 1, 1)
        beta1 = param[:, mid:].view(B, C, 1, 1, 1)
        gamma2 = param_[:, :mid].view(B, C, 1, 1, 1)
        beta2 = param_[:, mid:].view(B, C, 1, 1, 1)
        gamma3 = param__[:, :mid].view(B, C, 1, 1, 1)
        beta3 = param__[:, mid:].view(B, C, 1, 1, 1)

        x2 = self.sameconv1(x)
        x2 = x2 * gamma1 + beta1
        x3 = self.GELU(x2)

        x3 = self.sameconv2(x3)
        x3 = x3 * gamma2 + beta2
        x4 = self.GELU(x3)

        x4 = self.sameconv3(x4)
        x4 = x4 * gamma3 + beta3
        x4 = self.GELU(x4)

        x5 = self.upconv(x4)

        return x5


class AffineUNet3(nn.Module):
    """ViT image encoder + FiLM decoder.

    Needle position, needle direction and ablation time/power are encoded separately;
    each decoder stage is modulated three times (once per group).
    """
    def __init__(self, mode = "dmg"):
        super().__init__()

        self.mode = mode
        self.encoder_image = ViTEncoder(input_dim=2, dropout=0.1, num_layers=12)

        self.decoder_1 = DecoderBlock(512, 256, dropout=0.1)
        self.decoder_2 = DecoderBlock(256, 128, dropout=0.1)
        self.decoder_3 = DecoderBlock(128, 64, dropout=0.1)
        self.decoder_4 = DecoderBlock(64, 32, dropout=0.1)

        if mode == "dmg":
            self.last_layer = nn.Sequential(
                nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1),
                nn.Sigmoid(),
                )
        else :
            self.last_layer = nn.Sequential(
                nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1),
                nn.ReLU())

        self.coord_param = ParamEncoder(60, 256, 5, dropout=0.1)
        self.dir_param = ParamEncoder(60, 256, 5, dropout=0.1)
        self.cond_param = ParamEncoder(40, 256, 5, dropout=0.1)

    def forward(self, img, coord, cond):
        """
        img:   (B, 2, 81, 81, 81)  tumor and vessel masks
        coord: (B, 6)              needle position (3) and direction (3), normalized
        cond:  (B, 2)              ablation time and power, normalized
        returns (B, 81, 81, 81)    damage probability (mode="dmg") or scaled temperature (mode="temp")
        """
        img = F.interpolate(img, [80, 80, 80])  # (B, 2, 80, 80, 80)

        x = self.encoder_image(img)  # (B, 512, 5, 5, 5)

        # position (B, 60), direction (B, 60) and time + power (B, 40) -> FiLM parameters of the 4 decoder stages
        param1, param2, param3, param4 = self.coord_param(PositionalEncoding(10, coord[:,:3]))
        param1_, param2_, param3_, param4_ = self.dir_param(PositionalEncoding(10, coord[:,3:]))
        param1__, param2__, param3__, param4__ = self.cond_param(PositionalEncoding(10, cond))

        x1 = self.decoder_1(x, param1, param1_, param1__)  # (B, 256, 10, 10, 10)
        x2 = self.decoder_2(x1, param2, param2_, param2__)  # (B, 128, 20, 20, 20)
        x3 = self.decoder_3(x2, param3, param3_, param3__)  # (B, 64, 40, 40, 40)

        x4 = self.decoder_4(x3, param4, param4_, param4__)  # (B, 32, 80, 80, 80)
        x5 = self.last_layer(x4)  # (B, 1, 80, 80, 80)

        return F.interpolate(x5, [81, 81, 81]).squeeze(1)  # (B, 81, 81, 81)
