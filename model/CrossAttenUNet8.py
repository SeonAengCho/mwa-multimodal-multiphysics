import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.nn import MultiheadAttention

from .ViTEncoder import ViTEncoder


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


class ConvUp(nn.Module):
    """2x2x2 transposed conv with stride 2 (doubles the resolution), then optional InstanceNorm, GELU and 3D dropout."""
    def __init__(self, in_size, out_size, dropout, normalization, activation):
        super(ConvUp, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2)
        ]
        if normalization:
            layers.append(nn.InstanceNorm3d(out_size))
        if activation:
            layers.append(nn.GELU())
        if dropout:
            layers.append(nn.Dropout3d(dropout))
        self.Layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.Layers(x)


class ConvUp_withConv(nn.Module):
    """ConvUp with an extra 3x3x3 conv after the transposed conv. Used on the skip connections."""
    def __init__(self, in_size, out_size, dropout, normalization, activation):
        super(ConvUp_withConv, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2),
            nn.Conv3d(out_size, out_size, 3, 1, 1, bias=False)
        ]
        if normalization:
            layers.append(nn.InstanceNorm3d(out_size))
        if activation:
            layers.append(nn.GELU())
        if dropout:
            layers.append(nn.Dropout3d(dropout))
        self.Layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.Layers(x)


class ConvSame(nn.Module):
    """3x3x3 conv at the same resolution, then optional InstanceNorm, GELU and 3D dropout."""
    def __init__(self, in_size, out_size, dropout, normalization, activation):
        super(ConvSame, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, 3, 1, 1, bias=False)]
        if normalization:
            layers.append(nn.InstanceNorm3d(out_size))
        if activation:
            layers.append(nn.GELU())
        if dropout:
            layers.append(nn.Dropout3d(dropout))
        self.Layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.Layers(x)


class MLP(nn.Module):
    """Feed-forward block of the attention layer: d_model -> d_ff -> d_model."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.gelu(self.w_1(x))))


class LinearProjectionBlock(nn.Module):
    """Mixes the 3 parameter tokens into output_len tokens: (3, B, dim) -> (output_len, B, dim)."""
    def __init__(self, input_len=3, output_len=64, dim=512):

        super().__init__()

        self.proj = nn.Linear(input_len, output_len)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):

        x = x.permute(1, 2, 0)
        x = self.proj(x)
        x = x.permute(2, 0, 1)
        x = self.ln(self.act(x))

        return x


class SelfAndCrossAtt(nn.Module):
    """Pre-norm attention layer: self-attention over the image tokens, cross-attention to the
    parameter tokens, then an MLP, each with a residual connection. Tensors are (tokens, B, 512).
    """
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()

        self.query_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.key_norm   = nn.LayerNorm(embed_dim, eps=1e-6)
        self.self_norm  = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.cross_attn = MultiheadAttention(embed_dim, num_heads, dropout)
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, 1024)

    def forward(self, image_embed, vector_embed):
        # self-attention over the image tokens
        self_input = self.self_norm(image_embed)
        self_output, _ = self.self_attn(self_input, self_input, self_input)
        hidden_embed = self_output + image_embed

        # cross-attention: image tokens (query) attend to the parameter tokens (key, value)
        query = self.query_norm(hidden_embed)
        key = self.key_norm(vector_embed)
        value = self.key_norm(vector_embed)
        cross_output, _ = self.cross_attn(query, key, value)
        hidden_embed = cross_output + hidden_embed

        hidden_embed = self.mlp(self.mlp_norm(hidden_embed)) + hidden_embed
        return hidden_embed


class CA_Fusion(nn.Module):
    """Stack of layer_num SelfAndCrossAtt layers.

    Returns the outputs of the layers listed in extract_layers (1-based), each (B, 512, 5, 5, 5).
    """
    def __init__(self, layer_num, extract_layers, proj, dropout=0.1):
        super(CA_Fusion, self).__init__()

        self.extract_layers = extract_layers

        if proj == 0:
            self.proj = nn.Identity()
        else :
            self.proj = LinearProjectionBlock(input_len=3, output_len=proj, dim=512)

        self.layers = nn.ModuleList()
        for _ in range(layer_num):
            self.layers.append(SelfAndCrossAtt(512, 16, dropout))

    def forward(self, image_embed, coord_latent, cond_latent, dir_latent):
        B, C, D, H, W = image_embed.shape  # (B, 512, 5, 5, 5)

        # one token per voxel: (125, B, 512)
        image_embed = image_embed.view(B, C, -1).permute(2, 0, 1)

        # parameter tokens: (3, B, 512), or (proj, B, 512) if proj > 0
        vector_embed = torch.cat((coord_latent, cond_latent, dir_latent), dim=1)
        vector_embed = vector_embed.permute(1, 0, 2)
        vector_embed = self.proj(vector_embed)

        extracted_layers = []
        for depth, layer in enumerate(self.layers):
            image_embed = layer(image_embed, vector_embed)
            # keep the outputs of the layers in extract_layers, reshaped back to (B, 512, 5, 5, 5)
            if depth + 1 in self.extract_layers:
                extracted_layers.append(image_embed.permute(1, 2, 0).view(B, C, D, H, W))

        return extracted_layers


class ParamEncoder(nn.Module):
    """MLP that maps one encoded parameter group to a token: in_size -> mid_size -> out_size.

    layer_num is the total number of Linear layers.
    """
    def __init__(self, in_size, mid_size, out_size, layer_num, dropout):
        super(ParamEncoder, self).__init__()

        layers = []

        # first layer: in_size -> mid_size
        layers.append(nn.Linear(in_size, mid_size))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        # hidden layers: mid_size -> mid_size
        for _ in range(layer_num - 2):
            layers.append(nn.Linear(mid_size, mid_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        # last layer: mid_size -> out_size
        layers.append(nn.Linear(mid_size, out_size))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        self.Layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.Layers(x)


class Decoder_no(nn.Module):
    """Decoder without skip connections (type="noskip"): upsamples the last attention output only.

    extracted_layers: 4 x (B, 512, 5, 5, 5) -> (B, 32, 80, 80, 80)
    """
    def __init__(self, dropout):
        super().__init__()

        self.up1 = ConvUp(512, 256, dropout, normalization=True, activation=True)
        self.up2 = ConvUp(256, 128, dropout, normalization=True, activation=True)
        self.up3 = ConvUp(128, 64, dropout, normalization=True, activation=True)
        self.up4 = ConvUp(64, 32, dropout, normalization=True, activation=True)
        self.up_last = ConvSame(32, 32, dropout, normalization=True, activation=True)

    def forward(self, extracted_layers):

        z3, z6, z9, z12 = extracted_layers  # each (B, 512, 5, 5, 5)

        x = self.up1(z12)  # (B, 256, 10, 10, 10)
        x = self.up2(x)  # (B, 128, 20, 20, 20)
        x = self.up3(x)  # (B, 64, 40, 40, 40)
        x = self.up4(x)  # (B, 32, 80, 80, 80)

        x = self.up_last(x)  # (B, 32, 80, 80, 80)

        return x


class Decoder_36912(nn.Module):
    """Decoder with skip connections from all 4 extracted attention outputs (type="skipall").

    extracted_layers: 4 x (B, 512, 5, 5, 5), shallowest first -> (B, 32, 80, 80, 80).
    z3, z6, z9, z12 are named after the extracted layers of CrossAttenUNet12.
    """
    def __init__(self, dropout):
        super().__init__()

        self.up1 = ConvUp(512, 256, dropout, normalization=True, activation=True)
        self.up2 = ConvUp(512, 128, dropout, normalization=True, activation=True)
        self.up3 = ConvUp(256, 64, dropout, normalization=True, activation=True)
        self.up4 = ConvUp(128, 32, dropout, normalization=True, activation=True)
        self.up_last = ConvSame(32, 32, dropout, normalization=True, activation=True)

        self.skip1 = ConvUp_withConv(512, 256, dropout, normalization=True, activation=True)
        self.skip2 = nn.Sequential(ConvUp_withConv(512, 256, dropout, normalization=True, activation=True),
                                   ConvUp_withConv(256, 128, dropout, normalization=True, activation=True))
        self.skip3 = nn.Sequential(ConvUp_withConv(512, 256, dropout, normalization=True, activation=True),
                                   ConvUp_withConv(256, 128, dropout, normalization=True, activation=True),
                                   ConvUp_withConv(128, 64, dropout, normalization=True, activation=True))

    def forward(self, extracted_layers):

        z3, z6, z9, z12 = extracted_layers  # each (B, 512, 5, 5, 5)

        z9 = self.skip1(z9)  # (B, 256, 10, 10, 10)
        z6 = self.skip2(z6)  # (B, 128, 20, 20, 20)
        z3 = self.skip3(z3)  # (B, 64, 40, 40, 40)

        x = self.up1(z12)  # (B, 256, 10, 10, 10)
        x = self.up2(torch.cat([z9, x], dim=1))  # (B, 128, 20, 20, 20)
        x = self.up3(torch.cat([z6, x], dim=1))  # (B, 64, 40, 40, 40)
        x = self.up4(torch.cat([z3, x], dim=1))  # (B, 32, 80, 80, 80)

        x = self.up_last(x)  # (B, 32, 80, 80, 80)

        return x


class Decoder_612(nn.Module):
    """Decoder with one skip connection from the 2nd extracted attention output (type="oneskip").

    extracted_layers: 4 x (B, 512, 5, 5, 5) -> (B, 32, 80, 80, 80)
    """
    def __init__(self, dropout):
        super().__init__()

        self.up1 = ConvUp(512, 256, dropout, normalization=True, activation=True)
        self.up2 = ConvUp(256, 128, dropout, normalization=True, activation=True)
        self.up3 = ConvUp(256, 64, dropout, normalization=True, activation=True)
        self.up4 = ConvUp(64, 32, dropout, normalization=True, activation=True)
        self.up_last = ConvSame(32, 32, dropout, normalization=True, activation=True)

        self.skip2 = nn.Sequential(ConvUp(512, 256, dropout, normalization=True, activation=True),
                                   ConvUp(256, 128, dropout, normalization=True, activation=True))

    def forward(self, extracted_layers):

        z3, z6, z9, z12 = extracted_layers  # each (B, 512, 5, 5, 5)

        z6 = self.skip2(z6)  # (B, 128, 20, 20, 20)

        x = self.up1(z12)  # (B, 256, 10, 10, 10)
        x = self.up2(x)  # (B, 128, 20, 20, 20)
        x = self.up3(torch.cat([z6, x], dim=1))  # (B, 64, 40, 40, 40)
        x = self.up4(x)  # (B, 32, 80, 80, 80)

        x = self.up_last(x)  # (B, 32, 80, 80, 80)

        return x


class ImageEncoder(nn.Module):
    """CNN image encoder: four ConvDown blocks, (B, 2, 80, 80, 80) -> (B, 512, 5, 5, 5)."""
    def __init__(self, dropout):
        super(ImageEncoder,self).__init__()
        self.Layers = nn.Sequential(
            ConvDown(2, 64, dropout, normalization=True, activation=True),
            ConvDown(64, 128, dropout, normalization=True, activation=True),
            ConvDown(128, 256, dropout, normalization=True, activation=True),
            ConvDown(256, 512, dropout, normalization=True, activation=True))

    def forward(self,x):
        x = self.Layers(x)
        return x


class CrossAttenUNet8(nn.Module):
    """Image encoder + 8 cross-attention layers (image tokens <-> parameter tokens) + U-Net-style decoder.

    skip_layers: attention layers whose outputs feed the decoder (default [2, 4, 6, 8])
    type:    "skipall" (4 skip connections), "oneskip" (1) or "noskip" (0)
    proj:    0 = 3 parameter tokens (position, conditions, direction); > 0 = mixed into proj tokens
    encoder: "CNN" or "ViT"
    """
    def __init__(self, skip_layers=[2,4,6,8], mode=None, type = "skipall", proj = 0, encoder = "CNN"):
        super(CrossAttenUNet8, self).__init__()

        self.skip_layers = skip_layers
        self.mode = mode
        self.proj = proj
        self.model_type = type
        self.encoder = encoder
        self.embed_dim = 512
        self.dropout = 0.1

        if encoder == "ViT":
            self.encoder_image = ViTEncoder(input_dim=2, embed_dim=self.embed_dim, patch_size=16, dropout=0.3, num_layers=6)
        else:
            self.encoder_image = ImageEncoder(dropout=0.1)

        self.coord_param = ParamEncoder(60, 256, 512, 6, dropout=0.1)
        self.dir_param = ParamEncoder(60, 256, 512,6, dropout=0.1)
        self.cond_param = ParamEncoder(40, 256, 512,6, dropout=0.1)

        self.CrossAttenion = CA_Fusion(layer_num=8,
                                       extract_layers=self.skip_layers,
                                       proj=self.proj,
                                       dropout=0.3)

        if self.model_type == 'noskip':
            self.decoder = Decoder_no(dropout=self.dropout)
        elif self.model_type == 'oneskip':
            self.decoder = Decoder_612(dropout=self.dropout)
        else:
            self.decoder = Decoder_36912(dropout=self.dropout)

        if self.mode == "dmg":
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

        image_embed = self.encoder_image(img)  # (B, 512, 5, 5, 5)

        # parameter tokens, each (B, 1, 512): position (60), direction (60), time + power (40) -> 512
        coord_latent = self.coord_param(PositionalEncoding(10, coord[:,:3], max_cycles=16)).unsqueeze(1)
        dir_latent = self.dir_param(PositionalEncoding(10, coord[:,3:], max_cycles=16)).unsqueeze(1)
        cond_latent = self.cond_param(PositionalEncoding(10, cond, max_cycles=16)).unsqueeze(1)

        extracted_layers = self.CrossAttenion(image_embed, coord_latent, cond_latent, dir_latent)  # 4 x (B, 512, 5, 5, 5)

        x = self.decoder(extracted_layers)  # (B, 32, 80, 80, 80)
        x = self.last_layer(x)  # (B, 1, 80, 80, 80)

        return F.interpolate(x, [81, 81, 81]).squeeze(1)  # (B, 81, 81, 81)
