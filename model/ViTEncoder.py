# Adapted from https://github.com/tamasino52/UNETR (modified by the authors of this repository).
# The original code is distributed under the following license:
#
# MIT License
#
# Copyright (c) 2023 Minseok_Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class PositionwiseFeedForward(nn.Module):
    """Feed-forward block of the transformer layer: d_model -> d_ff -> d_model."""
    def __init__(self, d_model=768, d_ff=2048, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.GELU = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.GELU(self.w_1(x))))


class Embeddings(nn.Module):
    """Splits the volume into patch_size^3 patches and embeds each with a learned position embedding."""
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, embed_dim, 5, 5, 5) for an 80^3 input and 16^3 patches
        x = x.flatten(2)  # (B, embed_dim, 125)
        x = x.transpose(-1, -2)  # (B, 125, embed_dim): one token per patch
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerBlock(nn.Module):
    """Pre-norm transformer layer: self-attention and MLP, each with a residual connection."""
    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = PositionwiseFeedForward(embed_dim, 2048, dropout)
        # batch_first=True: inputs are (B, tokens, embed_dim)
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, need_weights=False):
        h = x
        x = self.attention_norm(x)
        # need_weights=False lets PyTorch use its fused attention kernel
        x, weights = self.attn(x, x, x, need_weights=need_weights)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights


class Transformer(nn.Module):
    """Patch embedding, num_layers TransformerBlocks and a final LayerNorm."""
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, num_heads, num_layers, dropout):
        super().__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, cube_size, patch_size, dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, dropout, cube_size, patch_size)
            self.layer.append(layer)

    def forward(self, x):
        hidden_states = self.embeddings(x)
        for layer_block in self.layer:
            hidden_states, _ = layer_block(hidden_states)

        return self.encoder_norm(hidden_states)


class ViTEncoder(nn.Module):
    """3D ViT image encoder.

    (B, input_dim, 80, 80, 80) -> (B, embed_dim, 5, 5, 5) with the default img_shape and patch_size
    """
    def __init__(self, img_shape=(80, 80, 80), input_dim=1, output_dim=1, embed_dim=512, patch_size=16,
                 num_heads=16, dropout=0.1, num_layers=12):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = tuple(img_shape)
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.patch_dim = [int(x / patch_size) for x in img_shape]

        self.transformer = Transformer(
            input_dim,
            embed_dim,
            img_shape,
            patch_size,
            num_heads,
            self.num_layers,
            dropout,
        )

    def forward(self, x):
        # resize to img_shape (80^3 by default) if needed
        if tuple(x.shape[2:]) != self.img_shape:
            x = F.interpolate(x, size=self.img_shape, mode='trilinear', align_corners=False)

        z = self.transformer(x)  # (B, 125, embed_dim)
        z = z.transpose(-1, -2).reshape(-1, self.embed_dim, *self.patch_dim)  # (B, embed_dim, 5, 5, 5)

        return z
