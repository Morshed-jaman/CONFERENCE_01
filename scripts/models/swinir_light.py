"""Compact SwinIR-style backbone for same-resolution grayscale SR.

The HC18 LQ images are already resized to the HR grid, so this model predicts a
residual correction at the same spatial resolution instead of upscaling.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition [B, H, W, C] into [B*nW, window, window, C]."""
    batch, height, width, channels = x.shape
    x = x.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, channels)


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Reverse window partition into [B, H, W, C]."""
    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch, height, width, -1)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        bias_shape = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(bias_shape, num_heads))

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_windows,
            tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        query = query * self.scale
        attention = query @ key.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attention = attention.view(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                tokens,
                tokens,
            )
            attention = attention + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, tokens, tokens)

        attention = F.softmax(attention, dim=-1)
        x = (attention @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        if shift_size >= window_size:
            raise ValueError("shift_size must be smaller than window_size")
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_dim=int(dim * mlp_ratio))

    def calculate_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        img_mask = torch.zeros((1, height, width, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(attention_mask != 0, -100.0)
        return attention_mask.masked_fill(attention_mask == 0, 0.0)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        height, width = x_size
        batch, length, channels = x.shape
        if length != height * width:
            raise ValueError(f"Token length {length} does not match H*W={height * width}")

        shortcut = x
        x = self.norm1(x)
        x = x.view(batch, height, width, channels)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attention_mask = self.calculate_mask(height, width, x.device)
        else:
            shifted_x = x
            attention_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)
        attention_windows = self.attn(x_windows, mask=attention_mask)
        attention_windows = attention_windows.view(-1, self.window_size, self.window_size, channels)
        shifted_x = window_reverse(attention_windows, self.window_size, height, width)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(batch, height * width, channels)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2).contiguous()


class PatchUnEmbed(nn.Module):
    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        height, width = x_size
        batch, _, channels = x.shape
        return x.transpose(1, 2).contiguous().view(batch, channels, height, width)


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        use_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if index % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                )
                for index in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and x.requires_grad:
                def custom_forward(tokens: torch.Tensor) -> torch.Tensor:
                    return block(tokens, x_size)

                try:
                    x = checkpoint(custom_forward, x, use_reentrant=False)
                except TypeError:
                    x = checkpoint(custom_forward, x)
            else:
                x = block(x, x_size)
        return x


class ResidualSwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        use_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.residual_group = BasicLayer(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            use_checkpoint=use_checkpoint,
        )
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        residual = self.residual_group(x, x_size)
        residual = self.patch_unembed(residual, x_size)
        residual = self.conv(residual)
        residual = self.patch_embed(residual)
        return x + residual


class SwinIRLight(nn.Module):
    """SwinIR-Light for grayscale same-resolution super-resolution."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 60,
        depths: Sequence[int] = (6, 6, 6, 6),
        num_heads: Sequence[int] = (6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        upscale: int = 1,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if upscale != 1:
            raise ValueError("SwinIRLight in this project only supports same-resolution upscale=1")
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length")
        self.window_size = int(window_size)
        self.conv_first = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1)
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()
        self.layers = nn.ModuleList(
            [
                ResidualSwinTransformerBlock(
                    dim=embed_dim,
                    depth=int(depth),
                    num_heads=int(heads),
                    window_size=self.window_size,
                    mlp_ratio=mlp_ratio,
                    use_checkpoint=use_checkpoint,
                )
                for depth, heads in zip(depths, num_heads)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv_hr = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(embed_dim, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def check_image_size(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        _, _, height, width = x.shape
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h == 0 and pad_w == 0:
            return x, height, width
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), height, width

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        tokens = self.patch_embed(x)
        for layer in self.layers:
            tokens = layer(tokens, x_size)
        tokens = self.norm(tokens)
        return self.patch_unembed(tokens, x_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, original_h, original_w = self.check_image_size(x)
        shallow = self.conv_first(x)
        body = self.conv_after_body(self.forward_features(shallow)) + shallow
        residual = self.conv_last(self.act(self.conv_hr(body)))
        output = torch.clamp(x + residual, 0.0, 1.0)
        return output[:, :, :original_h, :original_w]


def build_swinir_light(
    window_size: int = 8,
    depths: Sequence[int] = (6, 6, 6, 6),
    use_checkpoint: bool = True,
) -> SwinIRLight:
    return SwinIRLight(
        embed_dim=60,
        depths=depths,
        num_heads=(6,) * len(depths),
        window_size=window_size,
        mlp_ratio=2.0,
        upscale=1,
        use_checkpoint=use_checkpoint,
    )
