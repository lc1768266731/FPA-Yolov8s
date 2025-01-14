#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = "ADM"


# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 第一层全连接
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        # 第二层全连接
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取通道级的全局描述
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        # 通过全连接层生成通道注意力权重
        avg_out = self.fc1(avg_out)
        avg_out = F.relu(avg_out)
        avg_out = self.fc2(avg_out)
        avg_out = self.sigmoid(avg_out).view(x.size(0), x.size(1), 1, 1)
        # 应用通道注意力
        return x * avg_out


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ADM(nn.Module):
    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=True, attention=True):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        self.attention = attention

        assert style in ["lp", "pl"]
        if style == "pl":
            assert in_channels >= scale**2 and in_channels % scale**2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        # 通道注意力机制
        if self.attention:
            self.channel_attention = ChannelAttention(in_channels)

        if style == "pl":
            in_channels = in_channels // scale**2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale**2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.0)

        # 初始位置
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        # 定义基于 scale 的网格位置
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        grid = torch.meshgrid(h, h, indexing='xy')
        init_pos = torch.stack(grid, dim=-1)
        init_pos = init_pos.transpose(1, 2)
        init_pos = init_pos.repeat(1, self.groups, 1)
        init_pos = init_pos.reshape(1, -1, 1, 1)
        return init_pos

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h]))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device)
        )
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(
            1, 2, 1, 1, 1
        )
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if self.attention:
            x = self.channel_attention(x)  # 应用通道注意力
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if self.attention:
            x_ = self.channel_attention(x_)  # 应用通道注意力
        if hasattr(self, "scope"):
            offset = (
                F.pixel_unshuffle(
                    self.offset(x_) * self.scope(x_).sigmoid(), self.scale
                )
                * 0.5
                + self.init_pos
            )
        else:
            offset = (
                F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
            )
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == "pl":
            return self.forward_pl(x)
        return self.forward_lp(x)


if __name__ == "__main__":
    x = torch.rand(2, 64, 20, 20)
    dys = ADM(64)
    print(dys(x).shape)
