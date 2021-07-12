import torch
import torch.nn as nn

from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange


# helper methods


# classes

class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class DepthwiseConv2d(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride,
            padding,
            bias,
            activation=nn.GELU(),
        ):
        super(DepthwiseConv2d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(in_channels),
            activation,
        )

    def forward(self, x):
        return self.net(x)


class TransposedConv2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            activation=nn.GELU(),
        ):
        super(TransposedConv2d, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            activation,
        )


class FeedForward(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=16, activation=nn.GELU()):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            activation,
            nn.Conv2d(in_channels // reduction, out_channels, (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
        )

    def forward(self, x):
        x = self.ff(x)
        return x


class Attention(nn.Module):

    def __init__(self, in_channels, proj_kernel, proj_stride, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_channels = heads * dim_head
        self.proj_stride = proj_stride
        self.padding = proj_kernel // 2
        self.heads = heads
        self.scale = (in_channels // heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Sequential(
            DepthwiseConv2d(in_channels, in_channels, kernel_size=proj_kernel, stride=proj_stride, padding=self.padding, bias=False),
            DepthwiseConv2d(in_channels, in_channels, kernel_size=proj_kernel, stride=proj_stride, padding=self.padding, bias=False),
            DepthwiseConv2d(in_channels, in_channels * 3, kernel_size=proj_kernel, stride=proj_stride, padding=self.padding, bias=False),
        )
        self.adjust = nn.Sequential(
            TransposedConv2d(in_channels, in_channels, kernel_size=proj_kernel, stride=proj_stride, padding=self.padding, output_padding=padding),
            TransposedConv2d(in_channels, in_channels, kernel_size=proj_kernel, stride=proj_stride, padding=self.padding, output_padding=padding),
            TransposedConv2d(in_channels, in_channels, kernel_size=proj_kernel, stride=proj_stride, padding=self.padding, output_padding=padding),
        )
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_channels, in_channels, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, f):
        b, c, h, w, heads = *f.shape, self.heads
        q, k, v = self.to_qkv(f).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=heads), (q, k, v))

        dots = einsum('b h i c, b h j c -> b h i j', q, k)
        dots *= self.scale
        attn = self.attend(dots)

        fmap = einsum('b h i j, b h j c -> b h i c', attn, v)
        fmap = rearrange(fmap, 'b h (x y) c -> b (h c) x y', x=h // self.proj_stride ** 3, y=w // self.proj_stride ** 3)
        fmap = self.adjust(fmap)
        return self.to_out(fmap)
