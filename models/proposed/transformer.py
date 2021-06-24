import torch
import torch.nn as nn
import einops as eo


class Attention(nn.Module):

    def __init__(self, *, dim, heads=8, dim_head=8):
        super(Attention, self).__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=(1, 1), bias=False)
        self.q_conv = nn.Sequential(nn.Conv2d(inner_dim, inner_dim, (11, 11), (8, 8), padding=2), nn.GELU())
        self.k_conv = nn.Sequential(nn.Conv2d(inner_dim, inner_dim, (11, 11), (8, 8), padding=2), nn.GELU())
        self.adjust = nn.Upsample(scale_factor=64)

    def forward(self, f):
        heads = self.heads
        b, c, h, w = f.shape
        q, k, v = self.to_qkv(f).chunk(3, dim=1)
        q, k, v = self.q_conv(q), self.k_conv(k), v
        q, k, v = map(lambda t: eo.rearrange(t, 'b (h c) x y -> b h c (x y)', h=heads), (q, k, v))
        q *= self.scale

        dots = torch.einsum('b h c i, b h c j -> b h i j', q, k)
        dots = self.adjust(dots)
        attn = dots.softmax(dim=-1)

        fmap = torch.einsum('b h i j, b h c j -> b h c i', attn, v)
        fmap = eo.rearrange(fmap, 'b h c (x y) -> b (h c) x y', x=h, y=w)
        return fmap
