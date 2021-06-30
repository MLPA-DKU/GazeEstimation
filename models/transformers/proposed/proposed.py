import torch
import torch.nn as nn
import einops as eo


device = 'cuda:1'

batch_size = 1
in_channels = 64
resolutions = 112
t = torch.rand((batch_size, in_channels, resolutions, resolutions), device=device)  # dummy feature map

heads = 8  # number of heads in multi-head self-attention
head_channels = 8  # number of channels per head
inner_dim = heads * head_channels

to_qkv = nn.Conv2d(in_channels, inner_dim * 3, kernel_size=(1, 1), bias=False, device=device)
q_conv = nn.Sequential(nn.Conv2d(inner_dim, inner_dim, (11, 11), (8, 8), padding=2), nn.GELU()).to(device=device)
k_conv = nn.Sequential(nn.Conv2d(inner_dim, inner_dim, (11, 11), (8, 8), padding=2), nn.GELU()).to(device=device)

q, k, v = to_qkv(t).chunk(3, dim=1)
q, k, v = q_conv(q), k_conv(k), v
q, k, v = map(lambda t: eo.rearrange(t, 'b (h c) x y -> b h c x y', h=heads), (q, k, v))
breakpoint()
