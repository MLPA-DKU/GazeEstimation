import torch
import torch.nn as nn
import einops as eo


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class AbsPosEmb(nn.Module):

    def __init__(self, f, dim_head, device='cpu'):
        super(AbsPosEmb, self).__init__()
        height, width = pair(f)
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(dim_head, height, device=device) * scale)
        self.width = nn.Parameter(torch.randn(dim_head, width, device=device) * scale)

    def forward(self, q):
        emb = eo.rearrange(self.height, 'c h -> c h ()') + eo.rearrange(self.width, 'c w -> c () w')
        emb = eo.rearrange(emb, 'c h w -> c (h w)')
        logits = torch.einsum('b h c i, c j -> b h i j', q, emb)
        return logits


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

        self.pos_fn = AbsPosEmb

    def forward(self, f):
        heads = self.heads
        b, c, h, w = f.shape
        pos_emb = self.pos_fn(h, self.dim_head, f.device)
        q, k, v = self.to_qkv(f).chunk(3, dim=1)
        pos_emb = pos_emb(eo.rearrange(q, 'b (h c) x y -> b h c (x y)', h=heads))
        q, k, v = self.q_conv(q), self.k_conv(k), v
        q, k, v = map(lambda t: eo.rearrange(t, 'b (h c) x y -> b h c (x y)', h=heads), (q, k, v))
        q *= self.scale

        dots = torch.einsum('b h c i, b h c j -> b h i j', q, k)
        dots = self.adjust(dots) + pos_emb
        attn = dots.softmax(dim=-1)

        fmap = torch.einsum('b h i j, b h c j -> b h c i', attn, v)
        fmap = eo.rearrange(fmap, 'b h c (x y) -> b (h c) x y', x=h, y=w)
        return fmap


class Encoder(nn.Module):

    def __init__(self, dim, heads=8, dim_head=8, reduction=16, activation=nn.GELU()):
        super(Encoder, self).__init__()
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head)
        self.ff = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, (1, 1), bias=False),
            nn.BatchNorm2d(dim // reduction),
            activation,
            nn.Conv2d(dim // reduction, dim, (1, 1), bias=False),
            nn.BatchNorm2d(dim),
            activation
        )

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class GazeTR(nn.Module):

    def __init__(self, dim, depth, num_classes=2):
        super(GazeTR, self).__init__()
        self.layers = nn.Sequential(*[Encoder(64) for _ in range(depth)])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = self.fc(torch.flatten(self.gap(x)))
        return x


if __name__ == '__main__':
    d = torch.rand((1, 64, 112, 112))  # feature map
    d = d.to('cuda:0')
    model = GazeTR(64, 6)
    model.to('cuda:0')
    o = model(d)
    breakpoint()
