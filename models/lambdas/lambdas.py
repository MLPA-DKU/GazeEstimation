import torch
import torch.nn as nn
import einops


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class LambdaLayer(nn.Module):

    def __init__(self, dim, *, dim_k, n=None, r=None, heads=4, dim_out=None, dim_u=1):
        super(LambdaLayer, self).__init__()
        dim_out = default(dim_out, dim)
        self.u = dim_u  # intra-depth dimension
        self.heads = heads

        assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        dim_v = dim_out // heads

        self.to_q = nn.Conv2d(in_channels=dim, out_channels=dim_k * heads, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(in_channels=dim, out_channels=dim_k * dim_u, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_channels=dim, out_channels=dim_v * dim_u, kernel_size=1, bias=False)

        self.bn_q = nn.BatchNorm2d(num_features=dim_k * heads)
        self.bn_v = nn.BatchNorm2d(num_features=dim_v * dim_u)

        self.local_contexts = exists(r)
        if exists(r):
            assert (r % 2) == 1, 'receptive kernel size should be odd'
            self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r), padding=(0, r // 2, r // 2))
        else:
            assert exists(n), 'you must specify the total sequence length (h x w)'
            self.pos_emb = nn.Parameter(torch.randn(n, n, dim_k, dim_u))

    def forward(self, x):
        b, c, hh, ww = x.shape
        u, h = self.u, self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.bn_q(q)
        v = self.bn_v(v)

        q = einops.rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h=h)
        k = einops.rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u=u)
        v = einops.rearrange(v, 'b (u v) hh ww -> b u v (hh ww)', u=u)

        k = k.softmax(dim=-1)

        λc = torch.einsum('b u k m, b u v m -> b k v', k, v)
        yc = torch.einsum('b h k n, b k v -> b h v n', q, λc)

        if self.local_contexts:
            v = einops.rearrange(v, 'b u v (hh ww) -> b u v hh ww', hh=hh, ww=ww)
            λp = self.pos_conv(v)
            yp = torch.einsum('b h k n, b k v n -> b h v n', q, λp.flatten(3))
        else:
            λp = torch.einsum('n m k u, b u v m -> b n k v', self.pos_emb, v)
            yp = torch.einsum('b h k n, b n k v -> b h v n', q, λp)

        y = yc + yp
        out = einops.rearrange(y, 'b h v (hh ww) -> b (h v) hh ww', hh=hh, ww=ww)
        return out
