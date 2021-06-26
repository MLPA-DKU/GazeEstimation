import torch
import torch.nn as nn
import einops as eo


class DSConv(nn.Module):

    def __init__(self, in_channels, activation=nn.GELU()):
        super(DSConv, self).__init__()
        self.depth_wise = nn.Conv2d(in_channels, in_channels, (3, 3), (2, 2), padding=1, groups=in_channels)
        self.point_wise = nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), padding=0)
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.9997, eps=4e-5)
        self.activation = activation

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


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

    def __init__(self, in_channels, heads=8, activation=nn.GELU()):
        super(Attention, self).__init__()
        self.heads = heads
        self.scale = (in_channels // heads) ** -0.5

        self.to_qkv = nn.Sequential(
            DSConv(in_channels),
            DSConv(in_channels),
            DSConv(in_channels),
            nn.Conv2d(in_channels, in_channels * 3, (1, 1), (1, 1), padding=0),
            nn.BatchNorm2d(in_channels * 3),
            activation,
        )
        self.adjust = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, (3, 3), (2, 2), padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels, in_channels, (3, 3), (2, 2), padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels, in_channels, (3, 3), (2, 2), padding=1, output_padding=1),
        )

    def forward(self, f):
        heads = self.heads
        b, c, h, w = f.shape
        q, k, v = self.to_qkv(f).chunk(3, dim=1)
        q, k, v = map(lambda t: eo.rearrange(t, 'b (h c) x y -> b h (x y) c', h=heads), (q, k, v))

        dots = torch.einsum('b h i c, b h j c -> b h i j', q, k)
        dots *= self.scale
        attn = dots.softmax(dim=-1)

        fmap = torch.einsum('b h i j, b h j c -> b h i c', attn, v)
        fmap = eo.rearrange(fmap, 'b h (x y) c -> b (h c) x y', x=h // 8, y=w // 8)
        fmap = self.adjust(fmap)
        return fmap


class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels, heads=8, activation=nn.GELU()):
        super(Encoder, self).__init__()
        self.attn = Attention(in_channels, heads=heads, activation=activation)
        self.ff = FeedForward(in_channels, out_channels)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x)
        return x


class XiT(nn.Module):

    def __init__(self, channels, depth, heads=8, activation=nn.GELU()):
        super(XiT, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], (7, 7), (2, 2), padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.GELU())
        self.layers = nn.Sequential(*[Encoder(channels[i], channels[i + 1], heads=heads, activation=activation) for i in range(depth)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = torch.flatten(self.pool(x), 1)
        x = self.fc(x)
        return x


# if __name__ == '__main__':
#     device = 'cuda:0'
#     t = torch.rand((1, 3, 224, 224), device=device)
#     model = XiT([64, 64, 64, 128, 128, 256, 256], depth=6, heads=8)
#     model.to(device=device)
#     predict = model(t)
#     breakpoint()
