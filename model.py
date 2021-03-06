import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random

''' Device type'''
dev = 'cpu'
if torch.cuda.is_available():
    dev = 'cuda'


class ModulatedConv(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, w_dim=512, pad=1, upsample=True, modulate=True,
                 demodulate=True):
        super(ModulatedConv, self).__init__()
        self.fan_in = in_ch * k_size * k_size

        weight = torch.randn(1, out_ch, in_ch, k_size, k_size)
        self.weight = nn.Parameter(weight)
        self.linear = EqualizedLinear(w_dim, in_ch)

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k_size = k_size

        self.upsample = upsample
        self.modulate = modulate
        self.demodulate = demodulate
        self.pad = pad

    def forward(self, x, w):
        weight = self.weight * math.sqrt(2 / self.fan_in)
        w_out = self.linear(w)
        if self.modulate:
            weight = w_out.unsqueeze(1).unsqueeze(3).unsqueeze(4) * weight  # [B, out, in, K, K]

        if self.demodulate:
            sqare_value = (weight ** 2).sum(dim=[2, 3, 4], keepdim=True)
            weight = weight / torch.sqrt(sqare_value + 1e-8)

        batch, in_ch, h, w = x.shape
        x = x.view(1, batch * in_ch, h, w)
        if self.upsample:
            weight = weight.transpose(1, 2).reshape(batch*in_ch, self.out_ch, self.k_size,
                                                            self.k_size)
            out = F.conv_transpose2d(x, weight, stride=2, padding=0, groups=batch)
        else:
            weight = weight.reshape(batch*self.out_ch, in_ch, self.k_size, self.k_size)
            out = F.conv2d(x, weight, stride=1, padding=self.pad, groups=batch)

        _, _, h, w = out.shape
        out = out.view(batch, self.out_ch, h, w)
        return out


class LayerEpilogue(nn.Module):
    def __init__(self, in_ch):
        super(LayerEpilogue, self).__init__()
        bias = torch.zeros(1, in_ch, 1, 1)
        self.bias = nn.Parameter(bias)
        self.noise_injection = NoiseInjection(in_ch=in_ch)

    def forward(self, x):
        out = x + self.bias
        out = self.noise_injection(out)
        return out


class SynthesisConstBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, w_dim=512):
        super(SynthesisConstBlock, self).__init__()
        self.stride = stride
        self.padding = padding

        self.modulated_conv = ModulatedConv(in_ch, out_ch, k_size, w_dim, upsample=False)
        self.layer_epilogue = LayerEpilogue(out_ch)

    def forward(self, x, w):
        out = self.modulated_conv(x, w)
        out = self.layer_epilogue(out)
        return out


class SynthesisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size1=3, pad1=1, k_size2=3, pad2=1, w_dim=512):
        super(SynthesisBlock, self).__init__()
        self.modulated_conv1 = ModulatedConv(in_ch, in_ch, k_size1, w_dim)
        self.blur = Blur(in_ch)
        self.layer_epilogue1 = LayerEpilogue(in_ch)

        self.modulated_conv2 = ModulatedConv(in_ch, out_ch, k_size2, w_dim, upsample=False)
        self.layer_epilogue2 = LayerEpilogue(out_ch)

        self.pad1 = pad1
        self.pad2 = pad2

    def forward(self, x, w1, w2):
        out = self.modulated_conv1(x, w1)
        out = self.blur(out)
        out = self.layer_epilogue1(out)

        out = self.modulated_conv2(out, w2)
        out = self.layer_epilogue2(out)
        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x):
        out = x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)
        return out


class StdConcat(nn.Module):
    def __init__(self):
        super(StdConcat, self).__init__()

    def forward(self, x):
        mean_std = torch.mean(x.std(0))
        mean_std = mean_std.expand(x.size(0), 1, 4, 4)
        out = torch.cat([x, mean_std], dim=1)
        return out


class EqualizedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, padding):
        super(EqualizedConv2d, self).__init__()
        weight = torch.randn(out_ch, in_ch, k_size, k_size)
        bias = torch.zeros(out_ch)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.fan_in = in_ch * k_size * k_size
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        out = F.conv2d(x, self.weight * math.sqrt(2 / self.fan_in), self.bias, stride=self.stride,
                       padding=self.padding)
        return out


class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EqualizedLinear, self).__init__()
        weight = torch.randn(out_dim, in_dim)
        bias = torch.zeros(out_dim)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.fan_in = in_dim

    def forward(self, x):
        out = F.linear(x, self.weight * math.sqrt(2 / self.fan_in), self.bias)
        return out


class AffineBlock(nn.Module):
    def __init__(self, in_dim, out_dim, n_slope=0.2, non_linear='relu'):
        super(AffineBlock, self).__init__()
        layers = []
        layers.append(EqualizedLinear(in_dim, out_dim))

        if non_linear == 'relu':
            layers.append(nn.ReLU())
        elif non_linear == 'leaky_relu':
            layers.append(nn.LeakyReLU(negative_slope=n_slope))

        self.affine_block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.affine_block(x)
        return out


class NoiseInjection(nn.Module):
    def __init__(self, in_ch):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, in_ch, 1, 1))

    def forward(self, x):
        batch, c, h, w = x.shape
        noise = torch.randn(batch, c, h, w).to(dev)
        out = x + self.weight * noise
        return out


class Blur(nn.Module):
    def __init__(self, ch, weight=[1, 3, 3, 1], stride=1, normalized=True):
        super(Blur, self).__init__()
        weight = torch.tensor(weight, dtype=torch.float32)
        weight = weight.view(weight.size(0), 1) * weight.view(1, weight.size(0))

        if normalized:
            weight = weight / weight.sum()

        weight = weight.view(1, 1, weight.size(0), weight.size(0))
        self.register_buffer('weight', weight.repeat(ch, 1, 1, 1))
        self.stride = stride

    def forward(self, x):
        out = F.conv2d(x, self.weight, stride=self.stride, padding=int((self.weight.size(3) - 1)/2),
                       groups=x.size(1))
        return out


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize_1, pad_1, ksize_2, pad_2, n_slope=0.2,
                 fused_downsample=True):
        super(DownBlock, self).__init__()
        self.fused_downsample = fused_downsample
        self.conv1 = EqualizedConv2d(in_ch, in_ch, k_size=ksize_1, stride=1, padding=pad_1)
        self.lrelu = nn.LeakyReLU(negative_slope=n_slope)

        self.conv2 = EqualizedConv2d(in_ch, out_ch, k_size=ksize_2, stride=2, padding=pad_2)
        self.blur = Blur(out_ch, weight=[1, 2, 1])
        self.ksize_2 = ksize_2

    def forward(self, x):
        out = self.conv1(x)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.blur(out)
        out = self.lrelu(out)
        return out


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, n_mapping=8, w_dim=512, normalize=True):
        super(MappingNetwork, self).__init__()
        self.z_dim = z_dim
        blocks = []

        if normalize:
            blocks.append(PixelNorm())

        for i in range(n_mapping):
            if i == 0:
                blocks.append(AffineBlock(z_dim, w_dim, n_slope=0.2, non_linear='leaky_relu'))
            else:
                blocks.append(AffineBlock(w_dim, w_dim, n_slope=0.2, non_linear='leaky_relu'))

        self.mapping_network = nn.Sequential(*blocks)

    def forward(self, z):
        out = self.mapping_network(z)
        return out


class Generator(nn.Module):
    def __init__(self, channel_list=[512, 512, 512, 512, 256, 128, 64, 32], style_mixing_prob=0.9):
        super(Generator, self).__init__()
        self.style_mixing_prob = style_mixing_prob
        self.n_layer = len(channel_list)
        progress_layers = []
        to_rgb_layers = []
        for i in range(len(channel_list)):
            if i == 0:
                progress_layers.append(SynthesisConstBlock(channel_list[i], channel_list[i]))
            else:
                progress_layers.append(SynthesisBlock(channel_list[i - 1], channel_list[i]))
            to_rgb_layers.append(ModulatedConv(channel_list[i], 3, k_size=1, pad=0, upsample=False))

        self.progress = nn.ModuleList(progress_layers)
        self.to_rgb = nn.ModuleList(to_rgb_layers)

    def forward(self, dlatents_in):

        x = torch.ones(dlatents_in.size(0), 512, 4, 4).to(dev)
        # w1, w2 = torch.split(w, x.size(0), dim=0)
        #
        # w1 = w1.unsqueeze(1).repeat(1, 15, 1)
        # w2 = w2.unsqueeze(1).repeat(1, 15, 1)
        #
        # layer_idx = torch.from_numpy(np.arange(2*self.n_layer-1)[np.newaxis, :, np.newaxis]).to(dev)
        # if random.random() < self.style_mixing_prob:
        #     mixing_cutoff = random.randint(1, 2*self.n_layer-1)
        # else:
        #     mixing_cutoff = 2*self.n_layer-1
        #
        # dlatents_in = torch.where(layer_idx < mixing_cutoff, w1, w2)

        for i, (block, to_rgb) in enumerate(zip(self.progress, self.to_rgb)):
            if i == 0:
                out = block(x, dlatents_in[:, 2*i])
                rgb_img = to_rgb(out, dlatents_in[:, 2*i])
            else:
                upsampled_rgb_img = F.upsample(rgb_img, scale_factor=2)
                out = block(out, dlatents_in[:, 2*i-1], dlatents_in[:, 2*i])
                rgb_img = to_rgb(out, dlatents_in[:, 2*i])
                rgb_img = rgb_img + upsampled_rgb_img

        return rgb_img, dlatents_in


class Discriminator(nn.Module):
    def __init__(self, channel_list=[512, 512, 512, 512, 256, 128, 64, 32]):
        super(Discriminator, self).__init__()
        # reversed(channel_list)
        self.std_concat = StdConcat()

        progress_layers = []
        skip_layers = []
        blur_layers = []
        for i in range(len(channel_list) - 1, -1, -1):
            if i == 0:
                progress_layers.append(DownBlock(channel_list[i] + 1, channel_list[i], 3, 1, 4, 0))
            else:
                if channel_list[i-1] < 512:
                    progress_layers.append(DownBlock(channel_list[i], channel_list[i - 1],
                                                     3, 1, 3, 1, fused_downsample=True))
                else:
                    progress_layers.append(DownBlock(channel_list[i], channel_list[i - 1],
                                                     3, 1, 3, 1, fused_downsample=False))
            skip_layers.append(EqualizedConv2d(channel_list[i], channel_list[i - 1], k_size=1,
                                               stride=2, padding=0))
            blur_layers.append(Blur(channel_list[i-1], weight=[1, 2, 1]))

        self.progress = nn.ModuleList(progress_layers)
        self.skip = nn.ModuleList(skip_layers)
        self.blur = nn.ModuleList(blur_layers)

        self.n_layer = len(self.progress)
        self.from_rgb = EqualizedConv2d(3, channel_list[self.n_layer - 1], k_size=1, stride=1,
                                        padding=0)
        self.linear = EqualizedLinear(512, 1)

    def forward(self, x):
        out = self.from_rgb(x)
        for i in range(self.n_layer):
            if i == self.n_layer-1:
                out = self.std_concat(out)
                out = self.progress[i](out)
            else:
                downsample = self.skip[i](out)
                downsample = self.blur[i](downsample)
                out = self.progress[i](out)
                out = out + downsample

        out = out.squeeze(3).squeeze(2)
        out = self.linear(out)
        return out[:, 0]


if __name__ == "__main__":
    with torch.no_grad():
        z = torch.rand(8, 512).to(dev)
        const = torch.ones(4, 512, 4, 4).to(dev)

        m = MappingNetwork(z_dim=512).to(dev)
        g = Generator(channel_list=[512, 512, 512, 512, 256, 128, 64, 32]).to(dev)
        d = Discriminator(channel_list=[512, 512, 512, 512, 256, 128, 64, 32]).to(dev)
        alpha = 0.5

        w = m(z)
        w1, w2 = torch.split(w, 4, dim=0)
        print("Generator")
        out, latent = g(const, w1, w2)
        print(out.shape)
        print(latent.shape)

        print("Discriminator")
        out = d(out)
        print(out.shape)
