import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sinc_conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Sinc_conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.f1 = torch.rand(self.out_channels)
        self.band = (torch.ones(self.out_channels) - self.f1) * torch.rand(self.out_channels)
        self.f1 = self.f1.cuda(non_blocking=True)
        self.band = self.band.cuda(non_blocking=True)
        self.f2 = self.f1 + self.band

        n = torch.linspace(0, self.kernel_size, steps=self.kernel_size)
        hamming = 0.54 * torch.ones(self.kernel_size) - 0.46 * torch.cos(2 * math.pi * n / self.kernel_size)
        self.finestra = torch.ger(hamming, hamming)

        self.x = torch.linspace(1, int(self.kernel_size / 2), steps=int(self.kernel_size / 2))

        # Updatable parameters of the layer
        self.f1 = nn.Parameter(self.f1)
        self.band = nn.Parameter(self.band)

    def sinc2d(self, band, t_right):
        y_right = torch.sin(math.pi * band * t_right) / (math.pi * band * t_right)
        y_left = torch.flip(y_right, (0,))
        one = torch.ones(1).cuda(non_blocking=True)
        y_right = y_right.cuda(non_blocking=True)
        y_left = y_left.cuda(non_blocking=True)

        y = torch.cat([y_left, one, y_right])

        a = band * band * torch.ger(y, y)

        return a

    def forward(self, input):
        self.f2 = self.f1 + torch.abs(self.band)

        filters = torch.zeros(self.out_channels, self.kernel_size, self.kernel_size)

        filters = filters.cuda(non_blocking=True)
        self.finestra = self.finestra.cuda(non_blocking=True)
        self.x = self.x.cuda(non_blocking=True)

        for i in range(self.out_channels):
            sinc2d_f1 = self.sinc2d(self.f1[i], self.x)
            sinc2d_f2 = self.sinc2d(self.f2[i], self.x)
            filters[i] = (sinc2d_f2 - sinc2d_f1) * self.finestra

        out = F.conv2d(input,
                       filters.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size),
                       padding=int((self.kernel_size - 1) / 2))

        return out
