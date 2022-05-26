import math
import torch
import torch.nn as nn
import torch.nn.functional as functional


def gaussian2d(sigma, gx):
    gy = torch.exp(-gx * gx / (2 * sigma ** 2))
    return torch.ger(gy, gy) / (2 * math.pi * sigma ** 2)


class DoGLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DoGLayer, self).__init__()

        self.in_chans = in_channels
        self.out_chans = out_channels
        self.kernel = kernel_size

        self.s1 = torch.rand(self.out_chans, self.in_chans) * self.kernel / 6
        self.s2 = torch.rand(self.out_chans, self.in_chans) * self.kernel / 6
        self.s1 = nn.Parameter(self.s1)
        self.s2 = nn.Parameter(self.s2)

        self.gx = torch.linspace(-int(self.kernel / 2), int(self.kernel / 2), steps=self.kernel).cuda()
        # self.filters = torch.zeros(self.out_chans, self.in_chans, self.kernel, self.kernel).cuda()

    def forward(self, x):

        # if self.training:
        filters = torch.zeros(self.out_chans, self.in_chans, self.kernel, self.kernel).cuda()
        for i in range(self.out_chans):
            for j in range(self.in_chans):
                filters[i][j] = gaussian2d(self.s1[i, j], self.gx) - gaussian2d(self.s2[i, j], self.gx)
        # else:
        #    self.filters = self.filters.cuda()

        return functional.conv2d(x, filters, padding=int((self.kernel - 1) / 2))

    def getFilters(self):
        filters = torch.zeros(self.out_chans, self.in_chans, self.kernel, self.kernel)
        for i in range(self.out_chans):
            for j in range(self.in_chans):
                filters[i][j] = gaussian2d(self.s1[i, j], self.gx) - gaussian2d(self.s2[i, j], self.gx)
        return filters
