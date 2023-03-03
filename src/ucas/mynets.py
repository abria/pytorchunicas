import os

import torch
import torch.nn as nn
import torchvision.models as models

import ucas.DoGLayer as DoG
import ucas.Sinc_conv2D as Sinc


def flatten(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return x.view(-1, num_features)


def init_weights(m, verbose=False):
    if isinstance(m, nn.Conv2d):
        if verbose:
            print(f'...initializing {type(m).__name__} with {nn.init.xavier_normal_.__name__} and bias = 0')
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        if verbose:
            print(f'...initializing {type(m).__name__} with {nn.init.xavier_normal_.__name__} and bias = 0')
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    else:
        if verbose:
            print(f'...initializing {type(m).__name__} with default initialization method')


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, x):
        return self.resnet(x)


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.resnet = models.resnet50(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

    def forward(self, x):
        return self.resnet(x)


class DoGResNet18(nn.Module):
    def __init__(self):
        super(DoGResNet18, self).__init__()

        self.DoG = DoG.DoGLayer(1, 32, 47)
        self.ln = nn.LayerNorm([32, 48, 48])
        self.act = nn.ReLU()

        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, x):
        return self.resnet(self.act(self.ln(self.DoG(x))))


class DoGResNet50(nn.Module):
    def __init__(self):
        super(DoGResNet50, self).__init__()

        self.DoG = DoG.DoGLayer(1, 32, 47)
        self.ln = nn.LayerNorm([32, 48, 48])
        self.act = nn.ReLU()

        self.resnet = models.resnet50(weights=None)
        self.resnet.conv1 = nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

    def forward(self, x):
        return self.resnet(self.act(self.ln(self.DoG(x))))


class MCNet(nn.Module):
    def __init__(self):
        super(MCNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv5 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv7 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv8 = nn.Conv2d(32, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(32 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.apply(init_weights)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.pool(x)
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        x = self.pool(x)
        x = self.act(self.conv7(x))
        x = self.act(self.conv8(x))
        x = self.pool(x)
        x = flatten(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ConvMCNet(MCNet):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 32, 3, 1, 1)
        self.ln = nn.LayerNorm([32, 48, 48])
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.apply(init_weights)

    def forward(self, x):
        x = self.act(self.ln(self.conv0(x)))
        return super().forward(x)


class DeepMCNet(MCNet):
    def __init__(self):
        super().__init__()
        self.conv00 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv01 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.fc1 = nn.Linear(32, 256)
        self.apply(init_weights)

    def forward(self, x):
        x = self.act(self.conv00(x))
        x = self.act(self.conv01(x))
        x = self.pool(x)
        return super().forward(x)


class UniformDoGMCNet(ConvMCNet):
    def __init__(self):
        super().__init__()
        lim = 47 / 6
        b = lim / 31
        s1 = torch.linspace(b, lim, 32).unsqueeze(1)
        s2 = torch.linspace(0, lim - b, 32).unsqueeze(1)
        s2[0][0] = 0.1
        self.conv0 = DoG.NotLearnableDoGLayer(1, 32, 47, s2, s1)


class RandomDoGMCNet(ConvMCNet):
    def __init__(self):
        super().__init__()
        s1 = torch.rand(32, 1) * 47 / 6
        s2 = torch.rand(32, 1) * 47 / 6
        self.conv0 = DoG.NotLearnableDoGLayer(1, 32, 47, s1, s2)


def get_sigma(path, out_channel=32, in_channel=1):
    with open(path, "r") as f:
        a = f.readlines()
    b = [[float(x[23:27]), float(x[30:34])] for x in a]

    c = torch.tensor(b)[0:32]

    s1 = c[:, 0].unsqueeze(1)
    s2 = c[:, 1].unsqueeze(1)

    return s1, s2


class OptimizedDoGMCNet(ConvMCNet):
    def __init__(self):
        super().__init__()
        s1, s2 = get_sigma(os.path.dirname(os.path.abspath(__file__)) + "/DoG_params.txt")
        self.conv0 = DoG.NotLearnableDoGLayer(1, 32, 47, s1, s2)


class DoGMCNet(ConvMCNet):
    def __init__(self):
        super().__init__()
        self.conv0 = DoG.DoGLayer(1, 32, 47)

    def getFilters(self):
        return self.conv0.getFilters()


class SincMCNet(ConvMCNet):
    def __init__(self):
        super().__init__()
        self.conv0 = Sinc.Sinc_conv2D(1, 32, 47)
