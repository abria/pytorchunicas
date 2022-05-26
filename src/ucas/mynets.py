import torch.nn as nn
import torch.nn.functional as F
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


class VGGNetL4(nn.Module):
    def __init__(self):
        super(VGGNetL4, self).__init__()
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


class ControlConvLnL4(VGGNetL4):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 32, 3, 1, 1)
        self.ln = nn.LayerNorm([32, 48, 48])
        self.apply(init_weights)

    def forward(self, x):
        x = self.act(self.ln(self.conv0(x)))
        return super().forward(x)


class DoGNetL4(ControlConvLnL4):
    def __init__(self):
        super().__init__()
        self.conv0 = DoG.DoGLayer(1, 32, 47)

    def getFilters(self):
        return self.conv0.getFilters()


class SincNetL4(ControlConvLnL4):
    def __init__(self):
        super().__init__()
        self.conv0 = Sinc.Sinc_conv2D(1, 32, 47)
