import torch.nn as nn


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.relu(self.linear1(y))
        y = self.sigmoid(self.linear2(y)).view(b, c, 1, 1)
        return x * y


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.se = SqueezeExcitationBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x


class Model(nn.Module):
    def __init__(self, hidden_size, num_block, dropout):
        super(Model, self).__init__()
        self.convs = self._create_convs(hidden_size, num_block)
        self.flatten = nn.Flatten()
        self.bn = nn.BatchNorm1d(hidden_size * 8 * 8)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 8 * 8, 64)
        self.relu = nn.ReLU()

    def _create_convs(self, hidden_size, num_block):
        convs = nn.ModuleList()
        for i in range(num_block):
            in_channels = 2 if i == 0 else hidden_size
            convs.append(ConvolutionBlock(in_channels, hidden_size, 3, 1, 1))
        return convs

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.flatten(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        return x
