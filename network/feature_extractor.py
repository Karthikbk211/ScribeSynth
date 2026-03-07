import torch
import torch.nn as nn


class LearnableFrequencyFilter(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.filter = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return self.filter(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, first_dilation=False):
        super().__init__()
        if first_dilation:
            self.residual_function = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=stride, padding=1, bias=False, dilation=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * BasicBlock.expansion,
                          kernel_size=3, padding=2, bias=False, dilation=2),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        else:
            padding = 1 if dilation == 1 else 2
            self.residual_function = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                          padding=padding, bias=False, dilation=dilation),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * BasicBlock.expansion,
                          kernel_size=3, padding=padding, bias=False, dilation=dilation),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, num_block):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(
            block, 128, num_block[1], 1, dilation=1)
        self.conv4_x = self._make_layer(
            block, 256, num_block[2], 1, dilation=1)
        self.conv5_x = self._make_layer(
            block, 512, num_block[3], 1, dilation=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, num_blocks, stride, dilation=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        if dilation == 1:
            for stride in strides:
                layers.append(block(self.in_channels, out_channels, stride, 1))
                self.in_channels = out_channels * block.expansion
        else:
            layers.append(
                block(self.in_channels, out_channels, stride, 2, True))
            self.in_channels = out_channels * block.expansion
            for stride in strides[1:]:
                layers.append(block(self.in_channels, out_channels, stride, 2))
                self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        return output


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
