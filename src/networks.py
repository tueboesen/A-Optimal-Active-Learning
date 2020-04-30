import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    '''
    This is a relatively simple resnet, that by default zeros the mean of the output, meaning that the label probabilities returned fullfil the constraint ye=0
    '''
    def __init__(self, block, layers, num_classes=10, zero_center=False):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.zero_center = zero_center
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        if self.zero_center:
            out = out - torch.mean(out,dim=1)[:,None]
        return out

class ResNet_simple(nn.Module):
    def __init__(self, input_dim, num_classes=10, h=0.1):
        super(ResNet_simple, self).__init__()
        self.h = h
        self.layer1 = nn.Sequential(nn.Linear(input_dim, 56),nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(56,128),nn.Tanh())
        self.fc = nn.Linear(128,num_classes)
    def forward(self, x):
        out = x + self.h * self.layer1(x)
        out = out + self.h * self.layer2(out)
        out = self.fc(out)
        return out



class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, num_classes,bias=False)

    def forward(self, x):
        x = x.view(-1, self.fc.in_features)
        x = self.fc(x)
        return x
