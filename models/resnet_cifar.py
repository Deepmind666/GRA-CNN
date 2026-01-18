import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, cfg=None):
        super(BasicBlock, self).__init__()
        # cfg[0] is the output channels of conv1 (and input of conv2)
        # planes is the expected output of the block
        
        if cfg is None:
            mid_planes = planes
        else:
            mid_planes = cfg[0]
            
        self.conv1 = conv3x3(inplanes, mid_planes, stride)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, depth, num_classes=10, cfg=None):
        super(ResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # If cfg is None, generate default cfg
        # cfg contains the number of filters for the first conv of each block.
        # There are 3 stages, each with n blocks. Total 3*n blocks.
        if cfg is None:
            cfg = [[16]*n, [32]*n, [64]*n]
            # Flatten
            cfg = [item for sublist in cfg for item in sublist]
            
        self.cfg = cfg
        
        self.layer1 = self._make_layer(BasicBlock, 16, n, cfg=cfg[0:n])
        self.layer2 = self._make_layer(BasicBlock, 32, n, stride=2, cfg=cfg[n:2*n])
        self.layer3 = self._make_layer(BasicBlock, 64, n, stride=2, cfg=cfg[2*n:3*n])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, cfg=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # First block
        layers.append(block(self.inplanes, planes, stride, downsample, cfg=[cfg[0]] if cfg else None))
        self.inplanes = planes * block.expansion
        # Subsequent blocks
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg=[cfg[i]] if cfg else None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet20(cfg=None, **kwargs):
    return ResNet(depth=20, cfg=cfg, **kwargs)

def resnet32(cfg=None, **kwargs):
    return ResNet(depth=32, cfg=cfg, **kwargs)

def resnet44(cfg=None, **kwargs):
    return ResNet(depth=44, cfg=cfg, **kwargs)

def resnet56(cfg=None, **kwargs):
    return ResNet(depth=56, cfg=cfg, **kwargs)

def resnet110(cfg=None, **kwargs):
    return ResNet(depth=110, cfg=cfg, **kwargs)
