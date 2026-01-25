import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, cfg=None):
        super(BasicBlock, self).__init__()
        mid_planes = cfg[0] if cfg else planes
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

class ResNet18Tiny(nn.Module):
    def __init__(self, num_classes=200, cfg=None):
        super(ResNet18Tiny, self).__init__()
        self.inplanes = 64
        
        if cfg is None:
            # Default configuration for ResNet18: [64, 64] * 2 + [128, 128] * 2 + ...
            # But wait, BasicBlock has 2 convs. Only conv1 is pruned usually?
            # Or we can prune both? Standard is usually pruning conv1 output (which is conv2 input).
            # In the ResNet20 code, cfg controlled the output of conv1 in each block.
            # Let's stick to that.
            # ResNet18 has 4 layers (groups), each with 2 blocks. Total 8 blocks.
            # layer1: 2 blocks (64)
            # layer2: 2 blocks (128)
            # layer3: 2 blocks (256)
            # layer4: 2 blocks (512)
            cfg = [[64]*2, [128]*2, [256]*2, [512]*2]
            cfg = [item for sublist in cfg for item in sublist]
            
        self.cfg = cfg
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, cfg=cfg[0:2])
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, cfg=cfg[2:4])
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, cfg=cfg[4:6])
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, cfg=cfg[6:8])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

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
        layers.append(block(self.inplanes, planes, stride, downsample, cfg=[cfg[0]] if cfg else None))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg=[cfg[i]] if cfg else None))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet18_tiny(num_classes=200, cfg=None):
    return ResNet18Tiny(num_classes=num_classes, cfg=cfg)
