"""
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
"""

import torch
import torch.nn as nn
import math
from scipy.stats import shapiro

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_mnist(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet_mnist, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        # self.avgpool = nn.AvgPool2d(8, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.id = 'server'
        '''
            功能:定义增长集、最小距离集合、各类别阈值等参数，用以检测        
        '''
        self.D = []
        self.G = [[] for _ in range(num_classes)]
        self.DG = [[] for _ in range(num_classes)]
        self.attack = False
        self.Thre = [[torch.tensor(0)] for _ in range(num_classes)]
        self.delta = 0.9

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        t = self.conv1(x)
        t = self.bn1(t)
        t = self.relu(t)

        t = self.layer1(t)
        t = self.layer2(t)
        t = self.layer3(t)

        t = self.avgpool(t)
        t = t.view(t.size(0), -1)
        t = self.fc(t)

        if self.id == 'user':
            if self.training == False:
                if self.attack == True:
                    return 'Detected by PRADA'
                # 遍历batch中的每个样本
                batch_size = x.size(0)
                for i in range(batch_size) :
                    
                    # c ← F(x)
                    sample = x[i]
                    _, c = torch.max(t[i], dim=0)
                    
                    if not self.G[c]:
                        self.G[c].append(x[i])
                        self.DG[c].append(torch.tensor(0,device='cuda'))
                    else:
                        d = []
                        for y in self.G[c]:
                            d.append(torch.dist(x[i], y, p=2))
                        
                        dmin = min(d)
                        dmin = torch.tensor(dmin)
                        self.D.append(dmin)
                        
                        if dmin > self.Thre[c][0]:
                            self.G[c].append(x[i])
                            self.DG[c].append(dmin.to('cuda'))
                            DG = torch.stack(self.DG[c])
                            bar = torch.mean(DG)
                            std = torch.std(DG)
                            self.Thre[c][0] = torch.max(self.Thre[c][0], bar - std)
                    
                    #analyze distribution for D
                    if len(self.D) > 100:
                        # D'
                        D1 = []
                        
                        # 获取关于距离列表的均值和标准差
                        D_tensor = torch.stack(self.D)
                        bar = torch.mean(D_tensor)
                        std = torch.std(D_tensor)

                        # 获得D'
                        for z in self.D :
                            if z > bar - 3*std and z < bar + 3*std:
                                D1.append(z)
                        
                        D1 = torch.stack(D1).cpu().numpy()
                        W  = shapiro(D1)[0]
                        if W < self.delta :
                            self.attack = True
                            return 'Detected by PRADA'
                        else:
                            self.attack = False

                if self.attack == True:
                    return 'Detected by PRADA'
        
        return t
                
                

class ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=3):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        self.id = 'server'
        '''
            功能:定义增长集、最小距离集合、各类别阈值等参数，用以检测        
        '''
        self.D = []
        self.G = [[] for _ in range(num_classes)]
        self.DG = [[] for _ in range(num_classes)]
        self.attack = False
        self.Thre = [[torch.tensor(0)] for _ in range(num_classes)]
        self.delta = 0.9
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        t = self.conv1(x)
        t = self.bn1(t)
        t = self.relu(t)

        t = self.layer1(t)
        t = self.layer2(t)
        t = self.layer3(t)

        t = self.avgpool(t)
        t = t.view(x.size(0), -1)
        t = self.fc(t)

        if self.id == 'user':
            if self.training == False:
                if self.attack == True:
                    return 'Detected by PRADA'
                # 遍历batch中的每个样本
                batch_size = x.size(0)
                for i in range(batch_size) :
                    
                    # c ← F(x)
                    sample = x[i]
                    _, c = torch.max(t[i], dim=0)
                    
                    if not self.G[c]:
                        self.G[c].append(x[i])
                        self.DG[c].append(torch.tensor(0,device='cuda'))
                    else:
                        d = []
                        for y in self.G[c]:
                            d.append(torch.dist(x[i], y, p=2))
                        
                        dmin = min(d)
                        dmin = torch.tensor(dmin)
                        self.D.append(dmin)
                        
                        if dmin > self.Thre[c][0]:
                            self.G[c].append(x[i])
                            self.DG[c].append(dmin.to('cuda'))
                            DG = torch.stack(self.DG[c])
                            bar = torch.mean(DG)
                            std = torch.std(DG)
                            self.Thre[c][0] = torch.max(self.Thre[c][0], bar - std)
                    
                    #analyze distribution for D
                    if len(self.D) > 100:
                        # D'
                        D1 = []
                        
                        # 获取关于距离列表的均值和标准差
                        D_tensor = torch.stack(self.D)
                        bar = torch.mean(D_tensor)
                        std = torch.std(D_tensor)

                        # 获得D'
                        for z in self.D :
                            if z > bar - 3*std and z < bar + 3*std:
                                D1.append(z)
                        
                        D1 = torch.stack(D1).cpu().numpy()
                        W  = shapiro(D1)[0]
                        if W < self.delta :
                            self.attack = True
                            return 'Detected by PRADA'
                        else:
                            self.attack = False

                if self.attack == True:
                    return 'Detected by PRADA'
        
        return t


class PreAct_ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet8(num_classes=10, **kwargs):
    model = ResNet_Cifar(BasicBlock, [1, 1, 1], num_classes=num_classes, **kwargs)
    return model


def resnet20(num_classes=10, **kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], num_classes=num_classes, **kwargs)
    return model

def resnet20_brain(num_classes=3, **kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], num_classes=num_classes, in_channels=1, **kwargs)
    return model


def resnet20_mnist(num_classes=10, **kwargs):
    model = ResNet_mnist(BasicBlock, [3, 3, 3], num_classes=num_classes, **kwargs)
    return model


def resnet32(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model


if __name__ == "__main__":
    net = resnet20()
    y = net(torch.randn(1, 3, 64, 64))
    print(net)
    print(y.size())
