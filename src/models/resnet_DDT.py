"""
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import sys

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
    def __init__(self, block, layers, num_classes=10, T=torch.tensor(1), alpha = torch.tensor(0.6), Q_thre = torch.tensor(50), M_thre = torch.tensor(20)):
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
        
        '''
            功能：定义检测变量
            参数：
                self.T          : softmax温度
                self.T_decay    : softmax温度衰减因子
                self.alpha      : 恶性样本阈值
                self.Q          : 恶性样本查询计数器
                self.Q_thre     : 恶性用户判定阈值
                self.M          : 良性样本查询计数
                self.M_thre     : 温度补偿阈值
                self.noise_frac : 噪声大小比例
                self.tep_frac   : 增加的温度比例(被加如黑名单后,若查询一次良性样本)
            输出：
                在forward方法中,若处于eval()模式,则进行动态检测加扰
        '''
        
        # 温度
        self.T = T
        self.T = self.T.to(torch.float)
        
        # 温度下降因子
        self.T_decay = torch.tensor(0.9)
        
        # 置信度阈值
        self.alpha = alpha
        
        # 恶性样本（黑名单）查询数
        self.Q = torch.tensor(0)
        
        # 恶性（黑名单）用户判定阈值
        self.Q_thre = Q_thre

        self.M = torch.tensor(0)
        self.M_thre = M_thre

        # 噪声大小比例
        self.noise_frac = torch.tensor(1e-3)
        
        # 增加的温度比例
        self.tep_frac = torch.tensor(1e-4)

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
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        
        if self.id == "server":
        # 如果是服务商在使用模型进行训练或者测试，则进行正常返回
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        
        # 测试阶段根据logits进行检测，调整温度
        elif self.id == "user":
            if self.training :
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
            else:
                x = x.view(x.size(0), -1)
                logits_true = self.fc(x)

                # 对每个样本的logits_true_max进行归一化，获得每一个样本的最大值进行计数
                logits_true = F.softmax(logits_true, dim=1)
                max_values, _ = torch.max(logits_true,dim=1)
                max_values = max_values.to(self.alpha.device)
                
                self.Q += torch.sum(max_values < self.alpha)

                # print("当前温度：",self.T.item(),"\t","正确的最大logits项:",logits_true_max,"错误的最大logits项:",torch.max(F.softmax(logits_true/self.T)))
                # 恶性用户判定，向恶性用户返回衰减温度下的logits
                if self.Q >= self.Q_thre:
                    # 一个batch中所有的最大置信项
                    for values in max_values :
                        if values < self.alpha:
                            self.T *= self.T_decay
                            noise_mean = self.T * self.noise_frac
                            noise_std = noise_mean * self.noise_frac
                            noise = noise_std * torch.randn([]) + noise_mean 
                            self.T += noise
                            # 良性样本数清0
                            self.M = torch.tensor(0)
                        else :
                            self.M += torch.tensor(1)
                            if self.M >= self.M_thre:
                                # 温度补偿
                                if self.T < torch.tensor(1.0):
                                    self.T /= self.T_decay 
                                if self.T >= torch.tensor(1.0):
                                    self.T = torch.tensor(1.0)
                    logits_false = self.fc(x/self.T)
                    return logits_false
                # 若是良性用户，向良性用户返回正常温度下的logits(未经过softmax，故不能使用logits_ture)
                else:
                    logits_true = self.fc(x)
                    return logits_true
        else:
            sys.exit('Unspecified or incorrectly specified identity')
            
        


class ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=3, T = torch.tensor(1.0), alpha = torch.tensor(0.6), Q_thre = torch.tensor(50), M_thre = torch.tensor(20)):
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
        self.T = T
        self.id = None

        '''
            功能：定义检测变量
            参数：
                self.T          : softmax温度
                self.T_decay    : softmax温度衰减因子
                self.alpha      : 恶性样本阈值
                self.Q          : 恶性样本查询计数器
                self.Q_thre     : 恶性用户判定阈值
                self.M          : 良性样本查询计数
                self.M_thre     : 温度补偿阈值
                self.noise_frac : 噪声大小比例
                self.tep_frac   : 增加的温度比例(被加如黑名单后,若查询一次良性样本)
            输出：
                在forward方法中,若处于eval()模式,则进行动态检测加扰
        '''
        
        # 温度
        self.T = T
        self.T = self.T.to(torch.float)
        
        # 温度下降因子
        self.T_decay = torch.tensor(0.9)
        
        # 置信度阈值
        self.alpha = alpha
        
         # 恶性样本（黑名单）查询数
        self.Q = torch.tensor(0)
        
        # 恶性（黑名单）用户判定阈值
        self.Q_thre = Q_thre

        # 良性样本查询数
        self.M = torch.tensor(0)
        # 温度补偿阈值
        self.M_thre = M_thre
        
        # 噪声大小比例
        self.noise_frac = torch.tensor(1e-3)
        
        # 增加的温度比例
        self.tep_frac = torch.tensor(1e-4)
        
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        
        if self.id == "server":
        # 如果是服务商在使用模型进行训练或者测试，则进行正常返回
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        
        # 测试阶段根据logits进行检测，调整温度
        elif self.id == "user":
            if self.training :
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
            else:
                x = x.view(x.size(0), -1)
                logits_true = self.fc(x)

                # 对每个样本的logits_true_max进行归一化，获得每一个样本的最大值进行计数
                logits_true = F.softmax(logits_true, dim=1)
                max_values, _ = torch.max(logits_true,dim=1)
                max_values = max_values.to(self.alpha.device)
                
                self.Q += torch.sum(max_values < self.alpha)

                # print("当前温度：",self.T.item(),"\t","正确的最大logits项:",logits_true_max,"错误的最大logits项:",torch.max(F.softmax(logits_true/self.T)))
                # 恶性用户判定，向恶性用户返回衰减温度下的logits
                if self.Q >= self.Q_thre:
                    for values in max_values :
                        if values < self.alpha:
                            self.T *= self.T_decay
                            noise_mean = self.T * self.noise_frac
                            noise_std = noise_mean * self.noise_frac
                            noise = noise_std * torch.randn([]) + noise_mean 
                            self.T += noise
                        else :
                            self.M += torch.tensor(1)
                            if self.M >= self.M_thre:
                                # 温度补偿
                                if self.T < torch.tensor(1.0):
                                    self.T /= self.T_decay 
                                if self.T >= torch.tensor(1.0):
                                    self.T = torch.tensor(1.0)
                    logits_false = self.fc(x/self.T)
                    return logits_false
                
                # 若是良性用户，向良性用户返回正常温度下的logits(未经过softmax，故不能使用logits_ture)
                else:
                    logits_true = self.fc(x)
                    return logits_true
        else:
            sys.exit('Unspecified or incorrectly specified identity')

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
