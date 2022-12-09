from torchvision import models
import torch.nn as nn
from torch.hub import load_state_dict_from_url

pre_train_url = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
pre_train_se = 'https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl'


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Baseline(nn.Module):
    def __init__(self, num_class):
        super(Baseline, self).__init__()
        self.num_class = num_class
        resnet = models.resnet50()

        resnet.layer4[0].conv2 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.layer4[0].downsample = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2048))

        resnet.load_state_dict(load_state_dict_from_url(pre_train_url), strict=True)

        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.local_conv = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        nn.init.kaiming_normal_(self.local_conv[0].weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.local_conv[1].weight.data, 1)
        nn.init.constant_(self.local_conv[1].bias.data, 0)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, std=0.001)
        nn.init.constant_(self.fc.bias, 0)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.backbone(x)
        out = self.avg_pool(out)
        out = self.dropout(out)
        out = self.local_conv(out)
        batch = out.shape[0]
        out = out.view(batch, -1)
        out = self.fc(out)

        return out

    def extract_feature(self, x):
        out = self.backbone(x)
        out = self.avg_pool(out)
        batch = out.shape[0]
        out = out.view(batch, -1)
        return out


class PartBasedCNN(nn.Module):
    def __init__(self, num_class, num_part):
        super(PartBasedCNN, self).__init__()
        self.num_class = num_class
        self.num_part = num_part
        resnet = models.resnet50()

        resnet.layer4[0].conv2 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.layer4[0].downsample = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2048))

        resnet.load_state_dict(load_state_dict_from_url(pre_train_url), strict=True)

        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        self.avg_pool = nn.AdaptiveAvgPool2d((self.num_part, 1))

        self.local_convs = nn.ModuleList()
        for _ in range(self.num_part):
            local_conv = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True))
            nn.init.kaiming_normal_(local_conv[0].weight.data, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(local_conv[1].weight.data, 1)
            nn.init.constant_(local_conv[1].bias.data, 0)
            self.local_convs.append(local_conv)

        self.fcs = nn.ModuleList()
        for _ in range(self.num_part):
            fc = nn.Linear(256, num_class)
            nn.init.normal_(fc.weight, std=0.001)
            nn.init.constant_(fc.bias, 0)
            self.fcs.append(fc)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.backbone(x)
        out = self.avg_pool(out)
        out = self.dropout(out)
        batch = out.shape[0]
        out_list = []
        for i in range(self.num_part):
            t = out[:, :, i, :]
            t = t.view(batch, -1, 1, 1)
            t = self.local_convs[i](t)
            t = t.view(batch, -1)
            t = self.fcs[i](t)
            out_list.append(t)

        return out_list

    def extract_feature(self, x):
        out = self.backbone(x)
        out = self.avg_pool(out)
        batch = out.shape[0]
        out = out.view(batch, -1)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        nn.init.kaiming_normal_(self.fc[0].weight.data, a=0, mode='fan_out')
        nn.init.kaiming_normal_(self.fc[2].weight.data, a=0, mode='fan_out')

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BottleneckSE(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample=None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.se = SELayer(channel=planes * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PartBasedAttentionCNN(nn.Module):
    def __init__(self, num_class, num_part):
        super(PartBasedAttentionCNN, self).__init__()
        self.num_class = num_class
        self.num_part = num_part
        resnet_se = models.ResNet(BottleneckSE, [3, 4, 6, 3])

        resnet_se.layer4[0].conv2 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        resnet_se.layer4[0].downsample = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2048))

        # load pre-trained weights
        resnet_se.load_state_dict(load_state_dict_from_url(pre_train_se), strict=True)
        resnet_se.load_state_dict(load_state_dict_from_url(pre_train_url), strict=False)

        modules = list(resnet_se.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.avg_pool = nn.AdaptiveAvgPool2d((self.num_part, 1))

        self.local_convs = nn.ModuleList()
        for _ in range(self.num_part):
            local_conv = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True))
            nn.init.kaiming_normal_(local_conv[0].weight.data, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(local_conv[1].weight.data, 1)
            nn.init.constant_(local_conv[1].bias.data, 0)
            self.local_convs.append(local_conv)

        self.fcs = nn.ModuleList()
        for _ in range(self.num_part):
            fc = nn.Linear(256, num_class)
            nn.init.normal_(fc.weight, std=0.001)
            nn.init.constant_(fc.bias, 0)
            self.fcs.append(fc)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.backbone(x)
        out = self.avg_pool(out)
        out = self.dropout(out)
        batch = out.shape[0]
        out_list = []
        for i in range(self.num_part):
            t = out[:, :, i, :]
            t = t.view(batch, -1, 1, 1)
            t = self.local_convs[i](t)
            t = t.view(batch, -1)
            t = self.fcs[i](t)
            out_list.append(t)

        return out_list

    def extract_feature(self, x):
        out = self.backbone(x)
        out = self.avg_pool(out)
        batch = out.shape[0]
        out = out.view(batch, -1)
        return out
