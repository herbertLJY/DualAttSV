from __future__ import absolute_import
import torch
from torch import nn
from torch.nn import init

__all__ = ['resnet18', 'resnet34', 'resnet50']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=(stride, stride),
                     padding=1, bias=False)


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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=(stride, stride),
                               padding=1, bias=False)
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


class Input_preprocess(nn.Module):
    def __init__(self, freq_len=64):
        super(Input_preprocess, self).__init__()

        # mel stream 1
        self.m_1_conv1 = nn.Conv2d(1, 16, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.m_1_bn1 = nn.BatchNorm2d(16)

        # mel stream 2
        self.m_2_conv1 = nn.Conv2d(freq_len, freq_len, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=True)
        self.m_2_bn1 = nn.BatchNorm2d(1)

        self.mel_conv = nn.Conv2d(17, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.all_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.reset_params()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, mel_input):

        # mel stream1
        m1_out = self.m_1_conv1(mel_input)
        m1_out = self.m_1_bn1(m1_out)

        # mel stream2
        m2_out = mel_input.permute([0, 3, 2, 1]).contiguous()
        m2_out = self.m_2_conv1(m2_out)
        m2_out = m2_out.permute([0, 3, 2, 1]).contiguous()
        m2_out = self.m_2_bn1(m2_out)

        mel_out = torch.cat([m1_out, m2_out], dim=1)
        mel_out = self.relu(mel_out)
        mel_out = self.mel_conv(mel_out)

        out = self.all_bn(mel_out)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_features=256, dropout=0., num_classes=1000, pre_bn=False, **unused):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        if pre_bn:
            self.pre_bn = nn.BatchNorm2d(1)
            init.constant_(self.pre_bn.weight, 1)
            init.constant_(self.pre_bn.bias, 0)
        else:
            self.pre_bn = None
        self.input_preprocess = Input_preprocess()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.out_planes = 512 * block.expansion

        # Whether only classification
        self.has_embedding = num_features

        self.feat = nn.Linear(self.out_planes, self.num_features, bias=False)
        init.kaiming_normal_(self.feat.weight, mode='fan_out')
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
        init.kaiming_normal_(self.classifier.weight, mode='fan_out')

        self.dropout = dropout
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

        print('num_feature', self.num_features)
        self.reset_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(stride, stride), bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, mel):
        # mel input size [N, 1, H, W]

        if self.pre_bn is not None:
            mel = self.pre_bn(mel)
        x = self.input_preprocess(mel)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.mean(x, dim=-1, keepdim=False)

        batch_sz, seq_len = x.size(0), x.size(2)
        x = x.permute([0, 2, 1]).contiguous()

        x = x.view([-1, self.out_planes])

        raw = x.view(batch_sz, seq_len, self.out_planes)
        if self.dropout > 0:
            x = self.drop(x)

        x_feat = self.feat(x)
        x_feat = self.feat_bn(x_feat)
        x_feat = x_feat.view(batch_sz, seq_len, -1)
        x_cls = torch.mean(x_feat, dim=1, keepdim=False)

        x_cls = self.classifier(x_cls)

        outputs = [x_feat, x_cls, raw]
        return outputs

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
