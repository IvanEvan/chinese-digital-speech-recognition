#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Evan
from utils.BasicModule import BasicModule
from torch import nn

# use a dict to save the model

model_arch = {
    'VGG11_BN': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13_BN': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}


class VGG(BasicModule):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # 3 fully connected layer
            # 最后一层隐含层 做flatten之后的参数个数
            nn.Linear(12288, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
        self.__init__weight()

    def forward(self, x):
        x = self.features(x)
        # flatten eg: from (N,C,W,J)->(N,C*W*H)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def __init__weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # param1 mean param2 std
                # fully connected 层 参数
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # Relu常用的weight initialization 就是 He initialization
                # W = tf.Variable(np.random.randn(node_in, node_out)) / np.sqrt(node_in / 2)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def build_model(model_arch):
    layers = []

    input_channel = 1

    for l in model_arch:
        if l == 'M':
            # 卷积核大小与步长都为2
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # nn.Conv2d(输入)
            # VGG 用 3x3的小卷积核，可以有效减少参数数量
            # nn.Conv2d(input,output,kernel_size,padding)
            # input_channel 是上一层卷积的卷积核个数，本层卷积的卷积核深度
            # output_channel 本层卷积的卷积核个数，下层卷积的feature map
            conv2d = nn.Conv2d(input_channel, l, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(l), nn.ReLU(inplace=True)]

            input_channel = l

    return nn.Sequential(*layers)


def vgg13_bn(**kwargs):
    return VGG(build_model(model_arch['VGG13_BN']), **kwargs)
