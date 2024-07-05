#coding:utf8

# Copyright 2023 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com
#
# or create issues
# =============================================================================
import torch
import torch.nn as nn

# 卷积模块
class BasicConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=(1,1),padding=(0,0)):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

# 额外的损失分支
class SideBranch(nn.Module):
    def __init__(self, in_channels,num_classes):
        super(SideBranch, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1x1 = BasicConv(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.fc_1 = nn.Linear(in_features=2048, out_features=1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc_2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1x1(x)
        x = torch.flatten(x,1)
        x = self.fc_1(x)
        x = self.relu(x)
        x = torch.dropout(x, 0.7, train=True)
        x = self.fc_2(x)
        return x

# Inception模块
class InceptionBlock(nn.Module):
    def __init__(self,in_channels,ch1x1, ch3x3reduce,ch3x3,ch5x5reduce,ch5x5,chpool):
        super(InceptionBlock, self).__init__()
        self.branch_1 = BasicConv(in_channels=in_channels, out_channels=ch1x1,kernel_size=1)
        self.branch_2 = nn.Sequential(
            BasicConv(in_channels=in_channels, out_channels=ch3x3reduce, kernel_size=1),
            BasicConv(in_channels=ch3x3reduce, out_channels=ch3x3,kernel_size=3, padding=1)
        )
        self.branch_3 = nn.Sequential(
            BasicConv(in_channels=in_channels, out_channels=ch5x5reduce,kernel_size=1),
            BasicConv(in_channels=ch5x5reduce,out_channels=ch5x5,kernel_size=5, padding=2)
        )
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1,stride=1, ceil_mode=True),
            BasicConv(in_channels=in_channels,out_channels=chpool,kernel_size=1)
        )

    def forward(self, x):
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        x_4 = self.branch_4(x)
        x = torch.cat([x_1, x_2, x_3, x_4], dim=1)
        return x

# GoogLeNet/Inception模型
class Inception_V1(nn.Module):
    def __init__(self, num_classes):
        super(Inception_V1, self).__init__()
        self.BasicConv_1 = BasicConv(in_channels=3, out_channels=64, kernel_size=7,stride=2, padding=3)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)   # 把不足square_size的边保留下来，单独计算
        self.lrn_1 = nn.LocalResponseNorm(2)

        self.conv_1x1 = BasicConv(in_channels=64, out_channels=64, kernel_size=1)
        self.conv_3x3 = BasicConv(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.lrn_2 = nn.LocalResponseNorm(2)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        #   in_channels,ch1x1, ch3x3reduce,ch3x3,ch5x5reduce,ch5x5,chpool
        self.InceptionBlock_3a = InceptionBlock(192,64,96,128,16,32,32)
        self.InceptionBlock_3b = InceptionBlock(256,128,128,192,32,96,64)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.InceptionBlock_4a = InceptionBlock(480,192,96,208,16,48,64)

        self.SideBranch_1 = SideBranch(512, num_classes)

        self.InceptionBlock_4b = InceptionBlock(512,160,112,224,24,64,64)
        self.InceptionBlock_4c = InceptionBlock(512,128,128,256,24,64,64)
        self.InceptionBlock_4d = InceptionBlock(512,112,144,288,32,64,64)

        self.SideBranch_2 = SideBranch(528, num_classes)

        self.InceptionBlock_4e = InceptionBlock(528,256,160,320,32,128,128)

        self.max_pool_4 = nn.MaxPool2d(kernel_size=3,stride=2, ceil_mode=True)

        self.InceptionBlock_5a = InceptionBlock(832,256,160,320,32,128,128)
        self.InceptionBlock_5b = InceptionBlock(832,384,192,384,48,128,128)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=1024 ,out_features=num_classes)

    def forward(self, x):
        x = self.BasicConv_1(x)
        x = self.max_pool_1(x)
        x = self.lrn_1(x)

        x = self.conv_1x1(x)
        x = self.conv_3x3(x)
        x = self.lrn_1(x)
        x = self.max_pool_2(x)

        x = self.InceptionBlock_3a(x)
        x = self.InceptionBlock_3b(x)
        x = self.max_pool_3(x)

        x = self.InceptionBlock_4a(x)

        x_1 = self.SideBranch_1(x)

        x = self.InceptionBlock_4b(x)
        x = self.InceptionBlock_4c(x)
        x = self.InceptionBlock_4d(x)

        x_2 = self.SideBranch_2(x)

        x = self.InceptionBlock_4e(x)

        x = self.max_pool_4(x)

        x = self.InceptionBlock_5a(x)
        x = self.InceptionBlock_5b(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = torch.dropout(x, 0.4,train=True)
        x = self.fc(x)

        x_1 = torch.softmax(x_1, dim=1)
        x_2 = torch.softmax(x_2, dim=1)
        x_3 = torch.softmax(x, dim=1)

        # output = x_3 + (x_1 + x_2) * 0.3
        return x_3,x_2,x_1


if __name__ == '__main__':
    # 创建模型，给定输入，前向传播，存储模型
    input = torch.randn([1, 3, 224, 224])
    model = Inception_V1(num_classes=1000)
    torch.save(model, 'googlenet.pth')

    x_3,x_2,x_1 = model(input)

    # 观察输出，只需要观察shape是我们想要的即可
    print(x_1.shape)
    print(x_2.shape)
    print(x_3.shape)

    torch.onnx.export(model, input, 'googlenet.onnx')


