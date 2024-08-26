import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self,inc,outc,stride=1):
        super(BasicBlock, self).__init__()
        self.stride = stride
        if self.stride != 1:
            self.conv1 = nn.Conv2d(inc, outc, 3, stride, 1)
            self.conv1x1 = nn.Conv2d(inc, outc, 1)
            self.bn1x1 = nn.BatchNorm2d(outc)
        else:
            self.conv1 = nn.Conv2d(inc,outc,3,1,1)

        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc,outc,3,1,1)
        self.bn2 = nn.BatchNorm2d(outc)


    def forward(self, x): #前向传播函数
        identity = x # 保存输入
        if self.stride != 1:
            #print('before pool.shape', identity.shape)
            identity = self.conv1x1(identity)
            identity = self.bn1x1(identity)
            identity = F.max_pool2d(identity,3,2,1)
            #print('after pool.shape', identity.shape)

        # 残差分支
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        print('x.shape', x.shape)
        print('identity.shape', identity.shape)
        return x + identity

class BottleBlock(nn.Module):
    def __init__(self,cin,cout,stride=1):
        self.expand = 4
        super(BottleBlock, self).__init__()
        self.stride = stride
        self.flag = False
        if self.stride != 1:
            self.conv1_1 = nn.Conv2d(cin,cout,1,stride)
        else:
            self.conv1_1 = nn.Conv2d(cin,cout,1,1)

        self.bn1_1 = nn.BatchNorm2d(cout)
        self.conv1_2 = nn.Conv2d(cout,cout,3,1,1)
        self.bn1_2 = nn.BatchNorm2d(cout)
        self.conv1_3 = nn.Conv2d(cout,cout*self.expand,1,1)
        self.bn1_3 = nn.BatchNorm2d(cout*self.expand)

        if cin == cout * self.expand:
            self.flag = True
        else:
            self.conv2_1 = nn.Conv2d(cin, cout * self.expand, 1)
            self.bn2_1 = nn.BatchNorm2d(cout * self.expand)

    def forward(self, x): #前向传播函数
        identity = x # 保存输入
        if self.flag == False:
            identity = self.conv2_1(identity)
            identity = self.bn2_1(identity)
        if self.stride != 1:
            #print('before pool.shape', identity.shape)
            identity = F.max_pool2d(identity,3,2,1)
            #print('after pool.shape', identity.shape)
        # 残差分支
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = F.relu(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = F.relu(x)

        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = F.relu(x)
        print('x.shape', x.shape)
        print('identity.shape', identity.shape)
        return x + identity

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__() #继承Net类，并进行初始化
        # 第一个卷积模块部分
        self.conv1 = nn.Conv2d(3,64,7,2,3) # 224*224*3 -> 112*112*64
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3,2,1) # 112*112*64 -> 56*56*64

        self.conv2_1 = BottleBlock(64, 64)
        self.conv2_2 = BottleBlock(256, 64)
        self.conv2_3 = BottleBlock(256, 64)

        self.conv3_1 = BottleBlock(256, 128, stride=2)
        self.conv3_2 = BottleBlock(512, 128)
        self.conv3_3 = BottleBlock(512, 128)
        self.conv3_4 = BottleBlock(512, 128)

        self.conv4_1 = BottleBlock(512, 256, stride=2)
        self.conv4_2 = BottleBlock(1024, 256)
        self.conv4_3 = BottleBlock(1024, 256)
        self.conv4_4 = BottleBlock(1024, 256)
        self.conv4_5 = BottleBlock(1024, 256)
        self.conv4_6 = BottleBlock(1024, 256)

        self.conv5_1 = BottleBlock(1024, 512, stride=2)
        self.conv5_2 = BottleBlock(2048, 512)
        self.conv5_3 = BottleBlock(2048, 512)

        self.avgpool = nn.AvgPool2d(7)
        self.cls = nn.Linear(2048,1000)

    def forward(self, x): #前向传播函数
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        print('conv5_3.shape', x.shape)
        x = self.avgpool(x)
        x = x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
        x = self.cls(x)
        return x

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__() #继承Net类，并进行初始化
        # 第一个卷积模块部分
        self.conv1 = nn.Conv2d(3,64,7,2,3) # 224*224*3 -> 112*112*64
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3,2,1) # 112*112*64 -> 56*56*64

        self.conv2_1 = BasicBlock(64,64)
        self.conv2_2 = BasicBlock(64,64)

        self.conv3_1 = BasicBlock(64, 128,stride=2)
        self.conv3_2 = BasicBlock(128, 128)

        self.conv4_1 = BasicBlock(128, 256,stride=2)
        self.conv4_2 = BasicBlock(256, 256)

        self.conv5_1 = BasicBlock(256, 512,stride=2)
        self.conv5_2 = BasicBlock(512, 512) # 7*7*512

        self.avgpool = nn.AvgPool2d(7)
        self.cls = nn.Linear(512,1000)

    def forward(self, x): #前向传播函数
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        print('before pool1.shape', x.shape)
        x = self.pool1(x)
        print('before pool1.shape', x.shape)
        x = self.conv2_1(x)
        print('conv2_1.shape',x.shape)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        print('conv3_1.shape',x.shape)
        x = self.conv3_2(x)
        print('conv3_2.shape',x.shape)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.avgpool(x)
        x = x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
        x = self.cls(x)
        return x

if __name__ == '__main__':
    x = torch.randn(1,3,224,224)
    model = ResNet18()
    y = model(x)
    print(y.shape)
    torch.onnx.export(model,x,"resnet18.onnx")

