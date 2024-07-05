import torch 
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, num_channels, strides=1):
        super().__init__()
        self.c1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides)
        self.c2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if not strides == 1:     # use 1x1conv to make H&W the same
            self.c3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
        else:
            self.c3 = None  # Looks like it's not rigorous here.
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        
    def forward(self, x):
        y = F.relu(self.bn1(self.c1(x)))
        y = self.bn2(self.c2(y))
        if self.c3 is not None:
            x = self.c3(x)
        y += x
        return F.relu(y)
    
class Residual_bottle(nn.Module):
    def __init__(self, num_channels, strides=1):
        super().__init__()
        self.c1 = nn.LazyConv2d(num_channels//4, kernel_size=1, stride=strides)
        self.c2 = nn.LazyConv2d(num_channels//4, kernel_size=3, padding=1)
        self.c3 = nn.LazyConv2d(num_channels, kernel_size=1)
        
        # use 1x1conv to make C,H,W the same
        self.c4 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)

            
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        
    def forward(self, x):
        y = F.relu(self.bn1(self.c1(x)))
        y = F.relu(self.bn2(self.c2(y)))
        y = self.bn3(self.c3(y)) 

        x = self.c4(x)
        y += x
        return F.relu(y)
    
class ResNet(nn.Module):
    def __init__(self, arch, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3), # b*3*224*224 -> b*64*112*112 
            nn.LazyBatchNorm2d(), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # -> b*64*56*56
        ) # the same as those of the GoogLeNet (besides the BatchNorm Layer)
        
        blk_list = []
        for i, b in enumerate(arch):
            blk_list.append(self.block(*b, first_blk=(i==0)))
        self.conv2345 = nn.Sequential(*blk_list)
        
        self.last = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )
    
    def block(self, num_residuals, num_channels, flag_bottle=False, first_blk=False):
        blk = []
        for i in range(num_residuals):
            if not flag_bottle:
                if i == 0 and not first_blk:
                    blk.append(Residual(num_channels, strides=2)) 
                else:
                    blk.append(Residual(num_channels))
            else:
                if i == 0 and not first_blk:
                    blk.append(Residual_bottle(num_channels, strides=2))
                else:
                    blk.append(Residual_bottle(num_channels))
        return nn.Sequential(*blk)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2345(x)
        x = self.last(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.net = ResNet(arch=((2, 64), (2, 128), (2, 256), (2, 512)), num_classes=num_classes)
    def forward(self, x):
        return self.net(x)


class ResNet34(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.net = ResNet(arch=((3, 64), (4, 128), (6, 256), (3, 512)), num_classes=num_classes)
    def forward(self, x):
        return self.net(x)

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.net = ResNet(arch=((3, 256, True), (4, 512, True), (6, 1024, True), (3, 2048, True)), num_classes=num_classes)
    def forward(self, x):
        return self.net(x)

class ResNet152(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.net = ResNet(arch=((3, 256, True), (8, 512, True), (36, 1024, True), (3, 2048, True)), num_classes=num_classes)
    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    x = torch.randn(64, 3, 224, 224)
    model = ResNet50()
    y = model(x)
    print(y.shape)
    # print(model)
    torch.onnx.export(model,x,"./onnx/ResNet50.onnx")
    
    # y = Residual(num_channels=10,strides=1)(x)
    # print(y.shape())