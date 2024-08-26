import torch.nn as nn
import torch

class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.b1 = nn.Sequential(
            nn.LazyConv2d(c1, kernel_size=1),
            nn.ReLU()
        )
        self.b2 = nn.Sequential(
            nn.LazyConv2d(c2[0], kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(c2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.b3 = nn.Sequential(
            nn.LazyConv2d(c3[0], kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(c3[1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.LazyConv2d(c4, kernel_size=1),
            nn.ReLU()
        )
    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        y = torch.cat([y1, y2, y3, y4], dim=1)
        return y

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b3 = nn.Sequential(
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b4 = nn.Sequential(
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b5 = nn.Sequential(
            Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),   # just as in NiN
            nn.Flatten(),
            # nn.Dropout(0.4),
        )
        self.fc = nn.Linear(1024, num_classes)
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.fc(x)
        return x
    
if __name__ == '__main__':
    x = torch.randn(64, 3, 224, 224)
    model = GoogLeNet(num_classes=1000)
    y = model(x)
    print(y.shape)
    # print(model)
    # torch.onnx.export(model,x,"./onnx/GoogLeNet.onnx")
    