import torch.nn as nn
import torch

def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1),
        nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1),
        nn.ReLU(),
    )

class NiN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nin_block(96, kernel_size=11, strides=4, padding=0),  # b*3*224*224 -> b*96*56*56
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> b*96*26*26
            nin_block(256, kernel_size=5, strides=1, padding=2), # -> b*256*26*26
            nn.MaxPool2d(kernel_size=3, stride=2), # -> b*256*12*12
            nin_block(384, kernel_size=3, strides=1, padding=1), # ->b*384*12*12
            nn.MaxPool2d(kernel_size=3, stride=2), # -> b*384*5*5
            nn.Dropout(p=0.5),
            nin_block(10, kernel_size=3, strides=1, padding=1), # -> b*10*5*5
            nn.AdaptiveAvgPool2d((1, 1)),# -> b*10*1*1
            nn.Flatten(),
        )
        
    def forward(self, x):
        y = self.net(x)
        return y
    

if __name__ == '__main__':
    x = torch.randn(64, 3, 224, 224)
    model = NiN()
    y = model(x)
    print(y.shape)
    # torch.onnx.export(model,x,"./onnx/NiN.onnx")
    