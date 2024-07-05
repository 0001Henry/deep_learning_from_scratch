import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), # b*3*224*224 -> b*96*56*56
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # -> b*96*26*26
            nn.Conv2d(96, 256, kernel_size=5, padding=2), # -> b*256*26*26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # -> b*256*12*12
            nn.Conv2d(256, 384, kernel_size=3, padding=1), # ->b*384*12*12
            nn.Conv2d(384, 384, kernel_size=3, padding=1), # ->b*384*12*12
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # ->b*256*12*12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # -> b*256*5*5
            nn.Flatten(), # -> b*6400
            
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000), 
        )
        
    def forward(self, x):
        y = self.net(x)
        return y
    

if __name__ == '__main__':
    x = torch.randn(64, 3, 224, 224)
    model = AlexNet()
    y = model(x)
    print(y.shape)
    torch.onnx.export(model,x,"./onnx/AlexNet.onnx")
    