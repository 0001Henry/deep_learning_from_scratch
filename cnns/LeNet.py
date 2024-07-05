import torch.nn as nn
import torch

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # b*1*28*28 -> b*6*28*28
            nn.Sigmoid(),   
            nn.AvgPool2d(kernel_size=2, stride=2),  # -> b*6*14*14
            nn.Conv2d(6, 16, kernel_size=5, padding=0), # -> b*16*10*10
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # -> b*16*5*5
            
            nn.Flatten(), # -> b*400            
            nn.Linear(400, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10), # -> b*10            
        )
        
    def forward(self, x):
        y = self.net(x)
        return y
    

if __name__ == '__main__':
    x = torch.randn(64, 1, 28, 28)
    model = LeNet5()
    y = model(x)
    print(y.shape)
    torch.onnx.export(model,x,"./onnx/LeNet5.onnx")