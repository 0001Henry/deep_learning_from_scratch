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
    

class LeNetplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 增加卷积层的深度和通道数
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # b*1*28*28 -> b*32*28*28
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # b*32*28*28 -> b*64*28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> b*64*14*14
            nn.BatchNorm2d(64),  # 增加 Batch Normalization
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # b*64*14*14 -> b*128*14*14
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # b*128*14*14 -> b*256*14*14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> b*256*7*7
            nn.BatchNorm2d(256),  # 增加 Batch Normalization
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # b*256*7*7 -> b*512*7*7
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # b*512*7*7 -> b*512*7*7
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # -> b*512*3*3
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0), # -> b*512*3*3
            nn.BatchNorm2d(512),  # 增加 Batch Normalization
            
            nn.Flatten(),  # -> b*4608
            nn.Linear(512 * 3 * 3, 1024), 
            nn.ReLU(),
            nn.Dropout(0.5),  # 增加 Dropout
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),  # 增加 Dropout
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, 10)  # -> b*10
        )
        
    def forward(self, x):
        return self.net(x)



if __name__ == '__main__':
    x = torch.randn(64, 1, 28, 28)
    model = LeNetplus()
    y = model(x)
    print(y.shape)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')
    # torch.onnx.export(model,x,"./onnx/LeNet5.onnx")