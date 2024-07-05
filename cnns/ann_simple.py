import torch.nn as nn
import torch 
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == '__main__':
    x = torch.randn(64, 1, 28, 28)
    model = ANN()
    y = model(x)
    print(y.shape)
    torch.onnx.export(model,x,"./onnx/ANN.onnx")