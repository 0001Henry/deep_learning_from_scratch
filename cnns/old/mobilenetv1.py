#coding:utf8
import time 
import torch
import torch.nn as nn
import torchvision.models as models
 
class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1,self).__init__()
        
        # 标准卷积
        def conv_bn(inp,oup,stride):
            return nn.Sequential(
                    nn.Conv2d(inp,oup,3,stride,1,bias = False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace = True))
        
        # 深度可分离卷积，depthwise convolution + pointwise convolution
        def conv_dw(inp,oup,stride):
            return nn.Sequential(
                    nn.Conv2d(inp,inp,3,stride,1,groups = inp,bias = False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace = True),
                    
                    nn.Conv2d(inp,oup,1,1,0,bias = False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace = True))
            
        #网络模型声明
        self.model = nn.Sequential(
                conv_bn(3,32,2),
                conv_dw(32,64,1),
                conv_dw(64,128,2),
                conv_dw(128,128,1),
                conv_dw(128,256,2),
                conv_dw(256,256,1),
                conv_dw(256,512,2),
                conv_dw(512,512,1),
                conv_dw(512,512,1),
                conv_dw(512,512,1),
                conv_dw(512,512,1),
                conv_dw(512,512,1),
                conv_dw(512,1024,2),
                conv_dw(1024,1024,1),
                nn.AvgPool2d(7),)
      
        self.fc = nn.Linear(1024,1000)
    
    #网络的前向过程    
    def forward(self,x):
        x = self.model(x)
        x = x.view(-1,1024)
        x = self.fc(x)
        return x

#速度评估
def speed(model,name):
    t0 = time.time()
    input = torch.rand(1,3,224,224).cpu()
    t1 = time.time()
    
    model(input)
    t2 = time.time()

    for i in range(0,30):
        model(input)
    t3 = time.time()
    
    print('%10s : %f'%(name,(t3 - t2)/30))
 
if __name__ == '__main__':
    resnet18 = models.resnet18().cpu()
    alexnet = models.alexnet().cpu()
    vgg16 = models.vgg16().cpu()
    mobilenetv1 = MobileNetV1().cpu()
    
    speed(resnet18,'resnet18')
    speed(alexnet,'alexnet')
    speed(vgg16,'vgg16')
    speed(mobilenetv1,'mobilenet')
