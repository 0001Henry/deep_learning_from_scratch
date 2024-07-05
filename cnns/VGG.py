import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1)) 
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, conv_arch, num_classes=10):
        super(VGG, self).__init__()
        conv_blocks = []
        for (num_convs, out_channels) in conv_arch:
            conv_blocks.append(vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(
            *conv_blocks,
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes)
        )
        
    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    vgg11 = VGG(conv_arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)), num_classes=1000)
    vgg16 = VGG(conv_arch=((2, 64), (2, 128), (3, 256), (3, 512), (3, 512)), num_classes=1000)

    x = torch.randn((1,3,224,224))
    y = vgg16(x)
    print(vgg16)
    print(y.shape)

    # torch.save(vgg11,'vgg.pth')
    # torch.onnx.export(vgg11,x,'./onnx/vgg11.onnx')


# class VGG(nn.Module):
#     def __init__(self):
#         super(VGG,self).__init__()
#         self.conv1_1 = nn.Conv2d(3,64,3,1,1)
#         self.conv1_2 = nn.Conv2d(64,64,3,1,1)
#         self.pool1 = nn.MaxPool2d(2)

#         self.conv2_1 = nn.Conv2d(64,128,3,1,1)
#         self.conv2_2 = nn.Conv2d(128,128,3,1,1)
#         self.pool2 = nn.MaxPool2d(2)

#         self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
#         self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
#         self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
#         self.pool3 = nn.MaxPool2d(2)

#         self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
#         self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.pool4 = nn.MaxPool2d(2)

#         self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.pool5 = nn.MaxPool2d(2)

#         self.fc1 = nn.Linear(7*7*512,4096)
#         self.fc2 = nn.Linear(4096, 4096)
#         self.classifier = nn.Linear(4096, 1000)

#     def forward(self,x):
#         x = F.relu(self.conv1_1(x))
#         x = F.relu(self.conv1_2(x))
#         x = self.pool1(x)

#         x = F.relu(self.conv2_1(x))
#         x = F.relu(self.conv2_2(x))
#         x = self.pool2(x)

#         x = F.relu(self.conv3_1(x))
#         x = F.relu(self.conv3_2(x))
#         x = F.relu(self.conv3_3(x))
#         x = self.pool3(x)

#         x = F.relu(self.conv4_1(x))
#         x = F.relu(self.conv4_2(x))
#         x = F.relu(self.conv4_3(x))
#         x = self.pool4(x)

#         x = F.relu(self.conv5_1(x))
#         x = F.relu(self.conv5_2(x))
#         x = F.relu(self.conv5_3(x))
#         x = self.pool5(x)

#         #print('conv5.shape',x.shape) #n*7*7*512
#         x = x.reshape(-1,7*7*512)
#         #print('conv5.shape',x.shape) #n*7*7*512

#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.classifier(x)

#         return x