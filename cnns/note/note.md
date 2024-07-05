## AlexNet (2012)
![alt text](assets/note/image-2.png)
![alt text](assets/note/image.png)
- several conv+pooling, ReLU, Dropout, overlapping, Data Augmentation, two GPUs
- go deeper than LeNet, launch a Deep Learning Boom


## VGG (2014)
![alt text](assets/note/image-1.png)
![alt text](assets/note/image-9.png)
- using conv_blocks
- smaller conv kernals(3*3) are useful
- go deeper


## NiN (2013)
![alt text](assets/note/image-3.png)
#### 1*1 conv
- nonlinearity across channels (feature recombination)
- aids the quest for fewer parameters

#### global average pooling
- replace giant fully connected layers, fewer parameters 


## GoogLenNet (2014)
![alt text](assets/note/image-6.png)
#### Inception Module: concatenate multi-branch, parallel conv_kernel
-  “we need to go deeper”
![alt text](assets/note/image-5.png)


## ResNet (2015 Kaiming He)
![alt text](assets/note/image-7.png)
- solve the problem of gradient vanishing and degradation in deep networks
- make it possible to train ultra-deep networks (At least it doesn't make the model worse) 
![alt text](assets/note/image-8.png)
![alt text](assets/note/image-10.png)