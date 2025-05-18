# Convolutional Networks

fully connected layer会破坏图片的空间特征



filter

padding

set P=(K-1)/2 to make output have same size as input

input=W,filter: K, padding: P   -->   output: W=K+1+2P

downsample -->  减小图像尺寸/减小resolution的操作



**receptive field**: 

​	for convolution with kernel size K, each element in the output depends on a K*K receptive field

​	Each successive convolution adds K – 1 to the receptive field size with L layers the receptive field size is `1 + L * (K – 1)`

​	be careful: "receptive field in the input" vs "receptive field in the previous layer"



**stride**



![image-20250503140016864](D:\adolph\mark\image-20250503140016864.png)



![image-20250503141020564](D:\adolph\mark\image-20250503141020564.png)



**Batch Normalization**

在训练过程中，每一层的输入分布会不断变化（因为前面层的参数也在变），这会导致下一层不断需要重新适应新的分布，训练效率变低。

BN 的作用是：**把每层输入固定在一个稳定的分布**，加快训练速度



**analysis**

"flop": number of floating point operations (multiply + add)

​			= (number of output elements) * (ops per output element)

​			= $(C_{out}\times H'\times W')*(C_{in}\times K\times K)$



**AlexNet**

![image-20250503170116228](D:\adolph\mark\image-20250503170116228.png)

![image-20250503165957890](D:\adolph\mark\image-20250503165957890.png)



**VFNet**	---- bigger AlexNet

​	deeper networks work better  (and more computation)



**VGG**		deeper network

VGG design rules:

​	\- all conv are $3\times3$, stride 1, pad 1

​	\- all max pool are $2\times2$, stride 2

​	\- after pool, double #channels

第一条考虑因素：why use 3\*3 filters? 

如下图所示，两个3\*3的filter在实现相同field size的同时计算量和参数都更小。此外还可以在两层之间加ReLU层，可以比一个5\*5filter有更多的非线性操作

![image-20250503171634435](D:\adolph\mark\image-20250503171634435.png) 

第二三条考虑因素：

通过缩小图像尺寸并增加filter深度(channels)，这样在让参数量和占用内存更小的同时，使得<u>不同图像分辨度(resolution)下会有相同计算量</u>。（这样或许可以让计算资源利用更均衡、梯度传播更稳定）

![image-20250503172551556](D:\adolph\mark\image-20250503172551556.png)

****



**GoogLeNet**		focus on efficiency

- 创新点1: **Aggressive Stem**

Stem network 在起始阶段更快减少输入维度(downsample) (对比VGG: most of the compute was at the start)

- 创新点2: **Inception Module**![image-20250503173952573](D:\adolph\mark\image-20250503173952573.png)

- 创新点3: **Global Average Pooling**![image-20250503174208877](D:\adolph\mark\image-20250503174208877.png)

- 创新点4: **Auxiliary Classifier**![image-20250503174313231](D:\adolph\mark\image-20250503174313231.png)



**ResNet** 	Residual Networks

发明Batch Normalization之后，人们可以训练更深的神经网络，但人们发现，一味增加网络深度不再能产生更优秀的性能

ResNet让更深的网络实现更强的性能

![image-20250503174944992](D:\adolph\mark\image-20250503174944992.png)

- Uses regular design, like VGG: each residual block has two **3x3 conv** 
- Network is divided into stages: the first block of each stage **halves the resolution** (with stride-2 conv) and **doubles the number of channels**
- Uses the same **aggressive stem** as GoogLeNet to downsample the input 4x before applying residual blocks.
- Like GoogLeNet, no big fully-connected-layers: instead use **global average pooling** and a **single linear layer** at the end

- 参数更少，非线性操作数增加  ![image-20250503175924220](D:\adolph\mark\image-20250503175924220.png)



下图右表中，圆形大小表示参数量

![image-20250503180525767](D:\adolph\mark\image-20250503180525767.png)



**ResNeXt**

在Residual block中增加了并行的Bottleneck![image-20250503180726014](D:\adolph\mark\image-20250503180726014.png)

结果证明增加groups可以在保持计算复杂度不变的同时提高性能





![image-20250503182200506](D:\adolph\mark\image-20250503182200506.png)