# Neural Networks

SVM无法对一些特殊情况分类

**Feature Transforms**

​	(x, y) -> ($r, \theta$)

​	color histogram 忽略位置信息，只看像素颜色

​	Histogram of Oriented Gradient (HOG) 忽略颜色信息，只看方向

​	Bag of Words (data driven) 在数据中随机提取patches汇总成一个codebook

​	可以对图像进行多种特征转换，把结果拼接到一起

​	对每种不同数据可能需要使用不同的特征转换方法，训练也是根据特征来优化



**Neural Networks**

直接根据原数据训练优化

相当于neural network自己先做特征提取然后分类

Fully-connected neural network (MLP)

activation function

​	如果没有激励函数，那么叠加多个层后结果仍然是一个linear classifier ($W_1W_2=W$)，叠加没意义

​	$ReLU(z)=max(0,z)$

​	sigmoid: $\sigma(x)=\frac{1}{1+e^{-x}}$

**Universal Approximation**

拟合函数时，ReLU表示的是函数的斜率

![image-20250429122756556](D:\adolph\mark\image-20250429122756556.png)

in practice, networks don't really learn bumps

convex function: $\text{A function }f:X\subseteq\mathbb{R}^N\to\mathbb{R}\text{ is convex if for all }x_1,x_2\in X,t\in[0,1],f(tx_1+(1-t)x_2)\leq tf(x_1)+(1-t)f(x_2)$

![image-20250429123514434](D:\adolph\mark\image-20250429123514434.png)

![image-20250429123635973](D:\adolph\mark\image-20250429123635973.png)

linear classifier can optimize convex function

but most neural network need nonconvex optimization!